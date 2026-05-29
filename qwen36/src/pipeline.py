"""Phase 2.5: autoregressive decode pipeline for Qwen3.6.

Ties together the split inference seam (router-only backbone per layer +
externally dispatched experts) and the ExpertManager LRU cache into a single
`generate()` loop. Works on the pure-torch model (toy or real); the per-expert
callables can later be swapped for OV-compiled experts without changing this
file, since the ExpertManager only requires `expert(x) -> y`.

The model is decode-only (seq_len=1), so a prompt of length L is consumed by
running L single-token steps to fill the state, then new tokens are generated
one at a time.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .expert_manager import ExpertFrequency, ExpertManager
from .modeling_qwen36 import QwenForCausalLM
from .split_inference import combine_layer_output, moe_router_step


@dataclass
class GenerationResult:
    token_ids: list[int]
    expert_freq: ExpertFrequency | None = None
    cache_stats_str: str | None = None
    steps: int = 0
    seconds: float = 0.0

    @property
    def tokens_per_second(self) -> float:
        return self.steps / self.seconds if self.seconds > 0 else 0.0


class Qwen36Pipeline:
    """Drives split-path decoding through an ExpertManager.

    Args:
        model: a QwenForCausalLM (weights already loaded).
        expert_manager: optional. Defaults to an unbounded manager that pulls
            experts straight from the model's modules (no real eviction --
            fine for toy / when everything is already resident).
        collect_freq: if True, record per-layer expert activation counts.
    """

    def __init__(
        self,
        model: QwenForCausalLM,
        expert_manager: ExpertManager | None = None,
        collect_freq: bool = False,
    ):
        self.model = model
        self.cfg = model.config
        if expert_manager is None:
            expert_manager = ExpertManager(self._module_loader, capacity=None)
        self.experts = expert_manager
        self.expert_freq = ExpertFrequency() if collect_freq else None

    def _module_loader(self, layer_idx: int, expert_idx: int):
        return self.model.model.layers[layer_idx].mlp.experts[expert_idx]

    @torch.no_grad()
    def step(
        self,
        input_ids: torch.Tensor,  # (1, 1)
        position_ids: torch.Tensor,  # (1, 1)
        state: dict,
    ) -> tuple[torch.Tensor, dict]:
        mdl = self.model.model
        x = mdl.embed_tokens(input_ids)
        cos = mdl.rope_cos[position_ids.long()].to(x.dtype)
        sin = mdl.rope_sin[position_ids.long()].to(x.dtype)
        write_pos = position_ids[0, 0]

        k_caches = list(state["k_caches"])
        v_caches = list(state["v_caches"])
        conv_states = list(state["conv_states"])
        rec_states = list(state["rec_states"])

        full_i = lin_i = 0
        for layer_idx, layer in enumerate(mdl.layers):
            if layer.layer_type == "full_attention":
                h = layer.input_layernorm(x)
                attn_out, k_new, v_new = layer.attn(
                    h, cos, sin, k_caches[full_i], v_caches[full_i], write_pos
                )
                k_caches[full_i] = k_new
                v_caches[full_i] = v_new
                full_i += 1
            else:
                h = layer.input_layernorm(x)
                attn_out, conv_new, rec_new = layer.attn(h, conv_states[lin_i], rec_states[lin_i])
                conv_states[lin_i] = conv_new
                rec_states[lin_i] = rec_new
                lin_i += 1
            x_post_attn = x + attn_out

            h2 = layer.post_attention_layernorm(x_post_attn)
            flat, topk_idx, topk_w, shared = moe_router_step(layer.mlp, h2)
            if self.expert_freq is not None:
                self.expert_freq.add(layer_idx, topk_idx)

            BS, K = topk_idx.shape
            expert_outs = torch.zeros(BS, K, flat.shape[-1], dtype=flat.dtype)
            for t in range(BS):
                for k in range(K):
                    e_idx = int(topk_idx[t, k].item())
                    expert_outs[t, k] = self.experts.run(layer_idx, e_idx, flat[t : t + 1]).squeeze(0)

            x = combine_layer_output(x_post_attn, expert_outs, topk_w, shared)

        x = mdl.norm(x)
        logits = self.model.lm_head(x)
        new_state = {
            "k_caches": k_caches, "v_caches": v_caches,
            "conv_states": conv_states, "rec_states": rec_states,
        }
        return logits, new_state

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: list[int] | torch.Tensor,
        max_new_tokens: int = 16,
        max_seq: int | None = None,
        greedy: bool = True,
        temperature: float = 1.0,
        eos_token_id: int | None = None,
        state_dtype: torch.dtype = torch.float32,
    ) -> GenerationResult:
        import time

        if isinstance(prompt_ids, torch.Tensor):
            prompt = prompt_ids.flatten().tolist()
        else:
            prompt = list(prompt_ids)

        if max_seq is None:
            max_seq = len(prompt) + max_new_tokens + 1

        state = self.model.empty_state(batch=1, max_seq=max_seq, dtype=state_dtype)
        generated: list[int] = []
        t0 = time.time()
        steps = 0

        # Prefill: consume prompt tokens one at a time.
        logits = None
        for pos, tok in enumerate(prompt):
            logits, state = self.step(
                torch.tensor([[tok]]), torch.tensor([[pos]]), state
            )
            steps += 1

        # Decode loop.
        next_pos = len(prompt)
        for _ in range(max_new_tokens):
            next_tok = self._sample(logits, greedy, temperature)
            generated.append(next_tok)
            if eos_token_id is not None and next_tok == eos_token_id:
                break
            logits, state = self.step(
                torch.tensor([[next_tok]]), torch.tensor([[next_pos]]), state
            )
            steps += 1
            next_pos += 1

        return GenerationResult(
            token_ids=generated,
            expert_freq=self.expert_freq,
            cache_stats_str=str(self.experts.stats),
            steps=steps,
            seconds=time.time() - t0,
        )

    @staticmethod
    def _sample(logits: torch.Tensor, greedy: bool, temperature: float) -> int:
        last = logits[0, -1].float()
        if greedy or temperature <= 0:
            return int(last.argmax().item())
        probs = torch.softmax(last / temperature, dim=-1)
        return int(torch.multinomial(probs, 1).item())
