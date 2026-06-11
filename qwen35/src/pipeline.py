"""Autoregressive decode pipeline for Qwen3.5-4B.

Drives the QwenForCausalLM model through a prompt-prefill + decode loop.
The pure-torch path works with both toy (random-weight) and real models.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from .modeling import QwenForCausalLM


@dataclass
class GenerationResult:
    token_ids: list[int]
    steps: int = 0
    seconds: float = 0.0

    @property
    def tokens_per_second(self) -> float:
        return self.steps / self.seconds if self.seconds > 0 else 0.0


class Qwen35Pipeline:
    """Single-token decode pipeline driven by QwenForCausalLM.

    Args:
        model: a QwenForCausalLM with weights already loaded.
    """

    def __init__(self, model: QwenForCausalLM):
        self.model = model

    @torch.no_grad()
    def step(
        self,
        input_ids: torch.Tensor,    # (1, 1)
        position_ids: torch.Tensor, # (1, 1)
        state: dict,
    ) -> tuple[torch.Tensor, dict]:
        logits, k_out, v_out, conv_out, rec_out = self.model(
            input_ids,
            position_ids,
            state["k_caches"],
            state["v_caches"],
            state["conv_states"],
            state["rec_states"],
        )
        new_state = {
            "k_caches":   k_out,
            "v_caches":   v_out,
            "conv_states": conv_out,
            "rec_states":  rec_out,
        }
        return logits, new_state

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: list[int] | torch.Tensor,
        max_new_tokens: int = 32,
        max_seq: int | None = None,
        greedy: bool = True,
        temperature: float = 1.0,
        eos_token_id: int | None = None,
        state_dtype: torch.dtype = torch.float32,
    ) -> GenerationResult:
        if isinstance(prompt_ids, torch.Tensor):
            prompt = prompt_ids.flatten().tolist()
        else:
            prompt = list(prompt_ids)

        if max_seq is None:
            max_seq = len(prompt) + max_new_tokens + 1

        state     = self.model.empty_state(batch=1, max_seq=max_seq, dtype=state_dtype)
        generated: list[int] = []
        t0 = time.time()
        steps = 0

        # Prefill: consume prompt tokens one at a time
        logits = None
        for pos, tok in enumerate(prompt):
            logits, state = self.step(
                torch.tensor([[tok]]), torch.tensor([[pos]]), state
            )
            steps += 1

        # Decode loop
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
