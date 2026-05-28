"""Load real Qwen3.6 weights from the palmfuture GPTQ checkpoint into our
modeling code.

Mapping (verified from the safetensors index):

  ours                            <-  safetensors
  -----------------------------------------------------------------------
  model.embed_tokens.weight       <-  model.language_model.embed_tokens.weight
  model.norm.weight               <-  model.language_model.norm.weight
  lm_head.weight                  <-  lm_head.weight

  Per layer (prefix P = model.language_model.layers.{i}):
    layer.input_layernorm.weight        <-  P.input_layernorm.weight
    layer.post_attention_layernorm.weight <- P.post_attention_layernorm.weight
    layer.mlp.gate.weight                <-  P.mlp.gate.weight
    layer.mlp.shared_expert.{gate,up,down}_proj.weight <- P.mlp.shared_expert.*.weight
    layer.mlp.shared_expert_gate.weight  <-  P.mlp.shared_expert_gate.weight

    layer_type == "linear_attention":
      layer.attn.in_proj_qkv.weight      <-  P.linear_attn.in_proj_qkv.weight
      layer.attn.in_proj_z.weight        <-  P.linear_attn.in_proj_z.weight
      layer.attn.in_proj_b.weight        <-  P.linear_attn.in_proj_b.weight
      layer.attn.in_proj_a.weight        <-  P.linear_attn.in_proj_a.weight
      layer.attn.conv1d.weight           <-  P.linear_attn.conv1d.weight
      layer.attn.dt_bias                 <-  P.linear_attn.dt_bias
      layer.attn.A_log                   <-  P.linear_attn.A_log
      layer.attn.norm.weight             <-  P.linear_attn.norm.weight
      layer.attn.out_proj.weight         <-  P.linear_attn.out_proj.weight

    layer_type == "full_attention":
      layer.attn.q_proj.weight           <-  P.self_attn.q_proj.weight
      layer.attn.k_proj.weight           <-  P.self_attn.k_proj.weight
      layer.attn.v_proj.weight           <-  P.self_attn.v_proj.weight
      layer.attn.o_proj.weight           <-  P.self_attn.o_proj.weight
      layer.attn.q_norm.weight           <-  P.self_attn.q_norm.weight
      layer.attn.k_norm.weight           <-  P.self_attn.k_norm.weight

    For each routed expert e:
      layer.mlp.experts[e].{gate,up,down}_proj.weight  <- dequantize_gptq(
          P.mlp.experts.{e}.*.{qweight, qzeros, scales, g_idx})

The vision tower (model.visual.*) and MTP head (mtp.*) are not loaded
(text-only port).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from torch import nn

from .configuration_qwen36 import Qwen36Config
from .gptq_dequant import (
    build_shard_index,
    dequantize_linear_for_nn,
    load_gptq_linear,
    load_plain_tensor,
)
from .modeling_qwen36 import (
    QwenAttention,
    QwenDecoderLayer,
    QwenForCausalLM,
    QwenGatedDeltaNet,
)


def _set_param(module: nn.Module, name: str, tensor: torch.Tensor) -> None:
    """Set a parameter or buffer by attribute name without trigger autograd."""
    attr = getattr(module, name)
    with torch.no_grad():
        if tensor.shape != attr.shape:
            raise ValueError(f"shape mismatch for {name}: got {tuple(tensor.shape)}, expected {tuple(attr.shape)}")
        attr.data.copy_(tensor.to(attr.dtype))


def _load_safetensors_weight(
    weight_map: dict[str, str],
    model_dir: Path,
    name: str,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    return load_plain_tensor(weight_map, model_dir, name).to(out_dtype)


def load_layer_weights(
    layer: QwenDecoderLayer,
    layer_idx: int,
    weight_map: dict[str, str],
    model_dir: Path,
    out_dtype: torch.dtype = torch.float32,
    experts: Iterable[int] | None = None,
) -> None:
    """Populate one decoder layer's parameters from the safetensors checkpoint.

    `experts`: optional iterable of expert indices to load (default: all 256).
    Useful for testing with a subset.
    """
    P = f"model.language_model.layers.{layer_idx}"

    # Per-layer norms
    _set_param(
        layer.input_layernorm, "weight",
        _load_safetensors_weight(weight_map, model_dir, f"{P}.input_layernorm.weight", out_dtype),
    )
    _set_param(
        layer.post_attention_layernorm, "weight",
        _load_safetensors_weight(weight_map, model_dir, f"{P}.post_attention_layernorm.weight", out_dtype),
    )

    # Attention block
    if layer.layer_type == "linear_attention":
        att: QwenGatedDeltaNet = layer.attn  # type: ignore[assignment]
        ln_prefix = f"{P}.linear_attn"
        _set_param(att.in_proj_qkv, "weight", _load_safetensors_weight(weight_map, model_dir, f"{ln_prefix}.in_proj_qkv.weight", out_dtype))
        _set_param(att.in_proj_z, "weight", _load_safetensors_weight(weight_map, model_dir, f"{ln_prefix}.in_proj_z.weight", out_dtype))
        _set_param(att.in_proj_b, "weight", _load_safetensors_weight(weight_map, model_dir, f"{ln_prefix}.in_proj_b.weight", out_dtype))
        _set_param(att.in_proj_a, "weight", _load_safetensors_weight(weight_map, model_dir, f"{ln_prefix}.in_proj_a.weight", out_dtype))
        # conv1d weight in the checkpoint is (conv_dim, 1, kernel_size); nn.Conv1d (depthwise) wants the same.
        _set_param(att.conv1d, "weight", _load_safetensors_weight(weight_map, model_dir, f"{ln_prefix}.conv1d.weight", out_dtype))
        _set_param(att, "dt_bias", _load_safetensors_weight(weight_map, model_dir, f"{ln_prefix}.dt_bias", out_dtype))
        _set_param(att, "A_log", _load_safetensors_weight(weight_map, model_dir, f"{ln_prefix}.A_log", out_dtype))
        _set_param(att.norm, "weight", _load_safetensors_weight(weight_map, model_dir, f"{ln_prefix}.norm.weight", out_dtype))
        _set_param(att.out_proj, "weight", _load_safetensors_weight(weight_map, model_dir, f"{ln_prefix}.out_proj.weight", out_dtype))
    elif layer.layer_type == "full_attention":
        att_full: QwenAttention = layer.attn  # type: ignore[assignment]
        sa_prefix = f"{P}.self_attn"
        _set_param(att_full.q_proj, "weight", _load_safetensors_weight(weight_map, model_dir, f"{sa_prefix}.q_proj.weight", out_dtype))
        _set_param(att_full.k_proj, "weight", _load_safetensors_weight(weight_map, model_dir, f"{sa_prefix}.k_proj.weight", out_dtype))
        _set_param(att_full.v_proj, "weight", _load_safetensors_weight(weight_map, model_dir, f"{sa_prefix}.v_proj.weight", out_dtype))
        _set_param(att_full.o_proj, "weight", _load_safetensors_weight(weight_map, model_dir, f"{sa_prefix}.o_proj.weight", out_dtype))
        _set_param(att_full.q_norm, "weight", _load_safetensors_weight(weight_map, model_dir, f"{sa_prefix}.q_norm.weight", out_dtype))
        _set_param(att_full.k_norm, "weight", _load_safetensors_weight(weight_map, model_dir, f"{sa_prefix}.k_norm.weight", out_dtype))
    else:
        raise ValueError(f"unknown layer_type {layer.layer_type}")

    # MoE non-expert pieces
    _set_param(layer.mlp.gate, "weight", _load_safetensors_weight(weight_map, model_dir, f"{P}.mlp.gate.weight", out_dtype))
    _set_param(layer.mlp.shared_expert.gate_proj, "weight", _load_safetensors_weight(weight_map, model_dir, f"{P}.mlp.shared_expert.gate_proj.weight", out_dtype))
    _set_param(layer.mlp.shared_expert.up_proj, "weight", _load_safetensors_weight(weight_map, model_dir, f"{P}.mlp.shared_expert.up_proj.weight", out_dtype))
    _set_param(layer.mlp.shared_expert.down_proj, "weight", _load_safetensors_weight(weight_map, model_dir, f"{P}.mlp.shared_expert.down_proj.weight", out_dtype))
    _set_param(layer.mlp.shared_expert_gate, "weight", _load_safetensors_weight(weight_map, model_dir, f"{P}.mlp.shared_expert_gate.weight", out_dtype))

    # Routed experts (dequant from GPTQ)
    expert_indices = list(range(len(layer.mlp.experts))) if experts is None else list(experts)
    for e in expert_indices:
        expert = layer.mlp.experts[e]
        for proj in ("gate_proj", "up_proj", "down_proj"):
            w = load_gptq_linear(
                weight_map, model_dir,
                prefix=f"{P}.mlp.experts.{e}.{proj}",
                out_dtype=out_dtype,
            )
            _set_param(getattr(expert, proj), "weight", w)


def load_global_weights(
    model: QwenForCausalLM,
    weight_map: dict[str, str],
    model_dir: Path,
    out_dtype: torch.dtype = torch.float32,
) -> None:
    """Load the non-layer tensors (embedding, final norm, lm_head)."""
    _set_param(
        model.model.embed_tokens, "weight",
        _load_safetensors_weight(weight_map, model_dir, "model.language_model.embed_tokens.weight", out_dtype),
    )
    _set_param(
        model.model.norm, "weight",
        _load_safetensors_weight(weight_map, model_dir, "model.language_model.norm.weight", out_dtype),
    )
    _set_param(
        model.lm_head, "weight",
        _load_safetensors_weight(weight_map, model_dir, "lm_head.weight", out_dtype),
    )


def load_real_model(
    model_dir: str | Path,
    out_dtype: torch.dtype = torch.float32,
    layers: Iterable[int] | None = None,
    experts_per_layer: Iterable[int] | None = None,
) -> QwenForCausalLM:
    """Build a QwenForCausalLM from the real config and populate the
    requested layers + experts.

    Memory: loading the full 40-layer model with all 256 experts in fp32 will
    consume well over 64GB. For testing, pass a subset of layers.
    """
    model_dir = Path(model_dir)
    config = Qwen36Config.from_pretrained_dir(model_dir)
    weight_map = build_shard_index(model_dir)

    # Build the model (random init) -- only allocate layers we'll touch
    # if a subset was requested. (We still build the full architecture so
    # forward() works end-to-end; the untouched layers carry their random
    # weights and shouldn't be relied upon.)
    model = QwenForCausalLM(config)
    model.eval()

    load_global_weights(model, weight_map, model_dir, out_dtype=out_dtype)

    layer_indices = list(range(config.num_hidden_layers)) if layers is None else list(layers)
    for i in layer_indices:
        load_layer_weights(
            model.model.layers[i], i, weight_map, model_dir,
            out_dtype=out_dtype, experts=experts_per_layer,
        )
    return model
