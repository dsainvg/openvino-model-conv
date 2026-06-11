"""Load real Qwen3.5-4B weights from a HuggingFace safetensors checkpoint
into our modeling code.

Mapping (verified from the safetensors index and supports variations):

  ours                            <-  safetensors candidates
  -----------------------------------------------------------------------
  model.embed_tokens.weight       <-  model.language_model.embed_tokens.weight / language_model.model.embed_tokens.weight / model.embed_tokens.weight
  model.norm.weight               <-  model.language_model.norm.weight / language_model.model.norm.weight / model.norm.weight
  lm_head.weight                  <-  lm_head.weight / model.language_model.lm_head.weight / language_model.lm_head.weight

  Per layer (prefix P = model.language_model.layers.{i} / language_model.model.layers.{i} / model.layers.{i}):
    layer.input_layernorm.weight        <-  P.input_layernorm.weight
    layer.post_attention_layernorm.weight <- P.post_attention_layernorm.weight
    layer.mlp.gate_proj.weight          <-  P.mlp.gate_proj.weight
    layer.mlp.up_proj.weight            <-  P.mlp.up_proj.weight
    layer.mlp.down_proj.weight          <-  P.mlp.down_proj.weight

    layer_type == "linear_attention":
      layer.attn.in_proj_qkv.weight     <-  P.linear_attn.in_proj_qkv.weight
      layer.attn.in_proj_z.weight       <-  P.linear_attn.in_proj_z.weight
      layer.attn.in_proj_b.weight       <-  P.linear_attn.in_proj_b.weight
      layer.attn.in_proj_a.weight       <-  P.linear_attn.in_proj_a.weight
      layer.attn.conv1d.weight          <-  P.linear_attn.conv1d.weight
      layer.attn.dt_bias                <-  P.linear_attn.dt_bias
      layer.attn.A_log                  <-  P.linear_attn.A_log
      layer.attn.norm.weight            <-  P.linear_attn.norm.weight
      layer.attn.out_proj.weight        <-  P.linear_attn.out_proj.weight

    layer_type == "full_attention":
      layer.attn.q_proj.weight          <-  P.self_attn.q_proj.weight
      layer.attn.k_proj.weight          <-  P.self_attn.k_proj.weight
      layer.attn.v_proj.weight          <-  P.self_attn.v_proj.weight
      layer.attn.o_proj.weight          <-  P.self_attn.o_proj.weight
      layer.attn.q_norm.weight          <-  P.self_attn.q_norm.weight
      layer.attn.k_norm.weight          <-  P.self_attn.k_norm.weight

The vision tower (model.visual.*) is not loaded (text-only port).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import torch
from torch import nn

from .configuration import Qwen35Config
from .modeling import (
    QwenAttention,
    QwenDecoderLayer,
    QwenForCausalLM,
    QwenGatedDeltaNet,
)


def _set_param(module: nn.Module, name: str, tensor: torch.Tensor) -> None:
    """Set a parameter or buffer by attribute name without triggering autograd."""
    attr = getattr(module, name)
    with torch.no_grad():
        if tensor.shape != attr.shape:
            raise ValueError(f"shape mismatch for {name}: got {tuple(tensor.shape)}, expected {tuple(attr.shape)}")
        if isinstance(attr, nn.Parameter):
            setattr(module, name, nn.Parameter(tensor, requires_grad=attr.requires_grad))
        else:
            setattr(module, name, tensor)


def load_plain_tensor(
    weight_map: dict[str, str],
    model_dir: Path | str,
    name: str,
) -> torch.Tensor:
    """Load a non-quantized tensor (BF16/FP16/etc) by full dotted name."""
    from safetensors import safe_open

    model_dir = Path(model_dir)
    shard = model_dir / weight_map[name]
    with safe_open(str(shard), framework="pt") as fh:
        return fh.get_tensor(name).clone()


def _load_safetensors_weight(
    weight_map: dict[str, str],
    model_dir: Path,
    name: str,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    tensor = load_plain_tensor(weight_map, model_dir, name)
    if tensor.dtype != out_dtype:
        return tensor.to(out_dtype)
    return tensor


def build_shard_index(model_dir: Path | str) -> dict[str, str]:
    """Read model.safetensors.index.json → {tensor_name: shard_filename}."""
    model_dir = Path(model_dir)
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        # Single-shard model — build a trivial map
        shard = model_dir / "model.safetensors"
        if not shard.exists():
            raise FileNotFoundError(f"No safetensors index or single shard found in {model_dir}")
        from safetensors import safe_open
        with safe_open(str(shard), framework="pt") as fh:
            return {k: "model.safetensors" for k in fh.keys()}
    with open(index_path, encoding="utf-8") as fh:
        return json.load(fh)["weight_map"]


def _find_key(weight_map: dict[str, str], suffix: str, candidates: list[str]) -> str:
    for c in candidates:
        if c in weight_map:
            return c
    for k in weight_map:
        if k.endswith(suffix) and ".layers." not in k and "attn." not in k and "visual." not in k:
            return k
    raise KeyError(f"Could not find weight key ending with '{suffix}' in weight_map")


def _find_layer_prefix(weight_map: dict[str, str], layer_idx: int) -> str:
    candidates = [
        f"model.language_model.layers.{layer_idx}",
        f"language_model.model.layers.{layer_idx}",
        f"model.layers.{layer_idx}",
    ]
    for c in candidates:
        prefix_dot = c + "."
        if any(k.startswith(prefix_dot) for k in weight_map):
            return c
    # Fallback search: find any key containing f".layers.{layer_idx}."
    target = f".layers.{layer_idx}."
    for k in weight_map:
        idx = k.find(target)
        if idx != -1:
            return k[:idx + len(target) - 1]
    raise KeyError(f"Could not find layer prefix for layer {layer_idx} in weight_map")


def load_linear_weight(
    weight_map: dict[str, str],
    model_dir: Path,
    prefix: str,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Load a linear layer's weight. If GPTQ int4 quantized, dequantize on-the-fly."""
    if f"{prefix}.qweight" in weight_map:
        from .gptq_dequant import load_gptq_linear
        return load_gptq_linear(weight_map, model_dir, prefix, out_dtype=out_dtype)
    else:
        return _load_safetensors_weight(weight_map, model_dir, f"{prefix}.weight", out_dtype)


def load_layer_weights(
    layer: QwenDecoderLayer,
    layer_idx: int,
    weight_map: dict[str, str],
    model_dir: Path,
    out_dtype: torch.dtype = torch.float32,
) -> None:
    """Populate one decoder layer's parameters from the safetensors checkpoint."""
    P = _find_layer_prefix(weight_map, layer_idx)

    # Per-layer norms (always plain tensors, never quantized)
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
        lp = f"{P}.linear_attn"
        _set_param(att.in_proj_qkv, "weight", load_linear_weight(weight_map, model_dir, f"{lp}.in_proj_qkv", out_dtype))
        _set_param(att.in_proj_z,   "weight", load_linear_weight(weight_map, model_dir, f"{lp}.in_proj_z", out_dtype))
        _set_param(att.in_proj_b,   "weight", load_linear_weight(weight_map, model_dir, f"{lp}.in_proj_b", out_dtype))
        _set_param(att.in_proj_a,   "weight", load_linear_weight(weight_map, model_dir, f"{lp}.in_proj_a", out_dtype))
        _set_param(att.conv1d,      "weight", _load_safetensors_weight(weight_map, model_dir, f"{lp}.conv1d.weight", out_dtype))
        _set_param(att, "dt_bias",            _load_safetensors_weight(weight_map, model_dir, f"{lp}.dt_bias", out_dtype))
        _set_param(att, "A_log",              _load_safetensors_weight(weight_map, model_dir, f"{lp}.A_log", out_dtype))
        _set_param(att.norm,        "weight", _load_safetensors_weight(weight_map, model_dir, f"{lp}.norm.weight", out_dtype))
        _set_param(att.out_proj,    "weight", load_linear_weight(weight_map, model_dir, f"{lp}.out_proj", out_dtype))

    elif layer.layer_type == "full_attention":
        att_full: QwenAttention = layer.attn  # type: ignore[assignment]
        sp = f"{P}.self_attn"
        _set_param(att_full.q_proj, "weight", load_linear_weight(weight_map, model_dir, f"{sp}.q_proj", out_dtype))
        _set_param(att_full.k_proj, "weight", load_linear_weight(weight_map, model_dir, f"{sp}.k_proj", out_dtype))
        _set_param(att_full.v_proj, "weight", load_linear_weight(weight_map, model_dir, f"{sp}.v_proj", out_dtype))
        _set_param(att_full.o_proj, "weight", load_linear_weight(weight_map, model_dir, f"{sp}.o_proj", out_dtype))
        _set_param(att_full.q_norm, "weight", _load_safetensors_weight(weight_map, model_dir, f"{sp}.q_norm.weight", out_dtype))
        _set_param(att_full.k_norm, "weight", _load_safetensors_weight(weight_map, model_dir, f"{sp}.k_norm.weight", out_dtype))

    else:
        raise ValueError(f"Unknown layer_type {layer.layer_type!r}")

    # MLP
    _set_param(layer.mlp.gate_proj, "weight", load_linear_weight(weight_map, model_dir, f"{P}.mlp.gate_proj", out_dtype))
    _set_param(layer.mlp.up_proj,   "weight", load_linear_weight(weight_map, model_dir, f"{P}.mlp.up_proj", out_dtype))
    _set_param(layer.mlp.down_proj, "weight", load_linear_weight(weight_map, model_dir, f"{P}.mlp.down_proj", out_dtype))


def load_global_weights(
    model: QwenForCausalLM,
    weight_map: dict[str, str],
    model_dir: Path,
    out_dtype: torch.dtype = torch.float32,
) -> None:
    """Load the non-layer tensors (embedding, final norm, lm_head)."""
    embed_candidates = [
        "model.language_model.embed_tokens.weight",
        "language_model.model.embed_tokens.weight",
        "model.embed_tokens.weight",
    ]
    norm_candidates = [
        "model.language_model.norm.weight",
        "language_model.model.norm.weight",
        "model.norm.weight",
    ]
    lm_head_candidates = [
        "lm_head.weight",
        "model.language_model.lm_head.weight",
        "language_model.lm_head.weight",
        "lm_head.qweight",
        "model.language_model.lm_head.qweight",
        "language_model.lm_head.qweight",
    ]

    embed_key = _find_key(weight_map, "embed_tokens.weight", embed_candidates)
    norm_key = _find_key(weight_map, "norm.weight", norm_candidates)
    try:
        lm_head_key = _find_key(weight_map, "lm_head.weight", lm_head_candidates)
    except KeyError:
        lm_head_key = _find_key(weight_map, "lm_head.qweight", lm_head_candidates)

    _set_param(
        model.model.embed_tokens, "weight",
        _load_safetensors_weight(weight_map, model_dir, embed_key, out_dtype),
    )
    _set_param(
        model.model.norm, "weight",
        _load_safetensors_weight(weight_map, model_dir, norm_key, out_dtype),
    )

    if lm_head_key.endswith(".qweight"):
        lm_head_prefix = lm_head_key[:-8]
    elif lm_head_key.endswith(".weight"):
        lm_head_prefix = lm_head_key[:-7]
    else:
        lm_head_prefix = lm_head_key

    _set_param(
        model.lm_head, "weight",
        load_linear_weight(weight_map, model_dir, lm_head_prefix, out_dtype),
    )


def load_real_model(
    model_dir: str | Path,
    out_dtype: torch.dtype = torch.float32,
    layers: Iterable[int] | None = None,
) -> QwenForCausalLM:
    """Build QwenForCausalLM from the real config and load all weights."""
    model_dir  = Path(model_dir)
    config     = Qwen35Config.from_pretrained_dir(model_dir)
    weight_map = build_shard_index(model_dir)

    model = QwenForCausalLM(config)
    model.eval()

    load_global_weights(model, weight_map, model_dir, out_dtype=out_dtype)

    layer_indices = list(range(config.num_hidden_layers)) if layers is None else list(layers)
    for i in layer_indices:
        print(f"  loading layer {i+1}/{config.num_hidden_layers} ...", end="\r")
        load_layer_weights(model.model.layers[i], i, weight_map, model_dir, out_dtype=out_dtype)
    print()

    return model
