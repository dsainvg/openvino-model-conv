"""Load real Qwen3.5-4B weights from a HuggingFace safetensors checkpoint
into our modeling code.

Qwen3.5-4B is a VLM (Qwen3_5ForConditionalGeneration). The text backbone
weights live under the prefix  language_model.  inside the safetensors shards.

Key mapping (our param → checkpoint key):
  model.embed_tokens.weight  ←  language_model.model.embed_tokens.weight
  model.norm.weight          ←  language_model.model.norm.weight
  lm_head.weight             ←  language_model.lm_head.weight

Per layer (prefix P = language_model.model.layers.{i}):
  input_layernorm.weight           ←  P.input_layernorm.weight
  post_attention_layernorm.weight  ←  P.post_attention_layernorm.weight

  layer_type == "linear_attention":
    attn.in_proj_qkv.weight  ←  P.linear_attn.in_proj_qkv.weight
    attn.in_proj_z.weight    ←  P.linear_attn.in_proj_z.weight
    attn.in_proj_b.weight    ←  P.linear_attn.in_proj_b.weight
    attn.in_proj_a.weight    ←  P.linear_attn.in_proj_a.weight
    attn.conv1d.weight       ←  P.linear_attn.conv1d.weight
    attn.dt_bias             ←  P.linear_attn.dt_bias
    attn.A_log               ←  P.linear_attn.A_log
    attn.norm.weight         ←  P.linear_attn.norm.weight
    attn.out_proj.weight     ←  P.linear_attn.out_proj.weight

  layer_type == "full_attention":
    attn.q_proj.weight       ←  P.self_attn.q_proj.weight
    attn.k_proj.weight       ←  P.self_attn.k_proj.weight
    attn.v_proj.weight       ←  P.self_attn.v_proj.weight
    attn.o_proj.weight       ←  P.self_attn.o_proj.weight
    attn.q_norm.weight       ←  P.self_attn.q_norm.weight
    attn.k_norm.weight       ←  P.self_attn.k_norm.weight

  mlp.gate_proj.weight       ←  P.mlp.gate_proj.weight
  mlp.up_proj.weight         ←  P.mlp.up_proj.weight
  mlp.down_proj.weight       ←  P.mlp.down_proj.weight

Vision tower (language_model.visual.*) is not loaded — text-only port.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from .configuration import Qwen35Config
from .modeling import QwenAttention, QwenDecoderLayer, QwenForCausalLM, QwenGatedDeltaNet


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _set_param(module: nn.Module, name: str, tensor: torch.Tensor) -> None:
    attr = getattr(module, name)
    with torch.no_grad():
        if tensor.shape != attr.shape:
            raise ValueError(
                f"shape mismatch for {name}: got {tuple(tensor.shape)}, "
                f"expected {tuple(attr.shape)}"
            )
        attr.data.copy_(tensor.to(attr.dtype))


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


def load_tensor(
    weight_map: dict[str, str],
    model_dir: Path,
    name: str,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Load a BF16/FP16 tensor from the appropriate shard."""
    from safetensors import safe_open

    shard = model_dir / weight_map[name]
    with safe_open(str(shard), framework="pt") as fh:
        return fh.get_tensor(name).to(out_dtype)


# ---------------------------------------------------------------------------
# Per-layer weight loader
# ---------------------------------------------------------------------------


def load_layer_weights(
    layer: QwenDecoderLayer,
    layer_idx: int,
    weight_map: dict[str, str],
    model_dir: Path,
    out_dtype: torch.dtype = torch.float32,
) -> None:
    P = f"language_model.model.layers.{layer_idx}"

    def _load(key: str) -> torch.Tensor:
        return load_tensor(weight_map, model_dir, key, out_dtype)

    _set_param(layer.input_layernorm,          "weight", _load(f"{P}.input_layernorm.weight"))
    _set_param(layer.post_attention_layernorm, "weight", _load(f"{P}.post_attention_layernorm.weight"))

    if layer.layer_type == "linear_attention":
        att: QwenGatedDeltaNet = layer.attn  # type: ignore[assignment]
        lp = f"{P}.linear_attn"
        _set_param(att.in_proj_qkv, "weight", _load(f"{lp}.in_proj_qkv.weight"))
        _set_param(att.in_proj_z,   "weight", _load(f"{lp}.in_proj_z.weight"))
        _set_param(att.in_proj_b,   "weight", _load(f"{lp}.in_proj_b.weight"))
        _set_param(att.in_proj_a,   "weight", _load(f"{lp}.in_proj_a.weight"))
        _set_param(att.conv1d,      "weight", _load(f"{lp}.conv1d.weight"))
        _set_param(att, "dt_bias",           _load(f"{lp}.dt_bias"))
        _set_param(att, "A_log",             _load(f"{lp}.A_log"))
        _set_param(att.norm,        "weight", _load(f"{lp}.norm.weight"))
        _set_param(att.out_proj,    "weight", _load(f"{lp}.out_proj.weight"))

    elif layer.layer_type == "full_attention":
        att_full: QwenAttention = layer.attn  # type: ignore[assignment]
        sp = f"{P}.self_attn"
        _set_param(att_full.q_proj,  "weight", _load(f"{sp}.q_proj.weight"))
        _set_param(att_full.k_proj,  "weight", _load(f"{sp}.k_proj.weight"))
        _set_param(att_full.v_proj,  "weight", _load(f"{sp}.v_proj.weight"))
        _set_param(att_full.o_proj,  "weight", _load(f"{sp}.o_proj.weight"))
        _set_param(att_full.q_norm,  "weight", _load(f"{sp}.q_norm.weight"))
        _set_param(att_full.k_norm,  "weight", _load(f"{sp}.k_norm.weight"))

    else:
        raise ValueError(f"Unknown layer_type {layer.layer_type!r}")

    _set_param(layer.mlp.gate_proj, "weight", _load(f"{P}.mlp.gate_proj.weight"))
    _set_param(layer.mlp.up_proj,   "weight", _load(f"{P}.mlp.up_proj.weight"))
    _set_param(layer.mlp.down_proj, "weight", _load(f"{P}.mlp.down_proj.weight"))


# ---------------------------------------------------------------------------
# Global weight loader (embed + norm + lm_head)
# ---------------------------------------------------------------------------


def load_global_weights(
    model: QwenForCausalLM,
    weight_map: dict[str, str],
    model_dir: Path,
    out_dtype: torch.dtype = torch.float32,
) -> None:
    def _load(key: str) -> torch.Tensor:
        return load_tensor(weight_map, model_dir, key, out_dtype)

    _set_param(model.model.embed_tokens, "weight", _load("language_model.model.embed_tokens.weight"))
    _set_param(model.model.norm,         "weight", _load("language_model.model.norm.weight"))
    _set_param(model.lm_head,            "weight", _load("language_model.lm_head.weight"))


# ---------------------------------------------------------------------------
# High-level: build + fully populate from a checkpoint directory
# ---------------------------------------------------------------------------


def load_real_model(
    model_dir: str | Path,
    out_dtype: torch.dtype = torch.bfloat16,
    layers: list[int] | None = None,
) -> QwenForCausalLM:
    """Build QwenForCausalLM from the real config and load all weights.

    Args:
        model_dir: path to the snapshot_download output (contains config.json
                   + model.safetensors.index.json + *.safetensors shards).
        out_dtype: dtype to cast weights to (bfloat16 saves ~2x vs float32).
        layers:    optional subset of layer indices to load (the rest keep
                   random init; only useful for debugging single layers).
    """
    model_dir  = Path(model_dir)
    config     = Qwen35Config.from_pretrained_dir(model_dir)
    weight_map = build_shard_index(model_dir)

    model = QwenForCausalLM(config)
    model.eval()

    load_global_weights(model, weight_map, model_dir, out_dtype=out_dtype)

    layer_indices = list(range(config.num_hidden_layers)) if layers is None else layers
    for i in layer_indices:
        print(f"  loading layer {i+1}/{config.num_hidden_layers} ...", end="\r")
        load_layer_weights(model.model.layers[i], i, weight_map, model_dir, out_dtype=out_dtype)
    print()

    return model
