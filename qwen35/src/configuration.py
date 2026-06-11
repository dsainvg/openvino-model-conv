"""Config dataclass for the Qwen3.5-4B OV port.

Qwen3.5-4B (model_type=qwen3_5) is a multimodal VLM with a hybrid text
backbone identical in structure to Qwen3.6-35B-A3B (qwen3_5_moe): every
full_attention_interval-th layer is a standard GQA full-attention layer;
the rest are Gated DeltaNet linear-attention layers. The FFN in every layer
is a dense SwiGLU MLP (no mixture-of-experts in the 4B variant).

Key difference from qwen36: dense FFN instead of 256-expert MoE, and
different sizes (2560 hidden, 32 layers, 4B active params).

The class reads only the fields that modeling.py actually uses. No
transformers dependency -- the venv only needs torch + openvino.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path


@dataclass
class Qwen35Config:
    # --- core ---
    hidden_size: int = 2560
    num_hidden_layers: int = 32
    vocab_size: int = 248320

    # --- full-attention layers ---
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 256
    attention_bias: bool = False
    attn_output_gate: bool = True
    partial_rotary_factor: float = 0.25
    rope_theta: float = 10_000_000.0
    max_position_embeddings: int = 262_144

    # --- linear-attention layers (Gated DeltaNet) ---
    linear_num_value_heads: int = 32
    linear_num_key_heads: int = 16
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4

    # --- dense FFN ---
    intermediate_size: int = 7168

    # --- layer pattern (full_attention_interval=4 means every 4th layer is full attn) ---
    full_attention_interval: int = 4
    layer_types: tuple[str, ...] | None = None

    # --- norm ---
    rms_norm_eps: float = 1e-6
    hidden_act: str = "silu"

    # --- token ids ---
    bos_token_id: int = 151644
    eos_token_id: int = 151645
    pad_token_id: int = 151643

    def __post_init__(self):
        if self.layer_types is None:
            self.layer_types = tuple(
                "full_attention" if (i + 1) % self.full_attention_interval == 0
                else "linear_attention"
                for i in range(self.num_hidden_layers)
            )
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"layer_types len {len(self.layer_types)} != "
                f"num_hidden_layers {self.num_hidden_layers}"
            )

    # ---- derived ----
    @property
    def linear_conv_dim(self) -> int:
        return self.linear_key_dim * 2 + self.linear_value_dim

    @property
    def linear_key_dim(self) -> int:
        return self.linear_key_head_dim * self.linear_num_key_heads

    @property
    def linear_value_dim(self) -> int:
        return self.linear_value_head_dim * self.linear_num_value_heads

    @property
    def num_v_per_k(self) -> int:
        return self.linear_num_value_heads // self.linear_num_key_heads

    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def partial_rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    # ---- loaders ----
    @classmethod
    def from_pretrained_dir(cls, model_dir: str | Path) -> "Qwen35Config":
        """Read config.json from the checkpoint directory.

        Qwen3.5-4B is a VLM; the text backbone lives under text_config.
        Falls back to the top-level if text_config is absent (text-only fork).

        Field location notes (verified against real Qwen/Qwen3.5-4B config.json):
          - partial_rotary_factor lives inside text_config.rope_parameters
          - rope_theta lives inside text_config.rope_parameters
          - layer_types is a flat list inside text_config
        """
        model_dir = Path(model_dir)
        with open(model_dir / "config.json", encoding="utf-8") as fh:
            raw = json.load(fh)

        # VLM layout: text backbone is nested under text_config
        text = raw.get("text_config", raw)
        rope_params = text.get("rope_parameters", {})

        # partial_rotary_factor: real config nests it in rope_parameters;
        # test fixtures may write it at top level — check both.
        partial_rotary_factor = (
            rope_params.get("partial_rotary_factor")
            or text.get("partial_rotary_factor")
            or 0.25  # Qwen3.5-4B architectural default
        )

        rope_theta = (
            rope_params.get("rope_theta")
            or text.get("rope_theta")
            or 10_000_000.0
        )

        layer_types_raw = text.get("layer_types")
        layer_types = tuple(layer_types_raw) if layer_types_raw else None

        return cls(
            hidden_size=text["hidden_size"],
            num_hidden_layers=text["num_hidden_layers"],
            vocab_size=text["vocab_size"],
            num_attention_heads=text["num_attention_heads"],
            num_key_value_heads=text["num_key_value_heads"],
            head_dim=text["head_dim"],
            attention_bias=text.get("attention_bias", False),
            attn_output_gate=text.get("attn_output_gate", True),
            partial_rotary_factor=partial_rotary_factor,
            rope_theta=rope_theta,
            max_position_embeddings=text["max_position_embeddings"],
            linear_num_value_heads=text["linear_num_value_heads"],
            linear_num_key_heads=text["linear_num_key_heads"],
            linear_key_head_dim=text["linear_key_head_dim"],
            linear_value_head_dim=text["linear_value_head_dim"],
            linear_conv_kernel_dim=text["linear_conv_kernel_dim"],
            intermediate_size=text["intermediate_size"],
            full_attention_interval=text.get("full_attention_interval", 4),
            layer_types=layer_types,
            rms_norm_eps=text.get("rms_norm_eps", 1e-6),
            hidden_act=text.get("hidden_act", "silu"),
            bos_token_id=text.get("bos_token_id", 151644),
            eos_token_id=text.get("eos_token_id", 151645),
            pad_token_id=text.get("pad_token_id", 151643),
        )


def make_toy_config(
    *,
    num_layers: int = 4,
    full_attention_interval: int = 4,
    hidden_size: int = 64,
    intermediate_size: int = 128,
    num_attention_heads: int = 4,
    num_key_value_heads: int = 2,
    head_dim: int = 16,
    linear_num_key_heads: int = 2,
    linear_num_value_heads: int = 4,
    linear_key_head_dim: int = 8,
    linear_value_head_dim: int = 8,
    linear_conv_kernel_dim: int = 4,
    vocab_size: int = 256,
) -> Qwen35Config:
    """Tiny config that exercises every code path (3 DeltaNet + 1 full-attn)
    in milliseconds on CPU with random weights."""
    return Qwen35Config(
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        vocab_size=vocab_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        partial_rotary_factor=0.25,
        rope_theta=10_000.0,
        max_position_embeddings=128,
        linear_num_value_heads=linear_num_value_heads,
        linear_num_key_heads=linear_num_key_heads,
        linear_key_head_dim=linear_key_head_dim,
        linear_value_head_dim=linear_value_head_dim,
        linear_conv_kernel_dim=linear_conv_kernel_dim,
        intermediate_size=intermediate_size,
        full_attention_interval=full_attention_interval,
        rms_norm_eps=1e-6,
        hidden_act="silu",
    )
