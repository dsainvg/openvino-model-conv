"""Minimal Qwen3.6 config dataclass for the OV port.

Lifted from the real config.json text_config (model_type qwen3_5_moe) and
trimmed to the fields the modeling code actually reads. No transformers
dependency, so the main venv (transformers 4.57) can use it too.

A "toy" config helper is included for building tiny test models.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
from pathlib import Path


@dataclass
class Qwen36Config:
    # --- core sizes ---
    hidden_size: int = 2048
    num_hidden_layers: int = 40
    vocab_size: int = 248320

    # --- attention (full) ---
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    head_dim: int = 256
    attention_bias: bool = False
    attention_dropout: float = 0.0
    attn_output_gate: bool = True
    partial_rotary_factor: float = 0.25
    rope_theta: float = 10_000_000.0  # for the full-attention layers
    max_position_embeddings: int = 262_144

    # --- linear attention (Gated DeltaNet) ---
    linear_num_value_heads: int = 32
    linear_num_key_heads: int = 16
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4

    # --- MoE ---
    num_experts: int = 256
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 512
    shared_expert_intermediate_size: int = 512

    # --- layer pattern ---
    full_attention_interval: int = 4  # every Nth layer is full attention
    layer_types: tuple[str, ...] | None = None  # if None, derived from full_attention_interval

    # --- norm ---
    rms_norm_eps: float = 1e-6
    hidden_act: str = "silu"

    # --- token IDs (kept for completeness) ---
    bos_token_id: int = 248044
    eos_token_id: int = 248044
    pad_token_id: int = 248044

    def __post_init__(self):
        if self.layer_types is None:
            self.layer_types = tuple(
                "full_attention" if (i + 1) % self.full_attention_interval == 0 else "linear_attention"
                for i in range(self.num_hidden_layers)
            )
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"layer_types len {len(self.layer_types)} != num_hidden_layers {self.num_hidden_layers}"
            )

    # ---- derived ----
    @property
    def linear_conv_dim(self) -> int:
        """Total feature dim of the depthwise conv1d in GatedDeltaNet:
        key_dim*2 + value_dim."""
        key_dim = self.linear_key_head_dim * self.linear_num_key_heads
        value_dim = self.linear_value_head_dim * self.linear_num_value_heads
        return key_dim * 2 + value_dim

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
    def from_pretrained_dir(cls, model_dir: str | Path) -> "Qwen36Config":
        """Read config.json from the checkpoint dir."""
        model_dir = Path(model_dir)
        with open(model_dir / "config.json") as fh:
            raw = json.load(fh)
        text = raw["text_config"]
        return cls(
            hidden_size=text["hidden_size"],
            num_hidden_layers=text["num_hidden_layers"],
            vocab_size=text["vocab_size"],
            num_attention_heads=text["num_attention_heads"],
            num_key_value_heads=text["num_key_value_heads"],
            head_dim=text["head_dim"],
            attention_bias=text.get("attention_bias", False),
            attention_dropout=text.get("attention_dropout", 0.0),
            attn_output_gate=text.get("attn_output_gate", True),
            partial_rotary_factor=text["partial_rotary_factor"],
            rope_theta=text.get("rope_parameters", {}).get("rope_theta", 10_000_000.0),
            max_position_embeddings=text["max_position_embeddings"],
            linear_num_value_heads=text["linear_num_value_heads"],
            linear_num_key_heads=text["linear_num_key_heads"],
            linear_key_head_dim=text["linear_key_head_dim"],
            linear_value_head_dim=text["linear_value_head_dim"],
            linear_conv_kernel_dim=text["linear_conv_kernel_dim"],
            num_experts=text["num_experts"],
            num_experts_per_tok=text["num_experts_per_tok"],
            moe_intermediate_size=text["moe_intermediate_size"],
            shared_expert_intermediate_size=text["shared_expert_intermediate_size"],
            full_attention_interval=text["full_attention_interval"],
            layer_types=tuple(text["layer_types"]),
            rms_norm_eps=text["rms_norm_eps"],
            hidden_act=text.get("hidden_act", "silu"),
            bos_token_id=text.get("bos_token_id", 248044),
            eos_token_id=text.get("eos_token_id", 248044),
            pad_token_id=text.get("pad_token_id", 248044),
        )


def make_toy_config(
    *,
    num_layers: int = 2,
    full_attention_interval: int = 2,
    hidden_size: int = 64,
    num_experts: int = 4,
    top_k: int = 2,
    moe_intermediate: int = 32,
    num_attention_heads: int = 4,
    num_key_value_heads: int = 2,
    head_dim: int = 16,
    linear_num_key_heads: int = 2,
    linear_num_value_heads: int = 4,
    linear_key_head_dim: int = 8,
    linear_value_head_dim: int = 8,
    linear_conv_kernel_dim: int = 4,
    vocab_size: int = 256,
) -> Qwen36Config:
    """Tiny config that preserves the architecture's structural features
    (hybrid layer pattern, MoE routing, GatedDeltaNet, GQA) at minimal cost."""
    return Qwen36Config(
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
        num_experts=num_experts,
        num_experts_per_tok=top_k,
        moe_intermediate_size=moe_intermediate,
        shared_expert_intermediate_size=moe_intermediate,
        full_attention_interval=full_attention_interval,
        rms_norm_eps=1e-6,
        hidden_act="silu",
    )
