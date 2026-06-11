"""Unit tests for the Qwen3.5-4B OV port using a toy config.

Mirrors qwen36/tests/test_toy_match.py:
- Tests are internal-consistency only (no transformers reference needed).
- Covers shapes, dtype handling, state propagation across steps, and the
  key invariants of each submodule.

Run:
    pytest tests/test_toy_match.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.configuration import make_toy_config
from src.modeling import (
    QwenAttention,
    QwenForCausalLM,
    QwenGatedDeltaNet,
    QwenMLP,
    QwenRMSNorm,
    QwenTextModel,
    build_rope_cache,
)


@pytest.fixture(scope="module")
def toy_cfg():
    return make_toy_config()


@pytest.fixture(scope="module")
def seed():
    torch.manual_seed(0)
    yield


# ---------------------------------------------------------------------------
# Norm
# ---------------------------------------------------------------------------


def test_rmsnorm_zero_weight_acts_as_identity_on_unit_rms():
    norm = QwenRMSNorm(dim=8)  # weight defaults to zeros → (1+0)=1
    x = torch.randn(2, 3, 8)
    x_unit = x / x.float().pow(2).mean(-1, keepdim=True).sqrt().to(x.dtype)
    out = norm(x_unit)
    assert torch.allclose(out, x_unit, atol=1e-5)


# ---------------------------------------------------------------------------
# Rope cache
# ---------------------------------------------------------------------------


def test_rope_cache_shape():
    cos, sin = build_rope_cache(rotary_dim=16, max_position=64, base=10_000.0, dtype=torch.float32)
    assert cos.shape == (64, 16)
    assert sin.shape == (64, 16)


# ---------------------------------------------------------------------------
# Dense MLP
# ---------------------------------------------------------------------------


def test_mlp_output_shape(toy_cfg, seed):
    mlp = QwenMLP(toy_cfg.hidden_size, toy_cfg.intermediate_size)
    mlp.eval()
    x   = torch.randn(1, 1, toy_cfg.hidden_size)
    out = mlp(x)
    assert out.shape == x.shape


# ---------------------------------------------------------------------------
# Full-attention
# ---------------------------------------------------------------------------


def test_attention_state_grows(toy_cfg, seed):
    layer = QwenAttention(toy_cfg)
    layer.eval()
    B, max_seq = 1, 8
    rotary_dim = toy_cfg.partial_rotary_dim
    x       = torch.randn(B, 1, toy_cfg.hidden_size)
    cos     = torch.randn(B, 1, rotary_dim)
    sin     = torch.randn(B, 1, rotary_dim)
    k_cache = torch.zeros(B, toy_cfg.num_key_value_heads, max_seq, toy_cfg.head_dim)
    v_cache = torch.zeros_like(k_cache)

    out0, k0, v0 = layer(x, cos, sin, k_cache, v_cache, torch.tensor(0))
    assert out0.shape == (B, 1, toy_cfg.hidden_size)
    assert not torch.all(k0[:, :, 0] == 0), "position 0 must be written"
    assert torch.all(k0[:, :, 1:] == 0),   "future positions must stay zero"

    x2 = torch.randn(B, 1, toy_cfg.hidden_size)
    out1, k1, v1 = layer(x2, cos, sin, k0, v0, torch.tensor(1))
    assert torch.equal(k1[:, :, 0], k0[:, :, 0]), "position 0 must be preserved"
    assert not torch.all(k1[:, :, 1] == 0),       "position 1 must be written"


# ---------------------------------------------------------------------------
# Gated DeltaNet
# ---------------------------------------------------------------------------


def test_gated_deltanet_shapes(toy_cfg, seed):
    layer = QwenGatedDeltaNet(toy_cfg)
    layer.eval()
    B = 1
    conv_in = torch.zeros(B, toy_cfg.linear_conv_dim, toy_cfg.linear_conv_kernel_dim)
    rec_in  = torch.zeros(
        B, toy_cfg.linear_num_value_heads,
        toy_cfg.linear_key_head_dim, toy_cfg.linear_value_head_dim,
    )
    x   = torch.randn(B, 1, toy_cfg.hidden_size)
    out, conv_out, rec_out = layer(x, conv_in, rec_in)

    assert out.shape      == (B, 1, toy_cfg.hidden_size)
    assert conv_out.shape == conv_in.shape
    assert rec_out.shape  == rec_in.shape


def test_gated_deltanet_conv_shift(toy_cfg, seed):
    """The conv state must shift left by 1 on each step."""
    layer    = QwenGatedDeltaNet(toy_cfg)
    layer.eval()
    B        = 1
    conv_in  = torch.randn(B, toy_cfg.linear_conv_dim, toy_cfg.linear_conv_kernel_dim)
    rec_in   = torch.zeros(
        B, toy_cfg.linear_num_value_heads,
        toy_cfg.linear_key_head_dim, toy_cfg.linear_value_head_dim,
    )
    x        = torch.randn(B, 1, toy_cfg.hidden_size)
    _, conv_out, _ = layer(x, conv_in, rec_in)
    # First K-1 columns of conv_out must equal last K-1 columns of conv_in
    assert torch.allclose(conv_out[:, :, :-1], conv_in[:, :, 1:])


def test_gated_deltanet_recurrent_state_changes(toy_cfg, seed):
    layer   = QwenGatedDeltaNet(toy_cfg)
    layer.eval()
    rec_in  = torch.zeros(
        1, toy_cfg.linear_num_value_heads,
        toy_cfg.linear_key_head_dim, toy_cfg.linear_value_head_dim,
    )
    conv_in = torch.zeros(1, toy_cfg.linear_conv_dim, toy_cfg.linear_conv_kernel_dim)
    x       = torch.randn(1, 1, toy_cfg.hidden_size)
    _, _, rec_out = layer(x, conv_in, rec_in)
    assert rec_out.abs().sum().item() > 0, "recurrent state must update"


# ---------------------------------------------------------------------------
# Full model end-to-end
# ---------------------------------------------------------------------------


def test_full_model_forward(toy_cfg, seed):
    model = QwenForCausalLM(toy_cfg)
    model.eval()
    state = model.empty_state(batch=1, max_seq=8)
    with torch.no_grad():
        logits, k_out, v_out, conv_out, rec_out = model(
            torch.tensor([[42]]),
            torch.tensor([[0]]),
            state["k_caches"], state["v_caches"],
            state["conv_states"], state["rec_states"],
        )
    assert logits.shape == (1, 1, toy_cfg.vocab_size)
    assert len(k_out)    == model.model.num_full_layers
    assert len(conv_out) == model.model.num_linear_layers
    assert torch.isfinite(logits).all()


def test_full_model_two_steps_state_propagates(toy_cfg, seed):
    model = QwenForCausalLM(toy_cfg)
    model.eval()
    state = model.empty_state(batch=1, max_seq=8)
    with torch.no_grad():
        logits0, k1, v1, conv1, rec1 = model(
            torch.tensor([[42]]), torch.tensor([[0]]),
            state["k_caches"], state["v_caches"],
            state["conv_states"], state["rec_states"],
        )
        logits1, *_ = model(
            torch.tensor([[7]]), torch.tensor([[1]]),
            k1, v1, conv1, rec1,
        )
    assert not torch.allclose(logits0, logits1), \
        "step-1 logits must differ from step-0 (different input + nonempty state)"


def test_different_tokens_produce_different_logits(toy_cfg, seed):
    model = QwenForCausalLM(toy_cfg)
    model.eval()
    state = model.empty_state(batch=1, max_seq=4)
    with torch.no_grad():
        la, *_ = model(torch.tensor([[1]]), torch.tensor([[0]]),
                       state["k_caches"], state["v_caches"],
                       state["conv_states"], state["rec_states"])
        lb, *_ = model(torch.tensor([[2]]), torch.tensor([[0]]),
                       state["k_caches"], state["v_caches"],
                       state["conv_states"], state["rec_states"])
    assert not torch.allclose(la, lb), \
        "different input tokens must produce different logits"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
