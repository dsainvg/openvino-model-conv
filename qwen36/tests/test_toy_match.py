"""Sanity tests for the OV-traceable port using a toy config.

We don't have a torch reference of our own architecture (the transformers
modeling code uses CUDA kernels and stateful Cache objects we can't depend
on), so these tests verify *internal consistency*: shapes, dtype handling,
state actually changes across steps, and (where math is fully determined
by inputs) reference values via hand-computed expectations.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.configuration import make_toy_config  # noqa: E402
from src.modeling import (  # noqa: E402
    QwenAttention,
    QwenForCausalLM,
    QwenGatedDeltaNet,
    QwenMoEBlock,
    QwenRMSNorm,
    QwenTextModel,
    build_rope_cache,
)


@pytest.fixture(scope="module")
def toy_config():
    return make_toy_config()


@pytest.fixture(scope="module")
def torch_seed():
    torch.manual_seed(0)
    yield


# ---------------------------------------------------------------------------
# Individual modules
# ---------------------------------------------------------------------------


def test_rmsnorm_zero_weight_is_identity_on_unit_norm():
    """When weight=0 and input is unit-norm along last dim, output = input."""
    norm = QwenRMSNorm(dim=8)
    # weight default is zeros, so the (1 + weight) = 1.
    x = torch.randn(2, 3, 8)
    # Normalize x manually: rms = sqrt(mean(x^2)); expected = x / (rms + eps_corrected)
    x_unit = x / x.float().pow(2).mean(-1, keepdim=True).sqrt().to(x.dtype)
    out = norm(x_unit)
    # After RMSNorm of an already-unit-norm-rms input, output equals input (within float).
    assert torch.allclose(out, x_unit, atol=1e-5)


def test_attention_state_grows(toy_config, torch_seed):
    layer = QwenAttention(toy_config)
    layer.eval()
    B, max_seq = 1, 8
    hidden = toy_config.hidden_size
    rotary_dim = toy_config.partial_rotary_dim
    x = torch.randn(B, 1, hidden)
    cos = torch.randn(B, 1, rotary_dim)
    sin = torch.randn(B, 1, rotary_dim)
    k_cache = torch.zeros(B, toy_config.num_key_value_heads, max_seq, toy_config.head_dim)
    v_cache = torch.zeros_like(k_cache)

    # Step 0
    out0, k0, v0 = layer(x, cos, sin, k_cache, v_cache, torch.tensor(0))
    assert out0.shape == (B, 1, hidden)
    # Position 0 should have nonzero values, others zero.
    assert not torch.all(k0[:, :, 0] == 0)
    assert torch.all(k0[:, :, 1:] == 0)

    # Step 1 with different input
    x2 = torch.randn(B, 1, hidden)
    out1, k1, v1 = layer(x2, cos, sin, k0, v0, torch.tensor(1))
    assert torch.equal(k1[:, :, 0], k0[:, :, 0]), "step 0 KV must be preserved"
    assert not torch.all(k1[:, :, 1] == 0), "step 1 must write into pos 1"


def test_gated_deltanet_state_changes(toy_config, torch_seed):
    layer = QwenGatedDeltaNet(toy_config)
    layer.eval()
    B = 1
    hidden = toy_config.hidden_size
    K = toy_config.linear_conv_kernel_dim

    x = torch.randn(B, 1, hidden)
    conv_in = torch.zeros(B, toy_config.linear_conv_dim, K)
    rec_in = torch.zeros(
        B, toy_config.linear_num_value_heads,
        toy_config.linear_key_head_dim, toy_config.linear_value_head_dim,
    )
    out, conv_out, rec_out = layer(x, conv_in, rec_in)
    assert out.shape == (B, 1, hidden)
    # Conv state shifted: last column of conv_out should be the new input projection
    # (we don't know the projection by name, but at least: conv_out[:, :, :-1]
    # equals conv_in[:, :, 1:] -- the shift property)
    assert torch.allclose(conv_out[:, :, :-1], conv_in[:, :, 1:])
    # Recurrent state should now be nonzero (the delta rule added k @ delta to the
    # zero initial state).
    assert rec_out.abs().sum().item() > 0


def test_moe_block_compute_all_matches_manual(toy_config, torch_seed):
    """Verify the compute-all + mask MoE produces the expected weighted sum
    of expert outputs."""
    block = QwenMoEBlock(toy_config)
    block.eval()
    B, S, H = 1, 1, toy_config.hidden_size
    x = torch.randn(B, S, H)

    with torch.no_grad():
        out_actual = block(x)

        # Recompute manually from the same expert weights.
        flat = x.reshape(B * S, H)
        router_logits = block.gate(flat)
        probs = torch.softmax(router_logits.float(), dim=-1)
        topk_w, topk_idx = probs.topk(block.top_k, dim=-1)
        topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)
        topk_w = topk_w.to(flat.dtype)
        manual = torch.zeros_like(flat)
        for token in range(B * S):
            for k in range(block.top_k):
                e = topk_idx[token, k].item()
                w = topk_w[token, k]
                manual[token] = manual[token] + block.experts[e](flat[token : token + 1]).squeeze(0) * w
        shared = block.shared_expert(flat) * torch.sigmoid(block.shared_expert_gate(flat))
        manual_total = (manual + shared).reshape(B, S, H)

    assert torch.allclose(out_actual, manual_total, atol=1e-5), (
        f"max diff {(out_actual - manual_total).abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# Full model end-to-end
# ---------------------------------------------------------------------------


def test_full_model_forward(toy_config, torch_seed):
    model = QwenForCausalLM(toy_config)
    model.eval()

    state = model.empty_state(batch=1, max_seq=8)
    input_ids = torch.tensor([[42]])
    position_ids = torch.tensor([[0]])
    with torch.no_grad():
        logits, k_out, v_out, conv_out, rec_out = model(
            input_ids, position_ids, **state
        )
    assert logits.shape == (1, 1, toy_config.vocab_size)
    assert len(k_out) == model.model.num_full_layers
    assert len(conv_out) == model.model.num_linear_layers
    # State changed: at least the first linear layer's recurrent state is nonzero.
    if rec_out:
        assert rec_out[0].abs().sum().item() > 0


def test_full_model_two_steps_state_propagates(toy_config, torch_seed):
    model = QwenForCausalLM(toy_config)
    model.eval()
    state = model.empty_state(batch=1, max_seq=8)
    with torch.no_grad():
        logits0, *new_state = model(
            torch.tensor([[42]]), torch.tensor([[0]]),
            state["k_caches"], state["v_caches"], state["conv_states"], state["rec_states"],
        )
        k1, v1, conv1, rec1 = new_state
        logits1, k2, v2, conv2, rec2 = model(
            torch.tensor([[7]]), torch.tensor([[1]]),
            k1, v1, conv1, rec1,
        )
    # Step-1 logits should differ from step-0 logits (different input + nonempty state).
    assert not torch.allclose(logits0, logits1)


def test_different_tokens_produce_different_logits(toy_config, torch_seed):
    model = QwenForCausalLM(toy_config)
    model.eval()
    state = model.empty_state(batch=1, max_seq=4)
    with torch.no_grad():
        l_a, *_ = model(torch.tensor([[1]]), torch.tensor([[0]]), **state)
        l_b, *_ = model(torch.tensor([[2]]), torch.tensor([[0]]), **state)
    assert not torch.allclose(l_a, l_b)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
