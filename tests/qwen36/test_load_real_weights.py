"""Smoke test for the real-weight loader.

Builds a 1-layer model with the real hidden sizes (2048, 256 experts, etc)
and loads the actual safetensors weights for one source layer. Verifies
the forward runs and produces finite logits.

Skipped if the checkpoint isn't on disk.
"""
from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.qwen36.configuration_qwen36 import Qwen36Config  # noqa: E402
from src.qwen36.load_weights import (  # noqa: E402
    build_shard_index,
    load_global_weights,
    load_layer_weights,
)
from src.qwen36.modeling_qwen36 import QwenForCausalLM  # noqa: E402


MODEL_DIR = Path(r"C:\Users\intel\models\qwen36-35b-int4")
HAVE_MODEL = MODEL_DIR.exists() and (MODEL_DIR / "model.safetensors.index.json").exists()
needs_model = pytest.mark.skipif(not HAVE_MODEL, reason="Qwen3.6 checkpoint not on disk")


def _one_layer_config(layer_type: str) -> Qwen36Config:
    """Real Qwen3.6 config but with only one layer of the chosen type."""
    real = Qwen36Config.from_pretrained_dir(MODEL_DIR)
    return replace(real, num_hidden_layers=1, layer_types=(layer_type,))


@needs_model
def test_load_one_linear_attention_layer():
    """Build a 1-layer linear-attn model with real hidden=2048, 256 experts,
    load real layer 0 (linear) + global weights, run one forward step,
    verify finite + reasonable logits."""
    cfg = _one_layer_config("linear_attention")
    model = QwenForCausalLM(cfg)
    model.eval()
    weight_map = build_shard_index(MODEL_DIR)

    t0 = time.time()
    load_global_weights(model, weight_map, MODEL_DIR, out_dtype=torch.bfloat16)
    print(f"\nloaded globals in {time.time()-t0:.1f}s")

    t0 = time.time()
    load_layer_weights(model.model.layers[0], 0, weight_map, MODEL_DIR, out_dtype=torch.bfloat16)
    print(f"loaded 1 linear layer (real layer 0, 256 experts) in {time.time()-t0:.1f}s")

    # Forward
    state = model.empty_state(batch=1, max_seq=4, dtype=torch.bfloat16)
    input_ids = torch.tensor([[42]])
    position_ids = torch.tensor([[0]])
    t0 = time.time()
    with torch.no_grad():
        logits, *_ = model(input_ids, position_ids, **state)
    print(f"forward in {time.time()-t0:.1f}s")

    assert logits.shape == (1, 1, cfg.vocab_size)
    assert torch.isfinite(logits).all()
    # Quick sanity: logits shouldn't be all zeros (untrained behavior) and
    # softmax mass shouldn't be on a single token (untrained behavior).
    probs = torch.softmax(logits.float(), dim=-1).flatten()
    max_p = probs.max().item()
    print(f"max logit prob = {max_p:.4f}")
    assert max_p < 0.99, f"degenerate distribution, max p = {max_p}"
    assert max_p > 1e-6, "softmax all-zero -- weights probably didn't load"


@needs_model
def test_load_one_full_attention_layer():
    """Same but for a full-attention layer, loading from real layer 3."""
    cfg = _one_layer_config("full_attention")
    model = QwenForCausalLM(cfg)
    model.eval()
    weight_map = build_shard_index(MODEL_DIR)

    load_global_weights(model, weight_map, MODEL_DIR, out_dtype=torch.bfloat16)
    t0 = time.time()
    # Layer 3 of the real model is the first full_attention layer (pattern:
    # 3 linear, then 1 full, repeating).
    load_layer_weights(model.model.layers[0], 3, weight_map, MODEL_DIR, out_dtype=torch.bfloat16)
    print(f"\nloaded 1 full-attn layer (real layer 3) in {time.time()-t0:.1f}s")

    state = model.empty_state(batch=1, max_seq=4, dtype=torch.bfloat16)
    with torch.no_grad():
        logits, *_ = model(torch.tensor([[100]]), torch.tensor([[0]]), **state)

    assert torch.isfinite(logits).all()
    probs = torch.softmax(logits.float(), dim=-1).flatten()
    assert 1e-6 < probs.max().item() < 0.99


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
