"""Phase 2.5 pipeline tests: autoregressive generation on the toy model."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.configuration import make_toy_config  # noqa: E402
from src.expert_manager import ExpertManager  # noqa: E402
from src.modeling import QwenForCausalLM  # noqa: E402
from src.pipeline import Qwen36Pipeline  # noqa: E402


def _toy_model():
    torch.manual_seed(0)
    model = QwenForCausalLM(make_toy_config())
    model.eval()
    return model


def test_generate_produces_requested_tokens():
    model = _toy_model()
    pipe = Qwen36Pipeline(model)
    result = pipe.generate([1, 2, 3], max_new_tokens=5, greedy=True)
    assert len(result.token_ids) == 5
    assert all(0 <= t < model.config.vocab_size for t in result.token_ids)
    assert result.tokens_per_second > 0


def test_greedy_is_deterministic():
    model = _toy_model()
    p1 = Qwen36Pipeline(model).generate([7, 8], max_new_tokens=6, greedy=True)
    p2 = Qwen36Pipeline(model).generate([7, 8], max_new_tokens=6, greedy=True)
    assert p1.token_ids == p2.token_ids


def test_pipeline_step_matches_monolithic_forward():
    """One pipeline step must match the monolithic model forward (the pipeline
    uses the split path + ExpertManager; result should be identical within
    fp tolerance)."""
    model = _toy_model()
    pipe = Qwen36Pipeline(model)
    state = model.empty_state(batch=1, max_seq=8)

    input_ids = torch.tensor([[11]])
    position_ids = torch.tensor([[0]])
    with torch.no_grad():
        mono_logits, *_ = model(
            input_ids, position_ids, state["k_caches"], state["v_caches"],
            state["conv_states"], state["rec_states"],
        )
        pipe_logits, _ = pipe.step(input_ids, position_ids, state)
    assert torch.allclose(mono_logits, pipe_logits, atol=1e-4)


def test_expert_frequency_collection():
    model = _toy_model()
    pipe = Qwen36Pipeline(model, collect_freq=True)
    pipe.generate([1, 2, 3, 4], max_new_tokens=8, greedy=True)
    freq = pipe.expert_freq
    assert freq is not None
    # Each MoE layer fires top_k experts per token; total selections =
    # steps * num_layers * top_k.
    total = sum(freq.counts.values())
    assert total > 0
    # Hot keys should be retrievable.
    hot = freq.hot_keys(top_n=3)
    assert len(hot) <= 3


def test_capacity_bounded_manager_still_correct():
    """Generation through a tiny-capacity LRU cache must match an unbounded
    one (caching is transparent to outputs)."""
    model = _toy_model()
    unbounded = Qwen36Pipeline(model, ExpertManager(
        lambda li, ei: model.model.layers[li].mlp.experts[ei], capacity=None))
    bounded = Qwen36Pipeline(model, ExpertManager(
        lambda li, ei: model.model.layers[li].mlp.experts[ei], capacity=2))

    r_unb = unbounded.generate([3, 1, 4], max_new_tokens=6, greedy=True)
    r_bnd = bounded.generate([3, 1, 4], max_new_tokens=6, greedy=True)
    assert r_unb.token_ids == r_bnd.token_ids
    # The bounded cache should have evicted at least once (toy has 4 experts,
    # capacity 2).
    assert bounded.experts.stats.evictions > 0


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
