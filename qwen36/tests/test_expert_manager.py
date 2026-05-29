"""Tests for the ExpertManager LRU cache + calibration helpers."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.expert_manager import ExpertFrequency, ExpertManager  # noqa: E402


def _counting_loader():
    """Loader that returns an identity-scaling expert and records how many
    times each key was actually loaded (to verify caching avoids reloads)."""
    load_calls: dict[tuple[int, int], int] = {}

    def loader(layer_idx, expert_idx):
        load_calls[(layer_idx, expert_idx)] = load_calls.get((layer_idx, expert_idx), 0) + 1
        scale = float(layer_idx * 100 + expert_idx + 1)
        return lambda x: x * scale

    return loader, load_calls


def test_cache_hit_avoids_reload():
    loader, load_calls = _counting_loader()
    mgr = ExpertManager(loader, capacity=8)

    x = torch.ones(2, 4)
    out1 = mgr.run(0, 3, x)
    out2 = mgr.run(0, 3, x)  # should hit cache
    assert torch.equal(out1, out2)
    assert load_calls[(0, 3)] == 1, "expert reloaded on cache hit"
    assert mgr.stats.hits == 1
    assert mgr.stats.misses == 1


def test_lru_eviction_order():
    loader, load_calls = _counting_loader()
    mgr = ExpertManager(loader, capacity=2)

    x = torch.ones(1, 2)
    mgr.run(0, 0, x)  # cache: [0,0]
    mgr.run(0, 1, x)  # cache: [0,0], [0,1]
    mgr.run(0, 0, x)  # touch 0,0 -> cache: [0,1], [0,0]  (0,1 now LRU)
    mgr.run(0, 2, x)  # over capacity -> evict 0,1. cache: [0,0],[0,2]
    assert (0, 1) not in mgr
    assert (0, 0) in mgr
    assert (0, 2) in mgr
    assert mgr.stats.evictions == 1

    # Re-access evicted entry -> reload
    mgr.run(0, 1, x)
    assert load_calls[(0, 1)] == 2, "evicted expert should reload"


def test_pinned_experts_not_evicted():
    loader, _ = _counting_loader()
    mgr = ExpertManager(loader, capacity=2)
    x = torch.ones(1, 2)

    mgr.prewarm([(0, 0)], pin=True)  # pin 0,0
    mgr.run(0, 1, x)
    mgr.run(0, 2, x)  # would evict LRU; 0,0 is pinned so 0,1 goes instead
    assert (0, 0) in mgr, "pinned expert was evicted"
    assert (0, 1) not in mgr


def test_unbounded_capacity():
    loader, _ = _counting_loader()
    mgr = ExpertManager(loader, capacity=None)
    x = torch.ones(1, 2)
    for e in range(50):
        mgr.run(0, e, x)
    assert len(mgr) == 50
    assert mgr.stats.evictions == 0


def test_expert_frequency_hot_keys():
    freq = ExpertFrequency()
    # Layer 0: expert 5 fires 3x, expert 2 fires 1x
    freq.add(0, torch.tensor([[5, 2], [5, 5]]))  # 5 appears 3x, 2 appears 1x
    # Layer 1: expert 7 fires 2x
    freq.add(1, torch.tensor([[7, 7]]))
    assert freq.counts[(0, 5)] == 3
    assert freq.counts[(0, 2)] == 1
    assert freq.counts[(1, 7)] == 2

    hot = freq.hot_keys(top_n=2)
    assert hot[0] == (0, 5)  # most frequent overall
    assert (1, 7) in hot

    per_layer = freq.hot_keys_per_layer(per_layer=1)
    assert (0, 5) in per_layer
    assert (1, 7) in per_layer
    assert (0, 2) not in per_layer  # not top-1 in its layer


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
