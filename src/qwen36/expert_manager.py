"""Phase 2.3: lazy expert loading with an LRU cache.

The real model has 256 experts x 40 layers = 10,240 routed experts, but only
8 per layer fire on any given token. Keeping every expert resident wastes
memory; recompiling/redequantizing on every access wastes time. The
ExpertManager sits in between: experts are produced lazily by a caller-
supplied loader, kept in an LRU cache of bounded capacity, and evicted when
the cache is full.

The manager is agnostic to what an "expert" is -- it can be a torch nn.Module
(toy / pure-torch path) or a compiled OpenVINO model wrapped in a small
callable. The only contract is `expert(x) -> y`.

Prewarming: given calibration statistics (which experts fire most often), the
hot set can be preloaded so the steady-state hit rate is high.
"""
from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Hashable, Iterable

import torch


ExpertKey = tuple[int, int]  # (layer_idx, expert_idx)
ExpertCallable = Callable[[torch.Tensor], torch.Tensor]
ExpertLoader = Callable[[int, int], ExpertCallable]


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    loads: int = 0

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total if self.total else 0.0

    def reset(self) -> None:
        self.hits = self.misses = self.evictions = self.loads = 0

    def __str__(self) -> str:
        return (
            f"hits={self.hits} misses={self.misses} evictions={self.evictions} "
            f"loads={self.loads} hit_rate={self.hit_rate:.1%}"
        )


class ExpertManager:
    """LRU-cached lazy expert store.

    Args:
        loader: callable (layer_idx, expert_idx) -> expert callable. Invoked on
            a cache miss; its cost (dequant + optional compile) is what the
            cache is amortizing.
        capacity: maximum number of experts kept resident. None = unbounded.
    """

    def __init__(self, loader: ExpertLoader, capacity: int | None = 64):
        self.loader = loader
        self.capacity = capacity
        self._cache: "OrderedDict[ExpertKey, ExpertCallable]" = OrderedDict()
        self._pinned: set[ExpertKey] = set()
        self.stats = CacheStats()

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, key: ExpertKey) -> bool:
        return key in self._cache

    def get(self, layer_idx: int, expert_idx: int) -> ExpertCallable:
        key = (layer_idx, expert_idx)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            self.stats.hits += 1
            return cached

        self.stats.misses += 1
        expert = self.loader(layer_idx, expert_idx)
        self.stats.loads += 1
        self._cache[key] = expert
        self._maybe_evict()
        return expert

    def run(self, layer_idx: int, expert_idx: int, x: torch.Tensor) -> torch.Tensor:
        return self.get(layer_idx, expert_idx)(x)

    def prewarm(self, keys: Iterable[ExpertKey], pin: bool = False) -> None:
        """Preload the given experts. If pin=True they are exempt from
        eviction (useful for the always-hot set)."""
        for layer_idx, expert_idx in keys:
            self.get(layer_idx, expert_idx)
            if pin:
                self._pinned.add((layer_idx, expert_idx))

    def _maybe_evict(self) -> None:
        if self.capacity is None:
            return
        while len(self._cache) > self.capacity:
            # Evict the least-recently-used non-pinned entry.
            for key in self._cache:
                if key not in self._pinned:
                    del self._cache[key]
                    self.stats.evictions += 1
                    break
            else:
                # All resident entries are pinned; nothing to evict.
                break


# ---------------------------------------------------------------------------
# Calibration: count expert activation frequency
# ---------------------------------------------------------------------------


@dataclass
class ExpertFrequency:
    """Per-(layer, expert) activation counts collected over calibration."""

    counts: dict[ExpertKey, int] = field(default_factory=lambda: defaultdict(int))
    tokens_seen: int = 0

    def add(self, layer_idx: int, expert_indices: torch.Tensor) -> None:
        """expert_indices: (num_tokens, top_k) int tensor of selected experts."""
        flat = expert_indices.reshape(-1).tolist()
        for e in flat:
            self.counts[(layer_idx, int(e))] += 1

    def hot_keys(self, top_n: int) -> list[ExpertKey]:
        """Return the top_n most-activated (layer, expert) keys."""
        return [k for k, _ in sorted(self.counts.items(), key=lambda kv: -kv[1])[:top_n]]

    def hot_keys_per_layer(self, per_layer: int) -> list[ExpertKey]:
        """Return the top `per_layer` experts within each layer."""
        by_layer: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for (layer, expert), count in self.counts.items():
            by_layer[layer].append((count, expert))
        out: list[ExpertKey] = []
        for layer, items in by_layer.items():
            items.sort(reverse=True)
            out.extend((layer, expert) for _, expert in items[:per_layer])
        return out
