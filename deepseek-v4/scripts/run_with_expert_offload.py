"""2.2 — Expert offloading on the split IRs from 2.1.

Backbone segments (embed, per-layer pre_moe, per-layer post_moe, final-head) are
compiled once and kept resident. Routed-expert IRs live on disk; the orchestrator
compiles them on demand and keeps a small LRU of compiled handles.

For the toy with topk=2 and 8 experts per layer, the LRU is symbolic — real value
comes when this same pattern moves to the real V4-Flash (256 experts, ~70 GB of
INT4 weights) where keeping every expert hot is what we're trying to avoid.

Stats reported per layer:
  - active_experts:  count of experts the gate picked (np.unique(indices))
  - cache_hits/misses: against the LRU before this layer started
  - evictions:       experts kicked out to make room
End-to-end correctness check: split-OV-with-LRU logits == monolithic-OV logits
(greedy match required).
"""
import sys
import time
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import openvino as ov

from src import DeepseekV4ForCausalLM
from test_modeling_smoke import make_toy_config


class ExpertLRU:
    """Compile-on-miss, LRU-evict, OpenVINO expert cache."""

    def __init__(self, core, ir_dir, layer_count, expert_count, max_size, device="CPU"):
        self.core = core
        self.ir_dir = ir_dir
        self.L = layer_count
        self.E = expert_count
        self.max_size = max_size
        self.device = device
        self._cache: "OrderedDict[tuple[int, int], object]" = OrderedDict()
        self.compile_time_ms = 0.0
        self.stats = {"hits": 0, "misses": 0, "evictions": 0}

    def _ir_path(self, layer, expert):
        return self.ir_dir / f"expert_L{layer}_E{expert}.xml"

    def get(self, layer, expert):
        key = (layer, expert)
        if key in self._cache:
            self._cache.move_to_end(key)
            self.stats["hits"] += 1
            return self._cache[key]
        self.stats["misses"] += 1
        t0 = time.perf_counter()
        compiled = self.core.compile_model(str(self._ir_path(layer, expert)), self.device)
        self.compile_time_ms += (time.perf_counter() - t0) * 1000.0
        if len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)
            self.stats["evictions"] += 1
        self._cache[key] = compiled
        return compiled

    def snapshot_stats(self):
        return dict(self.stats), set(self._cache.keys())


def main():
    torch.manual_seed(0)
    cfg = make_toy_config()
    model = DeepseekV4ForCausalLM(cfg).eval()
    L = cfg.num_hidden_layers
    H = cfg.hc_mult
    d = cfg.hidden_size
    E = cfg.n_routed_experts
    V = cfg.vocab_size

    B, S = 1, 128
    torch.manual_seed(1)
    input_ids = torch.randint(0, V, (B, S))

    print("=== Reference: monolithic OpenVINO IR ===")
    core = ov.Core()
    mono_ir = ROOT / "ov_ir_toy" / "deepseek_v4_toy.xml"
    mono_c = core.compile_model(str(mono_ir), "CPU")
    mono_logits = mono_c([input_ids.numpy().astype(np.int64)])[0]
    mono_top = int(torch.from_numpy(mono_logits)[0, -1].argmax().item())
    print(f"  greedy_top = {mono_top}")

    split_dir = ROOT / "ov_ir_toy" / "expert_split"
    if not split_dir.exists() or not (split_dir / "embed.xml").exists():
        raise RuntimeError(
            f"split IRs missing in {split_dir}. Run scripts/split_to_expert_irs.py first."
        )

    print("\n=== Compile backbone segments (persistent) ===")
    t0 = time.perf_counter()
    embed_c = core.compile_model(str(split_dir / "embed.xml"), "CPU")
    final_c = core.compile_model(str(split_dir / "final.xml"), "CPU")
    pre_c = [core.compile_model(str(split_dir / f"pre_moe_L{i}.xml"), "CPU") for i in range(L)]
    post_c = [core.compile_model(str(split_dir / f"post_moe_L{i}.xml"), "CPU") for i in range(L)]
    backbone_compile_ms = (time.perf_counter() - t0) * 1000.0
    print(f"  compiled {2 + 2 * L} backbone IRs in {backbone_compile_ms:.1f} ms")

    # Pick an LRU size where evictions are guaranteed: < unique experts seen per
    # whole batch. For this toy each layer activates 2-3 experts; size=2 should
    # trigger evictions on layers with 3 active.
    lru_size = 2
    expert_cache = ExpertLRU(core, split_dir, L, E, max_size=lru_size)
    print(f"  expert LRU size = {lru_size} (of {E} total per layer)")

    print("\n=== Run end-to-end with offloaded experts ===")
    input_ids_np = input_ids.numpy().astype(np.int64)
    h = embed_c([input_ids_np])[0]
    total_active = 0

    for i in range(L):
        out = pre_c[i]([h, input_ids_np])
        y2_flat = out[0]
        x_res = out[1]
        post2 = out[2]
        comb2 = out[3]
        weights = out[4]
        indices = out[5]
        shared_out = out[6]

        N = y2_flat.shape[0]
        gate_mat = np.zeros((N, E), dtype=np.float32)
        np.put_along_axis(gate_mat, indices.astype(np.int64), weights.astype(np.float32), axis=1)

        stats_before, _ = expert_cache.snapshot_stats()
        active = np.unique(indices).tolist()
        total_active += len(active)
        moe_out_flat = np.zeros((N, d), dtype=np.float32)
        for e in active:
            exp_c = expert_cache.get(i, e)
            expert_out = exp_c([y2_flat])[0]
            moe_out_flat += gate_mat[:, e:e + 1] * expert_out
        moe_out_flat += shared_out

        stats_after, resident = expert_cache.snapshot_stats()
        layer_hits = stats_after["hits"] - stats_before["hits"]
        layer_misses = stats_after["misses"] - stats_before["misses"]
        layer_evict = stats_after["evictions"] - stats_before["evictions"]

        moe_out_bsd = moe_out_flat.reshape(B, S, d)
        h = post_c[i]([moe_out_bsd, x_res, post2, comb2])[0]
        print(
            f"  layer {i}: active={len(active):>2}  "
            f"hits={layer_hits} miss={layer_misses} evict={layer_evict}  "
            f"resident={sorted(resident)}"
        )

    split_logits = final_c([h])[0]
    split_logits_t = torch.from_numpy(split_logits)
    mono_logits_t = torch.from_numpy(mono_logits)
    diff = (mono_logits_t - split_logits_t).abs()
    sp_top = int(split_logits_t[0, -1].argmax().item())

    print("\n=== Summary ===")
    print(f"  monolithic OV top : {mono_top}")
    print(f"  split-LRU OV top  : {sp_top}  match={sp_top == mono_top}")
    print(f"  abs diff (vs mono): max={diff.max().item():.4e}  mean={diff.mean().item():.4e}")
    final_stats = expert_cache.stats
    print(f"  expert cache: hits={final_stats['hits']} misses={final_stats['misses']} "
          f"evictions={final_stats['evictions']}")
    print(f"  expert compile total: {expert_cache.compile_time_ms:.1f} ms  (across {final_stats['misses']} misses)")
    print(f"  backbone compile   : {backbone_compile_ms:.1f} ms")
    print(f"  total experts activated across all layers: {total_active} (of {L * E} possible)")

    assert sp_top == mono_top, "split-LRU output diverged from monolithic OV"
    print("\nEXPERT OFFLOADING (LRU): PASSED")


if __name__ == "__main__":
    main()
