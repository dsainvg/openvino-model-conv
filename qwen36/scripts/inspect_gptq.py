"""Inspect one expert's GPTQ tensors to ground the dequant implementation.

Prints shapes, dtypes, and a sample of qzeros (to determine the zero offset
convention: 7 means gptqmodel/-1 store, 8 means raw symmetric center).
"""
import json
from pathlib import Path

import torch
from safetensors import safe_open

MODEL_DIR = Path(r"C:\Users\intel\models\qwen36-35b-int4")
INDEX = MODEL_DIR / "model.safetensors.index.json"

with open(INDEX) as fh:
    idx = json.load(fh)
weight_map = idx["weight_map"]

# Look at layer 0 expert 0 down_proj
target_prefix = "model.language_model.layers.0.mlp.experts.0.down_proj"
tensors = {k: v for k, v in weight_map.items() if k.startswith(target_prefix)}
print("Tensors for", target_prefix)
for name, shard in tensors.items():
    print(f"  {name:60s}  in shard: {shard}")

# Open the shard and inspect
print()
shard_path = MODEL_DIR / list(tensors.values())[0]
with safe_open(shard_path, framework="pt") as fh:
    for name in tensors:
        t = fh.get_tensor(name)
        leaf = name.split(".")[-1]
        print(f"  {leaf:10s} shape={str(tuple(t.shape)):20s} dtype={str(t.dtype):20s} ", end="")
        if leaf in ("qweight", "qzeros", "g_idx") and t.dtype in (torch.int32, torch.int64):
            print(f"min={t.min().item():>12d} max={t.max().item():>12d}")
        elif t.is_floating_point():
            print(f"min={t.float().min().item():>+.4e} max={t.float().max().item():>+.4e}")
        else:
            print(f"min={t.min().item()} max={t.max().item()}")

# For qzeros: unpack the first int32 and report the 8 int4 values inside.
# If sym=True and gptqmodel convention, expect all 7s. If "raw symmetric", expect 8s.
print()
print("qzeros[0, 0] unpacked (8 int4 values):")
with safe_open(shard_path, framework="pt") as fh:
    qzeros = fh.get_tensor(target_prefix + ".qzeros")
v = qzeros[0, 0].item() & 0xFFFFFFFF
nibbles = [(v >> (4 * j)) & 0xF for j in range(8)]
print(f"  {nibbles}  (all same -> uniform zero per group; 7 -> gptqmodel offset, 8 -> raw)")
print()
print(f"qzeros shape: {tuple(qzeros.shape)}")
print(f"qzeros first 5 packed ints (per group): {[hex(qzeros[g, 0].item() & 0xFFFFFFFF) for g in range(min(5, qzeros.shape[0]))]}")

# g_idx: with desc_act=False, should be range(in_features) // 128
print()
with safe_open(shard_path, framework="pt") as fh:
    g_idx = fh.get_tensor(target_prefix + ".g_idx")
expected = torch.arange(g_idx.shape[0]) // 128
matches = bool((g_idx == expected).all().item())
print(f"g_idx shape={tuple(g_idx.shape)}  matches arange//128: {matches}")
if not matches:
    diffs = (g_idx != expected).nonzero().flatten()[:8]
    print(f"  mismatch at indices {diffs.tolist()}: actual {g_idx[diffs].tolist()} vs expected {expected[diffs].tolist()}")

# Print expected weight matrix dims
text_cfg = json.load(open(MODEL_DIR / "config.json"))["text_config"]
print()
print(f"Expert down_proj logical shape: ({text_cfg['moe_intermediate_size']}, {text_cfg['hidden_size']}) = (intermediate, hidden)")
print(f"in_features = intermediate = {text_cfg['moe_intermediate_size']}")
print(f"out_features = hidden = {text_cfg['hidden_size']}")
print(f"in_features // 8 = {text_cfg['moe_intermediate_size'] // 8}, num_groups = {text_cfg['moe_intermediate_size'] // 128}")
