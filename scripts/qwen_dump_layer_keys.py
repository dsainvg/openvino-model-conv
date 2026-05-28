"""Dump per-layer non-expert tensor names from the safetensors index to
ground the state-dict mapping."""
import json
from pathlib import Path

MODEL_DIR = Path(r"C:\Users\intel\models\qwen36-35b-int4")
with open(MODEL_DIR / "model.safetensors.index.json") as fh:
    idx = json.load(fh)["weight_map"]

# Layer 0 (linear_attn) and layer 3 (full_attention)
for layer_idx in (0, 3):
    prefix = f"model.language_model.layers.{layer_idx}."
    names = sorted(n for n in idx if n.startswith(prefix) and "experts." not in n)
    print(f"\n{prefix}*  (non-expert, {len(names)} tensors)")
    for n in names:
        print(f"  {n[len(prefix):]}")

# Top-level / non-layer
print("\nTop-level non-layer tensors:")
top = sorted(n for n in idx if ".layers." not in n)
for n in top:
    print(f"  {n}")

# One expert's tensors (just to confirm structure)
print("\nLayer 0 expert 0 (full quant tensor names):")
names = sorted(n for n in idx if n.startswith("model.language_model.layers.0.mlp.experts.0."))
for n in names:
    print(f"  {n[len('model.language_model.layers.0.mlp.experts.0.'):]}")
