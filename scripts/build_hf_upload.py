"""Stage the Hugging Face repo contents for `bob798/deepseek-v4-toy-int4-ov`.

Builds a self-contained directory that can be loaded with
`OVModelForCausalLM.from_pretrained(REPO_ID, trust_remote_code=True, use_cache=False)`.

CONTENTS (all toy, NOT real V4-Flash):
    openvino_model.xml / openvino_model.bin    # INT4 IR (2.71 MB)
    openvino_model_fp32.xml / .bin             # FP32 IR for comparison (6.33 MB)
    config.json                                # toy config + auto_map
    configuration_deepseek_v4.py               # bundled for trust_remote_code
    modeling_deepseek_v4.py                    # bundled for trust_remote_code
    README.md                                  # model card
    .gitattributes                             # LFS rules

Run after `scripts/quantize_with_nncf.py` so the INT4 IR exists.
The actual upload happens in scripts/upload_to_hf.py.
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import torch

from deepseek_v4 import DeepseekV4ForCausalLM
from test_modeling_smoke import make_toy_config


HF_DIR_NAME = "hf_upload_toy_int4"


README = """\
---
license: mit
library_name: openvino
tags:
  - openvino
  - deepseek-v4
  - proof-of-concept
  - toy-model
  - architecture-validation
pipeline_tag: text-generation
---

# deepseek-v4-toy-int4-ov

> **WARNING — this is a TOY model with RANDOM weights, not real DeepSeek-V4-Flash.**
>
> Architecturally it is the V4-Flash design (hybrid sparse attention,
> manifold-constrained Hyper-Connections, MoE, indexer-driven KV compression),
> shrunk to 1.34M parameters with no training. It produces **arbitrary tokens**
> when run. The point of this repo is to validate the OpenVINO conversion path,
> not to do useful inference.
>
> If you're looking for real V4-Flash inference, this is not it.

## What this is

A working OpenVINO IR proof-of-concept for the DeepSeek-V4 architecture, built
on a 64 GB laptop. Includes both the FP32 IR and an INT4 weight-compressed IR
(via `nncf.compress_weights`). The same conversion code accepts the real
V4-Flash weights wherever there is enough RAM to host them.

Source repository: <https://github.com/bob798/deepseek-v4-openvino>

## Hardware requirements

| What | RAM / VRAM |
| --- | --- |
| Run this toy IR (CPU, GPU, NPU) | < 100 MB |
| Convert from PyTorch (toy) | < 1 GB |
| Convert real V4-Flash from PyTorch | ~500 GB peak (BF16 dequant) |
| Run real V4-Flash IR (after dequant + INT4) | ~140 GB |

Tested on Intel Core Ultra 9 285H + Arc 140T iGPU with OpenVINO 2026.1.0.

## Loading via optimum-intel

```python
from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer

# trust_remote_code=True pulls the bundled configuration_/modeling_ source files.
# use_cache=False because the IR is prefill-only — no past_key_values input.
model = OVModelForCausalLM.from_pretrained(
    "bob798/deepseek-v4-toy-int4-ov",
    trust_remote_code=True,
    use_cache=False,
)

# No tokenizer is shipped with this toy. Pass random input_ids:
import torch
input_ids = torch.randint(0, 512, (1, 64))
out = model(input_ids=input_ids)
print(out.logits.shape)   # torch.Size([1, 64, 512])
```

## Loading the raw IR with OpenVINO

```python
import openvino as ov
import numpy as np

core = ov.Core()
compiled = core.compile_model("openvino_model.xml", "CPU")  # or "GPU", "NPU"
input_ids = np.random.randint(0, 512, (1, 64), dtype=np.int64)
logits = compiled([input_ids])[0]
print(logits.shape)   # (1, 64, 512)
```

## Architecture summary

This toy implements every architectural feature of V4-Flash, just at a smaller
scale:

| Field | Toy value | Real V4-Flash |
| --- | --- | --- |
| `hidden_size` | 128 | 4096 |
| `num_hidden_layers` | 4 | 43 |
| `num_attention_heads` | 4 | 64 |
| `num_key_value_heads` | 1 (MQA) | 1 (MQA) |
| `head_dim` | 32 | 512 |
| `q_lora_rank` | 64 | 1024 |
| `n_routed_experts` | 8 | 256 |
| `num_experts_per_tok` | 2 | 6 |
| `n_shared_experts` | 1 | 1 |
| `compress_ratios` | `[0, 0, 4, 128]` | `[0, 0, 4, 128, 4, 128, ...]` |
| `hc_mult` | 4 | 4 |
| `hc_sinkhorn_iters` | 4 | 20 |
| total params | 1.34 M | ~284 B (~13 B active) |

Features exercised by the 4-layer toy:
- Layer 0–1: pure sliding-window attention
- Layer 2: window + indexer-driven sparse compression (compress_ratio=4)
- Layer 3: window + dense compressed-KV (compress_ratio=128)
- All layers: 4-way manifold-constrained Hyper-Connections + MoE (8 experts top-2)

## Files

| File | Size | Notes |
| --- | --- | --- |
| `openvino_model.xml`/`.bin` | 2.71 MB | INT4 weight-compressed IR (NNCF, `INT4_ASYM`, group_size=32) |
| `openvino_model_fp32.xml`/`.bin` | 6.33 MB | FP32 baseline IR for comparison |
| `config.json` | — | toy config with `auto_map` |
| `configuration_deepseek_v4.py` | — | bundled for `trust_remote_code` |
| `modeling_deepseek_v4.py` | — | bundled for `trust_remote_code` |

## Numerics

| IR | Size | Greedy next-token (B=1, seed=0) |
| --- | --- | --- |
| FP32 | 6.33 MB | 78 |
| INT8 (not shipped here) | 3.03 MB | 78 |
| INT4 (shipped) | 2.71 MB | 78 |

INT4 vs FP32: max abs diff 2.9e-1, mean 4.3e-2 on the toy. Greedy token matches.

## Known limitations

- **Toy weights only.** Loading this model and generating text will produce
  random gibberish. This repo exists to validate the conversion path, not to
  serve inference.
- **Prefill-only.** The IR has no `past_key_values` input. Use `use_cache=False`
  with `OVModelForCausalLM.from_pretrained`. Autoregressive decode will require
  KV-cache plumbing that isn't done yet.
- **B=2 numerical drift.** A single batch element diverges in greedy token
  vs. PyTorch — FP rounding-order, not a topology bug.
- **MTP and hash routing not implemented.** Multi-Token Prediction blocks and
  the hash-routing tables of the first 3 V4-Flash layers are not in the toy
  (`num_nextn_predict_layers=0`, `num_hash_layers=0`).
- **Not yet registered upstream.** As of `optimum-intel` 1.27.0, native
  `model_type="deepseek_v4"` is not registered; this repo uses
  `trust_remote_code=True` instead. See the upstream issue tracked at
  <https://github.com/bob798/deepseek-v4-openvino/blob/main/UPSTREAM_ISSUE.md>.

## License

MIT. Architecture is derived from the deepseek-ai V4-Flash reference (also MIT)
at <https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash>. The PyTorch port and
the OpenVINO conversion code are original.

## Citation / attribution

If you build on this work, please link
<https://github.com/bob798/deepseek-v4-openvino>.
"""


GITATTRIBUTES = """\
*.bin filter=lfs diff=lfs merge=lfs -text
*.safetensors filter=lfs diff=lfs merge=lfs -text
*.xml filter=lfs diff=lfs merge=lfs -text
"""


def main():
    out_dir = ROOT / HF_DIR_NAME
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir()

    # 1. Build a fresh toy model and dump config (so config.json has auto_map).
    cfg = make_toy_config()
    cfg.auto_map = {
        "AutoConfig": "configuration_deepseek_v4.DeepseekV4Config",
        "AutoModelForCausalLM": "modeling_deepseek_v4.DeepseekV4ForCausalLM",
    }
    cfg.architectures = ["DeepseekV4ForCausalLM"]
    cfg.save_pretrained(out_dir)
    print(f"  wrote {out_dir / 'config.json'}")

    # 2. Bundle modeling + configuration source files for trust_remote_code.
    src_dir = ROOT / "src" / "deepseek_v4"
    for fname in ("configuration_deepseek_v4.py", "modeling_deepseek_v4.py"):
        shutil.copy(src_dir / fname, out_dir / fname)
        print(f"  copied {fname}")

    # 3. Copy the INT4 IR as the primary openvino_model.{xml,bin}.
    int4_xml = ROOT / "ov_ir_toy" / "deepseek_v4_toy_int4.xml"
    int4_bin = ROOT / "ov_ir_toy" / "deepseek_v4_toy_int4.bin"
    if not int4_xml.exists():
        raise SystemExit(
            f"INT4 IR not found at {int4_xml}. Run scripts/quantize_with_nncf.py first."
        )
    shutil.copy(int4_xml, out_dir / "openvino_model.xml")
    shutil.copy(int4_bin, out_dir / "openvino_model.bin")
    print(f"  copied INT4 IR -> openvino_model.xml/.bin")

    # 4. Also include the FP32 IR for comparison.
    fp32_xml = ROOT / "ov_ir_toy" / "deepseek_v4_toy.xml"
    fp32_bin = ROOT / "ov_ir_toy" / "deepseek_v4_toy.bin"
    shutil.copy(fp32_xml, out_dir / "openvino_model_fp32.xml")
    shutil.copy(fp32_bin, out_dir / "openvino_model_fp32.bin")
    print(f"  copied FP32 IR -> openvino_model_fp32.xml/.bin")

    # 5. README + .gitattributes (Hugging Face uses LFS for >10 MB; ours are <10 MB
    #    but the rules are still good hygiene).
    (out_dir / "README.md").write_text(README, encoding="utf-8")
    (out_dir / ".gitattributes").write_text(GITATTRIBUTES, encoding="utf-8")
    print(f"  wrote README.md ({len(README)} chars) + .gitattributes")

    print(f"\n  staged dir: {out_dir}")
    print(f"  files:")
    for p in sorted(out_dir.iterdir()):
        size = p.stat().st_size
        print(f"    {p.name:42s}  {size / 1024:8.1f} KB")


if __name__ == "__main__":
    main()
