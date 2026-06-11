"""Push the converted OpenVINO IR to a new HuggingFace repository.

Creates the repo if it doesn't exist, writes a model card, then uploads
the entire IR directory via upload_folder.

Usage:
    python scripts/push_to_hf.py --ir-dir /path/to/ov_ir_qwen35_4b \\
        --repo-name qwen35-4b-openvino-fp16
    python scripts/push_to_hf.py --ir-dir /path/to/ov_ir_qwen35_4b \\
        --repo-name qwen35-4b-openvino-fp16 --private
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Push Qwen3.5-4B OpenVINO IR to HuggingFace.")
    p.add_argument("--ir-dir",    type=Path, required=True,
                   help="Directory containing openvino_model.xml + .bin (+ tokenizer files).")
    p.add_argument("--repo-name", required=True,
                   help="HuggingFace repo name (without username prefix).")
    p.add_argument("--base-model", default="Qwen/Qwen3.5-4B",
                   help="Source HF model id (for the model card).")
    p.add_argument("--token",   default=None,
                   help="HF token (or set HF_TOKEN env var).")
    p.add_argument("--private", action="store_true",
                   help="Create a private repository.")
    return p.parse_args()


def main() -> int:
    from huggingface_hub import HfApi, create_repo, upload_folder

    args  = parse_args()
    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    if not token:
        print("ERROR: no HF token found. Pass --token or set HF_TOKEN env var.", file=sys.stderr)
        return 1

    ir_dir = args.ir_dir.resolve()
    if not ir_dir.exists():
        print(f"ERROR: IR directory not found: {ir_dir}", file=sys.stderr)
        return 1

    xml_files = list(ir_dir.glob("*.xml"))
    if not xml_files:
        print(f"ERROR: no .xml files found in {ir_dir}", file=sys.stderr)
        return 1

    api      = HfApi(token=token)
    username = api.whoami()["name"]
    full_id  = f"{username}/{args.repo_name}"

    print(f"Logged in as : {username}")
    print(f"Repo         : {full_id}  (private={args.private})")

    create_repo(
        repo_id   = full_id,
        repo_type = "model",
        private   = args.private,
        exist_ok  = True,
        token     = token,
    )

    # Write model card
    readme = f"""---
library_name: openvino
base_model: {args.base_model}
tags:
  - openvino
  - qwen3.5
  - text-generation
  - fp16
license: apache-2.0
---

# Qwen3.5-4B — OpenVINO IR (FP16)

OpenVINO intermediate-representation export of [{args.base_model}](https://huggingface.co/{args.base_model}).

Converted on {date.today()} with [`dsainvg/openvino-model-conv`](https://github.com/dsainvg/openvino-model-conv)
via `qwen35/scripts/convert_to_openvino.py` — pure `ov.convert_model` on the PyTorch model
(no optimum-intel exporter).

## Load with OpenVINO

```python
import openvino as ov
import numpy as np
from transformers import AutoTokenizer

core     = ov.Core()
compiled = core.compile_model("openvino_model.xml", "CPU")
tok      = AutoTokenizer.from_pretrained("{full_id}")

inputs    = tok("Hello, world!", return_tensors="np")
# Build state tensors (zeros) matching your max_seq budget, then call compiled(...)
```

## Files

| File | Description |
|------|-------------|
| `openvino_model.xml` / `.bin` | OpenVINO IR (FP16 weights) |
| `tokenizer.json`, `tokenizer_config.json` | Tokeniser assets |
| `config.json`, `generation_config.json` | Model config |
"""
    (ir_dir / "README.md").write_text(readme, encoding="utf-8")
    print("Model card written.")

    print(f"\nUploading {ir_dir} → {full_id} ...")
    upload_folder(
        folder_path     = str(ir_dir),
        repo_id         = full_id,
        repo_type       = "model",
        commit_message  = "Add Qwen3.5-4B OpenVINO FP16 IR (ov.convert_model)",
        ignore_patterns = ["__pycache__", "*.pyc"],
        token           = token,
    )

    print(f"\n✓ Done! → https://huggingface.co/{full_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
