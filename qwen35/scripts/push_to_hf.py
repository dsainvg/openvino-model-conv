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

# Qwen3.5-4B — OpenVINO IR (Stateful NPU-Ready)

OpenVINO stateful intermediate-representation export of [{args.base_model}](https://huggingface.co/{args.base_model}).

Converted on {date.today()} with [`dsainvg/openvino-model-conv`](https://github.com/dsainvg/openvino-model-conv)
via `qwen35/scripts/convert_to_openvino.py` — stateful single-model tracing.

## Load with OpenVINO GenAI

This model is NPU-compatible and ready to load using `openvino_genai`:

```python
import openvino_genai as ov_genai

# Load model (automatically downloads from HuggingFace on first run)
pipe = ov_genai.LLMPipeline("{full_id}", "CPU") # or "NPU"
print(pipe.generate("Hello, my name is", max_new_tokens=64))
```

## Files

| File | Description |
|------|-------------|
| `openvino_model.xml` / `.bin` | OpenVINO IR (FP16 weights) |
| `openvino_tokenizer.xml` / `.bin` | Tokenizer |
| `openvino_detokenizer.xml` / `.bin` | Detokenizer |
| `config.json`, `generation_config.json` | Model config |
"""
    import tempfile
    import shutil

    with tempfile.TemporaryDirectory(prefix="hf_upload_") as temp_dir:
        upload_dir = Path(temp_dir)
        print(f"\n[1/2] Creating clean upload directory at {upload_dir} ...")

        # Copy only required files (exclude split IR layers, embedding, lm_head, and safetensors metadata index)
        exclude_prefixes = ("layer_", "embed_tokens", "lm_head")
        
        copied_count = 0
        for item in ir_dir.iterdir():
            if item.is_file():
                if item.name.startswith(exclude_prefixes) or "safetensors" in item.name:
                    print(f"  Ignoring file: {item.name}")
                    continue
                shutil.copy2(item, upload_dir / item.name)
                print(f"  Copying: {item.name}")
                copied_count += 1

        if copied_count == 0:
            print("ERROR: No files copied to clean upload directory.", file=sys.stderr)
            return 1

        # Write model card to the clean upload directory
        (upload_dir / "README.md").write_text(readme, encoding="utf-8")
        print("Model card written.")

        print(f"\n[2/2] Uploading {upload_dir} → {full_id} ...")
        upload_folder(
            folder_path     = str(upload_dir),
            repo_id         = full_id,
            repo_type       = "model",
            commit_message  = "Add Qwen3.5-4B OpenVINO FP16 IR (stateful single-model)",
            ignore_patterns = ["__pycache__", "*.pyc"],
            token           = token,
        )

    print(f"\n✓ Done! Clean upload complete → https://huggingface.co/{full_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
