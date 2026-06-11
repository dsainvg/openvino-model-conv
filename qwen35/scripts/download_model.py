"""Download Qwen/Qwen3.5-4B weights from HuggingFace.

Uses snapshot_download so the checkpoint is fully cached; re-running is safe
and instant if the files already exist.

Usage:
    python scripts/download_model.py
    python scripts/download_model.py --output /path/to/dir
    python scripts/download_model.py --model Qwen/Qwen3.5-4B-Instruct
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download Qwen3.5-4B from HuggingFace.")
    p.add_argument("--model",  default="Qwen/Qwen3.5-4B", help="HF repo id.")
    p.add_argument("--output", type=Path, default=None,
                   help="Local directory to download into. "
                        "Defaults to <repo_root>/models/<model_name>.")
    p.add_argument("--token",  default=None,
                   help="HF token (or set HF_TOKEN env var).")
    p.add_argument("--revision", default="main")
    return p.parse_args()


def main() -> int:
    import os
    from huggingface_hub import snapshot_download

    args   = parse_args()
    token  = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    repo_root = Path(__file__).resolve().parents[1]
    if args.output is None:
        model_name = args.model.split("/")[-1]
        output_dir = repo_root.parent / "models" / model_name
    else:
        output_dir = args.output.resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    # Skip download if safetensors shards are already present
    existing = list(output_dir.glob("*.safetensors"))
    if existing:
        print(f"Already downloaded ({len(existing)} shard(s)) at: {output_dir}")
        return 0

    print(f"Downloading {args.model} → {output_dir} ...")
    snapshot_download(
        repo_id         = args.model,
        local_dir       = str(output_dir),
        revision        = args.revision,
        token           = token,
        ignore_patterns = ["*.msgpack", "*.h5", "flax_model*", "tf_model*",
                           "rust_model*", "onnx/*"],
    )

    shards = list(output_dir.glob("*.safetensors"))
    print(f"\nDownload complete — {len(shards)} shard(s) at: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
