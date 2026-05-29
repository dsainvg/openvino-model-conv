"""Upload the staged HF directory to bob798/deepseek-v4-toy-int4-ov.

Prerequisites:
    1. Run scripts/build_hf_upload.py to populate hf_upload_toy_int4/
    2. Be logged into Hugging Face: `huggingface-cli login` or `hf auth login`,
       OR set HF_TOKEN env var.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

REPO_ID_DEFAULT = "imbob798/deepseek-v4-toy-int4-ov"
LOCAL_DIR = ROOT / "hf_upload_toy_int4"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=REPO_ID_DEFAULT)
    parser.add_argument("--local-dir", type=Path, default=LOCAL_DIR)
    parser.add_argument("--private", action="store_true",
                        help="Create as private repo (default: public).")
    parser.add_argument("--commit-message", default="Initial upload: toy V4 INT4 IR + model card")
    args = parser.parse_args()

    if not args.local_dir.exists():
        raise SystemExit(
            f"{args.local_dir} not found. Run scripts/build_hf_upload.py first."
        )

    from huggingface_hub import HfApi, create_repo, whoami

    try:
        user = whoami()
        print(f"  logged in as: {user.get('name', user)}")
    except Exception as e:
        raise SystemExit(
            f"Not logged into Hugging Face: {e}\n"
            "Run `huggingface-cli login` (or `hf auth login`) first, or set HF_TOKEN."
        )

    print(f"\n=== Creating repo {args.repo_id} (private={args.private}) ===")
    create_repo(
        repo_id=args.repo_id,
        private=args.private,
        repo_type="model",
        exist_ok=True,
    )

    print(f"\n=== Uploading {args.local_dir} -> {args.repo_id} ===")
    api = HfApi()
    api.upload_folder(
        folder_path=str(args.local_dir),
        repo_id=args.repo_id,
        repo_type="model",
        commit_message=args.commit_message,
    )
    print(f"\n  uploaded -> https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
