"""Print Qwen/Qwen3.5-4B config and local exporter support."""
from __future__ import annotations

import argparse
from pathlib import Path

from convert_to_openvino import (
    DEFAULT_MODEL,
    check_openvino_exporter,
    check_transformers_config,
    package_version,
    read_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Qwen/Qwen3.5-4B conversion support.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF model id or local checkpoint directory.")
    parser.add_argument("--task", default="image-text-to-text")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--token", nargs="?", const="true", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config, path = read_config(args)
    model_type = str(config.get("model_type", ""))
    exporter_ok, reason = check_openvino_exporter(model_type, args.task)

    print(f"config:        {path}")
    print(f"model_type:    {model_type}")
    print(f"architectures: {config.get('architectures')}")
    print(f"text_type:     {(config.get('text_config') or {}).get('model_type')}")
    print(f"vision_type:   {(config.get('vision_config') or {}).get('model_type')}")
    print(f"transformers:  {package_version('transformers')}")
    print(f"optimum-intel: {package_version('optimum-intel')}")
    print(f"tf config ok:  {check_transformers_config(model_type)}")
    print(f"ov export ok:  {exporter_ok}")
    if not exporter_ok:
        print(f"reason:        {reason}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
