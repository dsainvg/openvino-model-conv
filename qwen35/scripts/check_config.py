"""Print Qwen/Qwen3.5-4B config and package versions."""
from __future__ import annotations

import argparse
import json
import importlib.metadata
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Qwen/Qwen3.5-4B configuration.")
    parser.add_argument("--model-dir", type=Path, default=None,
                        help="Path to the downloaded model checkpoint directory.")
    return parser.parse_args()


def get_pkg_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def main() -> int:
    args = parse_args()
    
    print("Package Versions:")
    print(f"  transformers:  {get_pkg_version('transformers')}")
    print(f"  openvino:      {get_pkg_version('openvino')}")
    print(f"  optimum-intel: {get_pkg_version('optimum-intel')}")
    
    if args.model_dir:
        config_path = Path(args.model_dir) / "config.json"
        if not config_path.exists():
            print(f"\nERROR: config.json not found in {args.model_dir}")
            return 1
        
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            
        text_cfg = cfg.get("text_config", cfg)
        print(f"\nModel Configuration ({config_path}):")
        print(f"  model_type:             {cfg.get('model_type', 'unknown')}")
        print(f"  hidden_size:            {text_cfg.get('hidden_size')}")
        print(f"  num_hidden_layers:      {text_cfg.get('num_hidden_layers')}")
        print(f"  vocab_size:             {text_cfg.get('vocab_size')}")
        print(f"  full_attention_interval: {text_cfg.get('full_attention_interval', 4)}")
        print(f"  layer_types:            {text_cfg.get('layer_types')}")
    else:
        print("\nPass --model-dir to print configuration of a downloaded model.")
        
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

