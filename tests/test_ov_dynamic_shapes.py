"""Verify the saved IR runs on different (B, S) shapes than what was traced.

Trace input was [1, 128]. Try [1, 64], [1, 256], [2, 128] and confirm OpenVINO
produces logits matching the PyTorch reference (greedy token match)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import torch
import openvino as ov

from deepseek_v4 import DeepseekV4ForCausalLM
from test_modeling_smoke import make_toy_config


def main():
    torch.manual_seed(0)
    cfg = make_toy_config()
    model = DeepseekV4ForCausalLM(cfg).eval()

    ir_path = ROOT / "ov_ir_toy" / "deepseek_v4_toy.xml"
    core = ov.Core()
    compiled = core.compile_model(str(ir_path), "CPU")

    cases = [(1, 64), (1, 128), (1, 256), (2, 128)]
    for B, S in cases:
        ids = torch.randint(0, cfg.vocab_size, (B, S))
        with torch.inference_mode():
            ref = model(input_ids=ids).logits
        try:
            out = compiled([ids.numpy()])[0]
        except Exception as e:
            print(f"  shape ({B},{S}): OpenVINO failed -- {type(e).__name__}: {str(e)[:200]}")
            continue
        ov_logits = torch.from_numpy(out)
        ok_shape = tuple(ov_logits.shape) == (B, S, cfg.vocab_size)
        ref_top = ref[:, -1].argmax(-1)
        ov_top = ov_logits[:, -1].argmax(-1)
        match = bool((ref_top == ov_top).all())
        diff = (ref.float() - ov_logits.float()).abs()
        print(f"  shape ({B},{S}): ov_shape={tuple(ov_logits.shape)} shape_ok={ok_shape} "
              f"top_match={match} abs_diff_mean={diff.mean().item():.2e} max={diff.max().item():.2e}")


if __name__ == "__main__":
    main()
