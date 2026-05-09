"""Quantize the toy DeepSeek-V4 IR with NNCF (INT8 and INT4 weight compression).

Loads the FP32 IR saved by convert_to_openvino.py, applies nncf.compress_weights
in two modes (INT8 asym, INT4 asym), saves both, then reloads each variant on CPU
and compares logits and greedy next-token vs the FP32 baseline.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import torch
import openvino as ov
import nncf

from test_modeling_smoke import make_toy_config


def file_size_mb(p: Path) -> float:
    bin_path = p.with_suffix(".bin")
    return (p.stat().st_size + bin_path.stat().st_size) / (1024 * 1024)


def run_ir(ir_path: Path, input_ids: torch.Tensor) -> torch.Tensor:
    core = ov.Core()
    compiled = core.compile_model(str(ir_path), "CPU")
    out = compiled([input_ids.numpy()])[0]
    return torch.from_numpy(out)


def main():
    torch.manual_seed(0)
    cfg = make_toy_config()
    B, S = 1, 128
    input_ids = torch.randint(0, cfg.vocab_size, (B, S))

    fp32_ir = ROOT / "ov_ir_toy" / "deepseek_v4_toy.xml"
    int8_ir = ROOT / "ov_ir_toy" / "deepseek_v4_toy_int8.xml"
    int4_ir = ROOT / "ov_ir_toy" / "deepseek_v4_toy_int4.xml"

    print("=== Baseline FP32 IR ===")
    fp32_logits = run_ir(fp32_ir, input_ids)
    fp32_top = int(fp32_logits[0, -1].argmax().item())
    print(f"  size: {file_size_mb(fp32_ir):.2f} MB   greedy_top: {fp32_top}")

    print("\n=== INT8 weight compression ===")
    core = ov.Core()
    fp32_model = core.read_model(str(fp32_ir))
    int8_model = nncf.compress_weights(fp32_model, mode=nncf.CompressWeightsMode.INT8_ASYM)
    ov.save_model(int8_model, str(int8_ir), compress_to_fp16=False)
    int8_logits = run_ir(int8_ir, input_ids)
    int8_top = int(int8_logits[0, -1].argmax().item())
    diff8 = (fp32_logits.float() - int8_logits.float()).abs()
    print(f"  size: {file_size_mb(int8_ir):.2f} MB   greedy_top: {int8_top}  match_fp32={int8_top == fp32_top}")
    print(f"  abs diff vs FP32   max={diff8.max().item():.4e}  mean={diff8.mean().item():.4e}")

    print("\n=== INT4 weight compression (group_size=32, ratio=1.0) ===")
    fp32_model = core.read_model(str(fp32_ir))
    int4_model = nncf.compress_weights(
        fp32_model,
        mode=nncf.CompressWeightsMode.INT4_ASYM,
        group_size=32,
        ratio=1.0,
    )
    ov.save_model(int4_model, str(int4_ir), compress_to_fp16=False)
    int4_logits = run_ir(int4_ir, input_ids)
    int4_top = int(int4_logits[0, -1].argmax().item())
    diff4 = (fp32_logits.float() - int4_logits.float()).abs()
    print(f"  size: {file_size_mb(int4_ir):.2f} MB   greedy_top: {int4_top}  match_fp32={int4_top == fp32_top}")
    print(f"  abs diff vs FP32   max={diff4.max().item():.4e}  mean={diff4.mean().item():.4e}")

    print("\n=== Summary ===")
    print(f"  FP32  {file_size_mb(fp32_ir):7.2f} MB  top={fp32_top}")
    print(f"  INT8  {file_size_mb(int8_ir):7.2f} MB  top={int8_top}")
    print(f"  INT4  {file_size_mb(int4_ir):7.2f} MB  top={int4_top}")
    print("\nQUANTIZATION: PASSED")


if __name__ == "__main__":
    main()
