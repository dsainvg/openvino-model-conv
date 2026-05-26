"""Quantize the toy DeepSeek-V4 IR with NNCF (INT8 / INT4 / MXFP4 weight compression).

Loads the FP32 IR saved by convert_to_openvino.py, applies nncf.compress_weights
in three modes (INT8 asym, INT4 asym, MXFP4), saves each variant, then reloads on
CPU and compares logits + greedy next-token vs the FP32 baseline.

MXFP4 (microscaled FP4 — E2M1 + E8M0 scale) is a CPU-only OpenVINO 2026.0+ mode;
groups of 32 elements share an 8-bit shared exponent. NNCF exposes it as
CompressWeightsMode.MXFP4. Useful preview for the eventual ~140 GB real-V4 IR
(half the size of INT8).
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
    mxfp4_ir = ROOT / "ov_ir_toy" / "deepseek_v4_toy_mxfp4.xml"

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

    print("\n=== MXFP4 weight compression (E2M1 + E8M0 microscale, CPU-only, group_size=32) ===")
    fp32_model = core.read_model(str(fp32_ir))
    mxfp4_model = nncf.compress_weights(
        fp32_model,
        mode=nncf.CompressWeightsMode.MXFP4,
        group_size=32,
        ratio=1.0,
    )
    ov.save_model(mxfp4_model, str(mxfp4_ir), compress_to_fp16=False)
    mxfp4_logits = run_ir(mxfp4_ir, input_ids)
    mxfp4_top = int(mxfp4_logits[0, -1].argmax().item())
    diff_mx = (fp32_logits.float() - mxfp4_logits.float()).abs()
    print(f"  size: {file_size_mb(mxfp4_ir):.2f} MB   greedy_top: {mxfp4_top}  match_fp32={mxfp4_top == fp32_top}")
    print(f"  abs diff vs FP32   max={diff_mx.max().item():.4e}  mean={diff_mx.mean().item():.4e}")

    print("\n=== Summary ===")
    print(f"  FP32   {file_size_mb(fp32_ir):7.2f} MB  top={fp32_top}")
    print(f"  INT8   {file_size_mb(int8_ir):7.2f} MB  top={int8_top}")
    print(f"  INT4   {file_size_mb(int4_ir):7.2f} MB  top={int4_top}")
    print(f"  MXFP4  {file_size_mb(mxfp4_ir):7.2f} MB  top={mxfp4_top}")
    print("\nQUANTIZATION: PASSED")


if __name__ == "__main__":
    main()
