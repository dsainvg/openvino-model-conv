"""P0.3 — compile + run each toy IR on CPU and on the Arc 140T iGPU, record results.

Outputs a JSON record (per IR × per device) with compile_ms, infer_ms, abs/rel diff vs CPU,
and greedy-top equivalence. The KV-cache IR is exercised in both prefill (S_past=0) and
one-step decode modes.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import openvino as ov
import torch

from test_modeling_smoke import make_toy_config


def _bench(compiled, feed, n_warmup=2, n_iters=10):
    for _ in range(n_warmup):
        compiled(feed)
    t0 = time.perf_counter()
    for _ in range(n_iters):
        out = compiled(feed)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0 / n_iters
    return out, elapsed_ms


def _device_info(core, device):
    try:
        return core.get_property(device, "FULL_DEVICE_NAME")
    except Exception as e:
        return f"<error: {e}>"


def bench_simple_ir(core, ir_path, input_ids_np, results, label, devices=("CPU", "GPU")):
    print(f"\n=== {label} :: {ir_path.name} ===")
    cpu_logits = None
    for dev in devices:
        rec = {"ir": ir_path.name, "label": label, "device": dev, "ov_version": ov.__version__,
               "device_full_name": _device_info(core, dev)}
        try:
            t0 = time.perf_counter()
            compiled = core.compile_model(str(ir_path), dev)
            rec["compile_ms"] = (time.perf_counter() - t0) * 1000.0
            out, infer_ms = _bench(compiled, [input_ids_np])
            rec["infer_ms"] = infer_ms
            logits = torch.from_numpy(out[0])
            rec["logits_shape"] = list(logits.shape)
            rec["logits_finite"] = bool(torch.isfinite(logits).all().item())
            top = int(logits[0, -1].argmax().item())
            rec["greedy_top"] = top
            if dev == "CPU":
                cpu_logits = logits
                rec["status"] = "ok"
            else:
                diff = (cpu_logits.float() - logits.float()).abs()
                rec["abs_max_vs_cpu"] = float(diff.max().item())
                rec["abs_mean_vs_cpu"] = float(diff.mean().item())
                rec["top_match_cpu"] = bool(top == int(cpu_logits[0, -1].argmax().item()))
                rec["status"] = "ok"
            print(f"  {dev:>4}: compile {rec['compile_ms']:8.2f}ms infer {rec['infer_ms']:7.2f}ms "
                  f"top={top}" + (f"  abs_max_vs_cpu={rec['abs_max_vs_cpu']:.3e}" if dev != "CPU" else ""))
        except Exception as e:
            rec["status"] = "fail"
            rec["error_text"] = f"{type(e).__name__}: {str(e)[:500]}"
            print(f"  {dev:>4}: FAIL  {rec['error_text']}")
        results.append(rec)


def bench_kv_ir(core, ir_path, cfg, results, devices=("CPU", "GPU")):
    print(f"\n=== KV-cache IR :: {ir_path.name} ===")
    L = cfg.num_hidden_layers
    H = cfg.hc_mult
    dim = cfg.hidden_size
    B, S = 1, 128

    rng = np.random.default_rng(0)
    input_ids_prefill = rng.integers(0, cfg.vocab_size, size=(B, S)).astype(np.int64)
    # Pre-generate the decode token so both devices see identical inputs (so the only source
    # of CPU↔GPU divergence is the prefill past tensor, which we expect to differ slightly).
    new_ids = rng.integers(0, cfg.vocab_size, size=(B, 1)).astype(np.int64)
    empty_past = {f"past_x_layer_{i}": np.zeros((B, 0, H, dim), dtype=np.float32) for i in range(L)}
    prefill_feed = {"input_ids": input_ids_prefill, **empty_past}

    cpu_logits = cpu_past = cpu_decode_top = None
    for dev in devices:
        rec_pre = {"ir": ir_path.name, "label": "kv_prefill", "device": dev, "ov_version": ov.__version__,
                   "device_full_name": _device_info(core, dev)}
        try:
            t0 = time.perf_counter()
            compiled = core.compile_model(str(ir_path), dev)
            rec_pre["compile_ms"] = (time.perf_counter() - t0) * 1000.0
            out, infer_ms = _bench(compiled, prefill_feed, n_warmup=1, n_iters=3)
            rec_pre["infer_ms"] = infer_ms
            logits = torch.from_numpy(out["logits"])
            past = [torch.from_numpy(out[f"present_x_layer_{i}"]) for i in range(L)]
            top = int(logits[0, -1].argmax().item())
            rec_pre["logits_shape"] = list(logits.shape)
            rec_pre["greedy_top"] = top
            rec_pre["status"] = "ok"
            if dev == "CPU":
                cpu_logits, cpu_past = logits, past
            else:
                diff = (cpu_logits - logits).abs()
                rec_pre["abs_max_vs_cpu"] = float(diff.max().item())
                rec_pre["top_match_cpu"] = bool(top == int(cpu_logits[0, -1].argmax().item()))
            print(f"  {dev:>4} prefill: compile {rec_pre['compile_ms']:8.2f}ms infer {rec_pre['infer_ms']:7.2f}ms "
                  f"top={top}" + (f"  abs_max_vs_cpu={rec_pre['abs_max_vs_cpu']:.3e}" if dev != "CPU" else ""))
            results.append(rec_pre)

            # Decode step (S_new=1, S_past=S). Use the same new_ids for both devices;
            # each device feeds its own prefill past tensor.
            rec_dec = {"ir": ir_path.name, "label": "kv_decode_step", "device": dev,
                       "ov_version": ov.__version__, "device_full_name": _device_info(core, dev)}
            decode_feed = {"input_ids": new_ids}
            for i in range(L):
                decode_feed[f"past_x_layer_{i}"] = past[i].numpy()
            out_d, infer_ms_d = _bench(compiled, decode_feed, n_warmup=1, n_iters=5)
            rec_dec["compile_ms"] = rec_pre["compile_ms"]  # same compile
            rec_dec["infer_ms"] = infer_ms_d
            logits_d = torch.from_numpy(out_d["logits"])
            top_d = int(logits_d[0, 0].argmax().item())
            rec_dec["logits_shape"] = list(logits_d.shape)
            rec_dec["greedy_top"] = top_d
            rec_dec["status"] = "ok"
            note = ""
            if dev == "CPU":
                cpu_decode_top = top_d
            else:
                rec_dec["top_match_cpu"] = bool(top_d == cpu_decode_top)
                note = f"  cpu_top={cpu_decode_top}  match={top_d == cpu_decode_top}"
            print(f"  {dev:>4} decode : infer {rec_dec['infer_ms']:7.2f}ms  top={top_d}{note}")
            results.append(rec_dec)
        except Exception as e:
            rec_pre.setdefault("status", "fail")
            rec_pre.setdefault("error_text", f"{type(e).__name__}: {str(e)[:500]}")
            print(f"  {dev:>4}: FAIL  {rec_pre.get('error_text')}")
            results.append(rec_pre)


def main():
    core = ov.Core()
    print(f"OpenVINO {ov.__version__}")
    print(f"available devices: {core.available_devices}")
    for d in core.available_devices:
        try:
            print(f"  {d}: {core.get_property(d, 'FULL_DEVICE_NAME')}")
        except Exception as e:
            print(f"  {d}: <{type(e).__name__}: {e}>")

    cfg = make_toy_config()
    B, S = 1, 128
    rng = np.random.default_rng(0)
    input_ids_np = rng.integers(0, cfg.vocab_size, size=(B, S)).astype(np.int64)

    results = []
    devices = ("CPU", "GPU")

    simple_irs = [
        (ROOT / "ov_ir_toy" / "deepseek_v4_toy.xml", "fp32_prefill"),
        (ROOT / "ov_ir_toy" / "deepseek_v4_toy_int8.xml", "int8_prefill"),
        (ROOT / "ov_ir_toy" / "deepseek_v4_toy_int4.xml", "int4_prefill"),
        (ROOT / "ov_ir_toy" / "deepseek_v4_toy_mxfp4.xml", "mxfp4_prefill"),
    ]
    for ir, label in simple_irs:
        if ir.exists():
            bench_simple_ir(core, ir, input_ids_np, results, label, devices=devices)
        else:
            print(f"\n[skip] {ir.name} not found")

    kv_ir = ROOT / "ov_ir_toy" / "deepseek_v4_toy_kv.xml"
    if kv_ir.exists():
        bench_kv_ir(core, kv_ir, cfg, results, devices=devices)

    out_path = ROOT / "ov_ir_toy" / "igpu_bench_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {out_path}")
    pass_count = sum(1 for r in results if r.get("status") == "ok")
    fail_count = sum(1 for r in results if r.get("status") == "fail")
    print(f"Summary: {pass_count} ok, {fail_count} fail")


if __name__ == "__main__":
    main()
