"""Load real DeepSeek-V4-Flash weights into our DeepseekV4ForCausalLM.

Real V4-Flash:
    - ~284B params, ~160 GB on disk (FP8 main weights + FP4 expert weights, both
      with E8M0-microscaled per-block scales)
    - dequantizing to BF16 needs roughly 500 GB of host RAM

Three operating modes:

1. `--dry-run` — reads only `model.safetensors.index.json` to verify that every
   real-V4 parameter name maps to a valid slot in our PyTorch module tree (or is
   on the explicit skip list). No tensor loading. Always safe.

2. (default) `--full-bf16` — full shard-by-shard dequantization to BF16
   safetensors. Peak RAM ~500 GB; will OOM on this 64 GB host. Documented for
   the cloud / large-RAM box.

3. `--per-expert-ir` — Direction-2 path. Loads + dequantizes ONE expert at a
   time, builds a tiny per-expert PyTorch module, traces with `ov.convert_model`,
   writes the IR to disk, frees memory. Backbone (non-expert) weights are
   collected into a single BF16 safetensors. Peak RAM ~5-10 GB (backbone + one
   expert), fits comfortably on a 64 GB host. Output layout:

       <output>/backbone.safetensors           BF16 backbone params
       <output>/expert_L{i}_E{e}.{xml,bin}     one IR per routed expert (43*256
                                                = 11008 IRs for real V4-Flash)

   See scripts/split_to_expert_irs.py for the toy version that built the
   same per-expert IR shape from a synthetic in-memory model — the layout here
   is intentionally identical so scripts/run_with_expert_offload.py can drive
   either source.

The dequant logic is verified on synthetic tensors in tests/test_dequant.py.
Entry point:

    python scripts/load_real_v4_weights.py --dry-run
    python scripts/load_real_v4_weights.py --per-expert-ir \\
        --weights-dir /path/to/DeepSeek-V4-Flash \\
        --output      /path/to/output_dir
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# FP4 / FP8 dequantization
# ---------------------------------------------------------------------------

# DeepSeek's FP4 (e2m1fn) lookup table. Index is the raw 4-bit value; the high
# bit is the sign. Reproduced from v4_flash_meta/inference/convert.py.
FP4_TABLE = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=torch.float32)


def dequant_fp4(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """FP4 (e2m1fn, 2 vals/byte) + per-32-col scale -> FP32.

    packed: int8 tensor [out_dim, in_packed]; in_dim = 2*in_packed.
            byte layout: low nibble is column 2k, high nibble is column 2k+1.
    scale:  numeric tensor [out_dim, in_dim // 32].
    """
    assert packed.dtype == torch.int8, f"expected int8, got {packed.dtype}"
    assert packed.dim() == 2
    out_dim, in_packed = packed.shape
    in_dim = 2 * in_packed
    fp4_block = 32
    assert in_dim % fp4_block == 0
    assert tuple(scale.shape) == (out_dim, in_dim // fp4_block), (
        f"scale shape {tuple(scale.shape)} != ({out_dim}, {in_dim // fp4_block})"
    )

    bytes_ = packed.view(torch.uint8)
    low = (bytes_ & 0x0F).long()
    high = ((bytes_ >> 4) & 0x0F).long()
    table = FP4_TABLE.to(packed.device)
    # convert.py order: low first, then high, in adjacent columns.
    vals = torch.stack([table[low], table[high]], dim=-1).reshape(out_dim, in_dim)

    s = scale.float().repeat_interleave(fp4_block, dim=-1)
    return vals * s


def dequant_fp8(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """FP8 (e4m3fn) + per-128x128-block scale -> FP32."""
    if weight.dtype == torch.int8:
        weight = weight.view(torch.float8_e4m3fn)
    assert weight.dtype == torch.float8_e4m3fn, f"expected fp8_e4m3fn, got {weight.dtype}"
    assert weight.dim() == 2
    out_dim, in_dim = weight.shape
    block = 128
    assert out_dim % block == 0 and in_dim % block == 0, (
        f"FP8 weight shape ({out_dim},{in_dim}) not divisible by 128"
    )
    assert tuple(scale.shape) == (out_dim // block, in_dim // block)

    w = weight.float().view(out_dim // block, block, in_dim // block, block)
    s = scale.float()[:, None, :, None]
    return (w * s).reshape(out_dim, in_dim)


# ---------------------------------------------------------------------------
# Real-V4 → ours name mapping
# ---------------------------------------------------------------------------

# Patterns for real-V4 keys we DELIBERATELY skip:
#   - mtp.*           : Multi-Token Prediction blocks (set num_nextn_predict_layers=0)
#   - *.tid2eid       : Hash-routing token-id -> expert-id tables (num_hash_layers=0)
#   - *.gate.bias     : Bias on routed-expert gate (toy uses `bias=False`)
SKIP_PATTERNS = (
    re.compile(r"^mtp\."),
    re.compile(r"\.tid2eid$"),
    re.compile(r"\.gate\.bias$"),
)


def map_real_to_ours(name: str) -> Optional[str]:
    """Map a real-V4 safetensors key to our DeepseekV4ForCausalLM parameter name.

    Returns None when the key is on the explicit skip list. Raises on unrecognized
    keys so that the dry-run can flag them.
    """
    for pat in SKIP_PATTERNS:
        if pat.search(name):
            return None

    if name == "embed.weight":
        return "model.embed.weight"
    if name == "head.weight":
        return "lm_head.weight"
    if name == "norm.weight":
        return "model.norm.weight"
    if name in ("hc_head_base", "hc_head_fn", "hc_head_scale"):
        return f"model.{name}"
    if name.startswith("layers."):
        return f"model.{name}"

    raise KeyError(f"Unrecognized real-V4 key: {name!r}")


def is_quant_scale(name: str) -> bool:
    """True if this name is a quantization scale (paired with a .weight).

    `hc_*_scale` are HC parameters, NOT quantization scales — they have no
    paired weight and stay as full-precision params.
    """
    return name.endswith(".scale")


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def dry_run(weights_dir: Path) -> Tuple[int, int, list]:
    """Read only the index.json and verify every key maps cleanly."""
    index_path = weights_dir / "model.safetensors.index.json"
    with open(index_path) as f:
        index = json.load(f)
    keys = list(index["weight_map"].keys())

    ok = 0
    skipped = 0
    paired_weights = 0  # `.weight` keys that have a paired `.scale`
    unmapped = []
    weight_keys = set(k for k in keys if not k.endswith(".scale"))
    scale_keys = set(k for k in keys if k.endswith(".scale"))

    for k in keys:
        try:
            mapped = map_real_to_ours(k)
        except KeyError:
            unmapped.append(k)
            continue
        if mapped is None:
            skipped += 1
        else:
            ok += 1

    for w in weight_keys:
        if w.replace(".weight", ".scale") in scale_keys:
            paired_weights += 1

    print(f"  total keys           : {len(keys)}")
    print(f"  mapped to our params : {ok}")
    print(f"  on skip list (mtp/hash/gate-bias): {skipped}")
    print(f"  weights paired w/ scale (FP4 or FP8): {paired_weights}")
    print(f"  free-floating .scale keys (HC, not quant): {len(scale_keys) - paired_weights}")
    if unmapped:
        print(f"\n  UNMAPPED ({len(unmapped)}):")
        for k in unmapped[:20]:
            print(f"    {k}")
        if len(unmapped) > 20:
            print(f"    ... and {len(unmapped) - 20} more")
    return ok, skipped, unmapped


def full_load(
    weights_dir: Path,
    output_dir: Path,
    target_dtype: torch.dtype = torch.bfloat16,
    skip_existing: bool = True,
) -> None:
    """Load real V4-Flash weights, dequant FP4/FP8, save BF16 safetensors.

    Streams shard by shard so peak RAM is bounded by the largest shard plus
    the largest single weight after dequant. Writes one BF16 .safetensors file
    per input shard to `output_dir`.
    """
    from safetensors.torch import safe_open, save_file

    index_path = weights_dir / "model.safetensors.index.json"
    with open(index_path) as f:
        index = json.load(f)
    weight_map: Dict[str, str] = index["weight_map"]
    shards: Dict[str, list] = defaultdict(list)
    for k, shard in weight_map.items():
        shards[shard].append(k)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"fp4": 0, "fp8": 0, "passthrough": 0, "skipped": 0}

    # Resolve which (.weight, .scale) pairs need joint loading. Most pairs are
    # in the same shard; rarely they straddle. Build a global scale->shard map.
    scale_to_shard = {k: v for k, v in weight_map.items() if k.endswith(".scale")}

    for shard_name, keys in sorted(shards.items()):
        out_path = output_dir / shard_name.replace(".safetensors", "-bf16.safetensors")
        if skip_existing and out_path.exists():
            print(f"  [skip] {out_path.name} (already exists)")
            continue
        print(f"  [load] {shard_name} ({len(keys)} tensors)")
        out: Dict[str, torch.Tensor] = {}
        with safe_open(weights_dir / shard_name, framework="pt", device="cpu") as f:
            keys_in_shard = set(f.keys())
            handled = set()
            for k in f.keys():
                if k in handled:
                    continue
                mapped = map_real_to_ours(k)
                if mapped is None:
                    summary["skipped"] += 1
                    continue
                if k.endswith(".scale"):
                    # Free-floating HC scale — not a quant scale.
                    out[mapped] = f.get_tensor(k).to(target_dtype)
                    summary["passthrough"] += 1
                    continue

                scale_name = k.replace(".weight", ".scale")
                if scale_name in scale_to_shard:
                    # Paired weight + scale: dequantize to BF16.
                    weight = f.get_tensor(k)
                    if scale_to_shard[scale_name] == shard_name:
                        scale = f.get_tensor(scale_name)
                        handled.add(scale_name)
                    else:
                        with safe_open(
                            weights_dir / scale_to_shard[scale_name],
                            framework="pt",
                            device="cpu",
                        ) as g:
                            scale = g.get_tensor(scale_name)

                    if weight.dtype == torch.int8 and weight.size(-1) * 2 // 32 == scale.size(-1):
                        deq = dequant_fp4(weight, scale)
                        summary["fp4"] += 1
                    else:
                        deq = dequant_fp8(weight, scale)
                        summary["fp8"] += 1
                    out[mapped] = deq.to(target_dtype)
                else:
                    # Plain BF16/FP32 tensor.
                    out[mapped] = f.get_tensor(k).to(target_dtype)
                    summary["passthrough"] += 1
        save_file(out, str(out_path))
        print(f"    -> wrote {out_path.name} ({len(out)} tensors)")
    print(f"\n  Summary: {summary}")


# ---------------------------------------------------------------------------
# Per-expert streaming IR conversion (Direction 2 / 2.3)
# ---------------------------------------------------------------------------

# A routed-expert key looks like: layers.{i}.ffn.experts.{e}.{w1|w2|w3}.weight
EXPERT_KEY = re.compile(
    r"^layers\.(?P<layer>\d+)\.ffn\.experts\.(?P<expert>\d+)\.(?P<proj>w1|w2|w3)\.weight$"
)


def _enumerate_experts(weight_map: Dict[str, str]):
    """Return {(layer, expert): {w1: shard, w2: shard, w3: shard, scales:[...]}}."""
    experts: Dict[Tuple[int, int], Dict[str, str]] = defaultdict(dict)
    for k, shard in weight_map.items():
        m = EXPERT_KEY.match(k)
        if not m:
            continue
        L = int(m["layer"])
        E = int(m["expert"])
        experts[(L, E)][m["proj"]] = shard
        scale = k.replace(".weight", ".scale")
        if scale in weight_map:
            experts[(L, E)].setdefault("_scale_shards", {})[m["proj"]] = weight_map[scale]
    return experts


def _load_expert_tensors(
    weights_dir: Path,
    layer: int,
    expert: int,
    shard_for: Dict[str, str],
    target_dtype: torch.dtype,
) -> Dict[str, torch.Tensor]:
    """Load w1/w2/w3 for one (layer, expert), dequantizing FP4 with paired scales.

    Returns {"w1": [inter, hidden], "w2": [hidden, inter], "w3": [inter, hidden]} in
    target_dtype. Each .safe_open context closes immediately after use so peak
    extra RAM is one expert's worth (~120 MB for real V4-Flash: 3 * 2048 * 4096
    * 2 bytes BF16 = ~50 MB; FP4 raw is half that on disk).
    """
    from safetensors.torch import safe_open

    out: Dict[str, torch.Tensor] = {}
    for proj in ("w1", "w2", "w3"):
        key = f"layers.{layer}.ffn.experts.{expert}.{proj}.weight"
        scale_key = key.replace(".weight", ".scale")
        weight_shard = shard_for[proj]
        with safe_open(weights_dir / weight_shard, framework="pt", device="cpu") as f:
            w = f.get_tensor(key)
            try:
                s = f.get_tensor(scale_key)
                same_shard = True
            except Exception:
                s = None
                same_shard = False
        if s is None:
            scale_shards = shard_for.get("_scale_shards", {})
            if proj in scale_shards:
                with safe_open(weights_dir / scale_shards[proj], framework="pt", device="cpu") as g:
                    s = g.get_tensor(scale_key)
        if s is None:
            # Plain BF16 weight (no microscale).
            out[proj] = w.to(target_dtype)
            continue
        if w.dtype == torch.int8 and w.size(-1) * 2 // 32 == s.size(-1):
            deq = dequant_fp4(w, s)
        else:
            deq = dequant_fp8(w, s)
        out[proj] = deq.to(target_dtype)
        del w, s, deq
    return out


def _convert_expert_to_ir(
    weights: Dict[str, torch.Tensor],
    swiglu_limit: float,
    hidden_size: int,
    inter_size: int,
    out_path: Path,
) -> None:
    """Build an Expert nn.Module, load the dequantized weights, trace to IR."""
    from src.modeling import Expert
    import openvino as ov

    expert = Expert(hidden_size, inter_size, swiglu_limit).eval()
    expert.w1.weight.data = weights["w1"].float()
    expert.w2.weight.data = weights["w2"].float()
    expert.w3.weight.data = weights["w3"].float()

    example = torch.zeros(1, hidden_size, dtype=torch.float32)
    ov_model = ov.convert_model(
        expert,
        example_input=(example,),
        input=[([-1, hidden_size], ov.Type.f32)],
    )
    ov_model.outputs[0].set_names({"expert_out"})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(ov_model, str(out_path), compress_to_fp16=False)


def full_load_per_expert_ir(
    weights_dir: Path,
    output_dir: Path,
    target_dtype: torch.dtype = torch.bfloat16,
    skip_existing: bool = True,
    max_experts: Optional[int] = None,
) -> None:
    """Stream V4-Flash → one IR per routed expert, plus a single backbone BF16
    safetensors. Peak RAM ≈ backbone (~5 GB) + one expert (~few hundred MB).

    `max_experts` caps the number of experts processed (useful for smoke testing
    on a partial download).
    """
    from safetensors.torch import safe_open, save_file

    index_path = weights_dir / "model.safetensors.index.json"
    with open(index_path) as f:
        index = json.load(f)
    weight_map: Dict[str, str] = index["weight_map"]

    # Read config.json so we know hidden_size / moe_intermediate_size / swiglu_limit.
    cfg_path = weights_dir / "config.json"
    with open(cfg_path) as f:
        real_cfg = json.load(f)
    hidden_size = real_cfg["hidden_size"]
    inter_size = real_cfg["moe_intermediate_size"]
    swiglu_limit = real_cfg.get("swiglu_limit", 10.0)

    output_dir.mkdir(parents=True, exist_ok=True)
    expert_dir = output_dir
    backbone_path = output_dir / "backbone.safetensors"

    # Enumerate experts and stream-convert each.
    experts = _enumerate_experts(weight_map)
    print(f"  {len(experts)} routed-expert (layer, idx) pairs to convert")
    print(f"  hidden_size={hidden_size}  moe_intermediate_size={inter_size}  swiglu_limit={swiglu_limit}")

    processed = 0
    for (layer, expert), shard_for in sorted(experts.items()):
        out_path = expert_dir / f"expert_L{layer}_E{expert}.xml"
        if skip_existing and out_path.exists():
            processed += 1
            continue
        weights = _load_expert_tensors(weights_dir, layer, expert, shard_for, target_dtype)
        _convert_expert_to_ir(weights, swiglu_limit, hidden_size, inter_size, out_path)
        del weights
        processed += 1
        if processed % 64 == 0:
            print(f"    [{processed}/{len(experts)}] L{layer} E{expert} -> {out_path.name}")
        if max_experts is not None and processed >= max_experts:
            print(f"    stopped at max_experts={max_experts}")
            break
    print(f"  total experts converted: {processed}")

    # Backbone: everything that is NOT an expert weight or expert scale. One pass
    # over shards, collect into a single dict, save once. This is the only step
    # whose peak is roughly the full backbone size (~5 GB) — well within 64 GB.
    if backbone_path.exists() and skip_existing:
        print(f"  [skip] backbone safetensors already at {backbone_path}")
        return

    print(f"  loading backbone (non-expert) weights...")
    shards: Dict[str, list] = defaultdict(list)
    for k, shard in weight_map.items():
        if EXPERT_KEY.match(k):
            continue
        if k.endswith(".scale") and EXPERT_KEY.match(k.replace(".scale", ".weight")):
            continue
        shards[shard].append(k)

    scale_to_shard = {k: v for k, v in weight_map.items() if k.endswith(".scale")}
    backbone: Dict[str, torch.Tensor] = {}
    summary = {"fp4": 0, "fp8": 0, "passthrough": 0, "skipped": 0}
    for shard_name, keys in sorted(shards.items()):
        print(f"    [load] {shard_name} ({len(keys)} backbone tensors)")
        with safe_open(weights_dir / shard_name, framework="pt", device="cpu") as f:
            handled = set()
            for k in keys:
                if k in handled or k.endswith(".scale"):
                    continue
                mapped = map_real_to_ours(k)
                if mapped is None:
                    summary["skipped"] += 1
                    continue
                scale_name = k.replace(".weight", ".scale")
                if scale_name in scale_to_shard:
                    weight = f.get_tensor(k)
                    if scale_to_shard[scale_name] == shard_name:
                        scale = f.get_tensor(scale_name)
                        handled.add(scale_name)
                    else:
                        with safe_open(weights_dir / scale_to_shard[scale_name], framework="pt", device="cpu") as g:
                            scale = g.get_tensor(scale_name)
                    if weight.dtype == torch.int8 and weight.size(-1) * 2 // 32 == scale.size(-1):
                        deq = dequant_fp4(weight, scale)
                        summary["fp4"] += 1
                    else:
                        deq = dequant_fp8(weight, scale)
                        summary["fp8"] += 1
                    backbone[mapped] = deq.to(target_dtype)
                    del weight, scale, deq
                else:
                    backbone[mapped] = f.get_tensor(k).to(target_dtype)
                    summary["passthrough"] += 1
    print(f"  backbone summary: {summary}, total tensors={len(backbone)}")
    save_file(backbone, str(backbone_path))
    print(f"  wrote {backbone_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-dir", type=Path, default=ROOT / "v4_flash_meta",
                        help="Directory containing model.safetensors.index.json (and shards for full mode).")
    parser.add_argument("--output", type=Path, default=ROOT / "v4_flash_bf16",
                        help="Output directory for dequantized BF16 shards or per-expert IRs.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Read only index.json and verify name coverage. No tensor loading.")
    parser.add_argument("--per-expert-ir", action="store_true",
                        help="Direction-2 streaming mode: emit one IR per routed expert + a single backbone safetensors. Peak RAM ~5-10 GB.")
    parser.add_argument("--max-experts", type=int, default=None,
                        help="(per-expert-ir mode) Stop after N experts. Useful for partial downloads / smoke testing.")
    args = parser.parse_args()

    if args.dry_run:
        print(f"=== Dry-run name-coverage check on {args.weights_dir} ===")
        ok, skipped, unmapped = dry_run(args.weights_dir)
        if unmapped:
            print("\nFAIL: unmapped keys remain")
            sys.exit(1)
        print("\nDRY-RUN: PASSED")
        return

    if args.per_expert_ir:
        print(f"=== Per-expert IR streaming: {args.weights_dir} -> {args.output} ===")
        print("Peak RAM ~5-10 GB. Output: one .xml/.bin per expert + backbone.safetensors.")
        full_load_per_expert_ir(args.weights_dir, args.output, max_experts=args.max_experts)
        print("\nPER-EXPERT-IR LOAD: PASSED")
        return

    print(f"=== Full load from {args.weights_dir} -> {args.output} ===")
    print("WARNING: real V4-Flash needs ~500GB peak RAM; on a 64GB host this WILL OOM.")
    full_load(args.weights_dir, args.output)


if __name__ == "__main__":
    main()
