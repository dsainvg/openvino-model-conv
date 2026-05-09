"""Smoke test: build toy DeepSeek-V4, run a forward, check shape and finiteness."""
import sys
from pathlib import Path

# Add src/ to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from deepseek_v4 import DeepseekV4Config, DeepseekV4ForCausalLM


def make_toy_config() -> DeepseekV4Config:
    """A small V4 config that exercises every architectural feature.
    4 layers with compress_ratios = [0, 0, 4, 128] so we get:
      layer 0: sliding-window only
      layer 1: sliding-window only
      layer 2: window + indexer-driven sparse compression (ratio 4)
      layer 3: window + dense compression (ratio 128)
    """
    return DeepseekV4Config(
        vocab_size=512,
        hidden_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=32,
        qk_rope_head_dim=8,         # 2*8=16 dims rotated; fits inside head_dim=32
        q_lora_rank=64,
        o_lora_rank=64,
        o_groups=2,
        sliding_window=8,
        compress_ratios=[0, 0, 4, 128],
        index_n_heads=4,
        index_head_dim=16,
        index_topk=8,
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        scoring_func="sqrtsoftplus",
        topk_method="noaux_tc",
        norm_topk_prob=True,
        routed_scaling_factor=1.5,
        num_hash_layers=0,
        hc_mult=4,
        hc_sinkhorn_iters=4,        # fewer iters for toy speed
        hc_eps=1e-6,
        hidden_act="silu",
        swiglu_limit=10.0,
        rms_norm_eps=1e-6,
        max_position_embeddings=512,
        rope_theta=10000.0,
        compress_rope_theta=160000.0,
        rope_scaling={
            "type": "yarn",
            "factor": 1.0,           # no scaling for toy
            "original_max_position_embeddings": 512,
            "beta_fast": 32,
            "beta_slow": 1,
        },
        num_nextn_predict_layers=0,
        torch_dtype="float32",
    )


def main():
    torch.manual_seed(0)
    cfg = make_toy_config()
    print("Config:", cfg)
    print()

    model = DeepseekV4ForCausalLM(cfg).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {n_params:,}")

    # We need seq_len >= 128 to test the compress_ratio=128 layer's compressor outputs >0 chunks.
    # And >=4 for the ratio=4 indexer.
    B, S = 1, 128
    input_ids = torch.randint(0, cfg.vocab_size, (B, S))
    print(f"Running forward on input_ids shape={tuple(input_ids.shape)}")

    with torch.inference_mode():
        out = model(input_ids=input_ids)

    logits = out.logits
    print(f"Logits shape: {tuple(logits.shape)}")
    print(f"Logits dtype: {logits.dtype}")
    print(f"Logits finite: {torch.isfinite(logits).all().item()}")
    print(f"Logits min/mean/max: {logits.min().item():.4f} / {logits.mean().item():.4f} / {logits.max().item():.4f}")

    assert logits.shape == (B, S, cfg.vocab_size), f"unexpected logits shape {logits.shape}"
    assert torch.isfinite(logits).all(), "logits contain NaN/inf"
    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
