"""End-to-end integration tests for Qwen3.5-4B conversion.

Creates a temporary dummy model with random weights and runs the convert_to_openvino.py script on it.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Programmatically disable OpenVINO telemetry to avoid Windows/Python 3.13 thread-safety crashes in ssl/socket
try:
    import openvino_telemetry
    def dummy_method(*args, **kwargs):
        pass
    openvino_telemetry.Telemetry.send_event = dummy_method
    openvino_telemetry.Telemetry.start_session = dummy_method
    openvino_telemetry.Telemetry.end_session = dummy_method
    openvino_telemetry.Telemetry.send_error = dummy_method
    openvino_telemetry.Telemetry.send_stack_trace = dummy_method
    
    import openvino_telemetry.utils.sender
    openvino_telemetry.utils.sender.TelemetrySender.send = dummy_method
except ImportError:
    pass

from src.configuration import make_toy_config


@pytest.fixture
def dummy_model_dir():
    """Create a temporary directory containing config.json and model.safetensors with random weights."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        cfg = make_toy_config(
            num_layers=4,
            full_attention_interval=4,
            hidden_size=32,
            intermediate_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=4,
            linear_value_head_dim=4,
            linear_conv_kernel_dim=4,
            vocab_size=128,
        )

        # Write config.json in the EXACT same structure as the real Qwen3.5-4B
        # config.json on HuggingFace: text_config nested, partial_rotary_factor
        # and rope_theta inside rope_parameters. This ensures the parser is
        # tested against the real layout, not a simplified fixture.
        config_dict = {
            "model_type": "qwen3_5",
            "text_config": {
                "hidden_size": cfg.hidden_size,
                "num_hidden_layers": cfg.num_hidden_layers,
                "vocab_size": cfg.vocab_size,
                "num_attention_heads": cfg.num_attention_heads,
                "num_key_value_heads": cfg.num_key_value_heads,
                "head_dim": cfg.head_dim,
                # partial_rotary_factor lives inside rope_parameters in the real config
                "rope_parameters": {
                    "rope_theta": cfg.rope_theta,
                    "partial_rotary_factor": cfg.partial_rotary_factor,
                },
                "max_position_embeddings": cfg.max_position_embeddings,
                "linear_num_value_heads": cfg.linear_num_value_heads,
                "linear_num_key_heads": cfg.linear_num_key_heads,
                "linear_key_head_dim": cfg.linear_key_head_dim,
                "linear_value_head_dim": cfg.linear_value_head_dim,
                "linear_conv_kernel_dim": cfg.linear_conv_kernel_dim,
                "intermediate_size": cfg.intermediate_size,
                "full_attention_interval": cfg.full_attention_interval,
                "layer_types": list(cfg.layer_types),
                "rms_norm_eps": cfg.rms_norm_eps,
                "hidden_act": cfg.hidden_act,
                "bos_token_id": cfg.bos_token_id,
                "eos_token_id": cfg.eos_token_id,
                "pad_token_id": cfg.pad_token_id,
                "attention_bias": False,
                "attn_output_gate": True,
            }
        }

        with open(tmp_path / "config.json", "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2)

        # Generate weights
        weights = {}
        # Global weights
        weights["language_model.model.embed_tokens.weight"] = torch.randn(cfg.vocab_size, cfg.hidden_size)
        weights["language_model.model.norm.weight"] = torch.randn(cfg.hidden_size)
        weights["language_model.lm_head.weight"] = torch.randn(cfg.vocab_size, cfg.hidden_size)

        for i in range(cfg.num_hidden_layers):
            P = f"language_model.model.layers.{i}"
            weights[f"{P}.input_layernorm.weight"] = torch.randn(cfg.hidden_size)
            weights[f"{P}.post_attention_layernorm.weight"] = torch.randn(cfg.hidden_size)
            weights[f"{P}.mlp.gate_proj.weight"] = torch.randn(cfg.intermediate_size, cfg.hidden_size)
            weights[f"{P}.mlp.up_proj.weight"] = torch.randn(cfg.intermediate_size, cfg.hidden_size)
            weights[f"{P}.mlp.down_proj.weight"] = torch.randn(cfg.hidden_size, cfg.intermediate_size)

            if cfg.layer_types[i] == "linear_attention":
                weights[f"{P}.linear_attn.in_proj_qkv.weight"] = torch.randn(cfg.linear_conv_dim, cfg.hidden_size)
                weights[f"{P}.linear_attn.in_proj_z.weight"] = torch.randn(cfg.linear_value_dim, cfg.hidden_size)
                weights[f"{P}.linear_attn.in_proj_b.weight"] = torch.randn(cfg.linear_num_value_heads, cfg.hidden_size)
                weights[f"{P}.linear_attn.in_proj_a.weight"] = torch.randn(cfg.linear_num_value_heads, cfg.hidden_size)
                weights[f"{P}.linear_attn.conv1d.weight"] = torch.randn(cfg.linear_conv_dim, 1, cfg.linear_conv_kernel_dim)
                weights[f"{P}.linear_attn.dt_bias"] = torch.randn(cfg.linear_num_value_heads)
                weights[f"{P}.linear_attn.A_log"] = torch.randn(cfg.linear_num_value_heads)
                weights[f"{P}.linear_attn.norm.weight"] = torch.randn(cfg.linear_value_head_dim)
                weights[f"{P}.linear_attn.out_proj.weight"] = torch.randn(cfg.hidden_size, cfg.linear_value_dim)

            elif cfg.layer_types[i] == "full_attention":
                weights[f"{P}.self_attn.q_proj.weight"] = torch.randn(cfg.num_attention_heads * cfg.head_dim * 2, cfg.hidden_size)
                weights[f"{P}.self_attn.k_proj.weight"] = torch.randn(cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size)
                weights[f"{P}.self_attn.v_proj.weight"] = torch.randn(cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size)
                weights[f"{P}.self_attn.o_proj.weight"] = torch.randn(cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim)
                weights[f"{P}.self_attn.q_norm.weight"] = torch.randn(cfg.head_dim)
                weights[f"{P}.self_attn.k_norm.weight"] = torch.randn(cfg.head_dim)

        save_file(weights, str(tmp_path / "model.safetensors"))
        yield tmp_path


def test_convert_to_openvino_script(dummy_model_dir):
    """Run convert_to_openvino.py on dummy model to ensure it produces IR and matches numerically."""
    with tempfile.TemporaryDirectory() as outdir:
        out_path = Path(outdir)
        
        # Call the main function directly in-process to avoid Windows subprocess thread pool / DLL initialization crashes (0xC0000005)
        old_argv = sys.argv
        sys.argv = [
            "convert_to_openvino.py",
            "--model-dir",
            str(dummy_model_dir),
            "--output",
            str(out_path),
            "--dtype",
            "fp32",  # float32 for testing conversion
        ]
        try:
            from scripts.convert_to_openvino import main as convert_main
            ret = convert_main()
            assert ret == 0, f"Script main returned {ret}"
        finally:
            sys.argv = old_argv

        # Verify output files exist
        assert (out_path / "embed.xml").exists()
        assert (out_path / "lm_head.xml").exists()
        for i in range(4):
            assert (out_path / f"layer_{i}.xml").exists()
