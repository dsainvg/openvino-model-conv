from transformers import AutoConfig

cfg = AutoConfig.from_pretrained(r"C:\Users\intel\models\qwen36-35b-int4")
print("model_type:", cfg.model_type)
print("architectures:", cfg.architectures)
print("text num_layers:", cfg.text_config.num_hidden_layers)
print("text num_experts:", cfg.text_config.num_experts)
print("text num_experts_per_tok:", cfg.text_config.num_experts_per_tok)
print("hidden_size:", cfg.text_config.hidden_size)
print("vocab_size:", cfg.text_config.vocab_size)
qc = cfg.quantization_config if hasattr(cfg, "quantization_config") else None
if qc is not None:
    print("quant method:", qc.get("quant_method") if isinstance(qc, dict) else getattr(qc, "quant_method", None))
print("AutoConfig OK")
