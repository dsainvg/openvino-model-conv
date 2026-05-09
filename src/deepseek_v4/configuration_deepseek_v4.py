"""DeepSeek-V4 configuration. Mirrors fields used by deepseek-ai/DeepSeek-V4-Flash."""
from transformers import PretrainedConfig


class DeepseekV4Config(PretrainedConfig):
    model_type = "deepseek_v4"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 129280,
        hidden_size: int = 4096,
        num_hidden_layers: int = 43,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 1,
        head_dim: int = 512,
        qk_rope_head_dim: int = 64,
        q_lora_rank: int = 1024,
        o_lora_rank: int = 1024,
        o_groups: int = 8,
        sliding_window: int = 128,
        compress_ratios=None,
        # Indexer
        index_n_heads: int = 64,
        index_head_dim: int = 128,
        index_topk: int = 512,
        # MoE
        n_routed_experts: int = 256,
        n_shared_experts: int = 1,
        num_experts_per_tok: int = 6,
        moe_intermediate_size: int = 2048,
        scoring_func: str = "sqrtsoftplus",
        topk_method: str = "noaux_tc",
        norm_topk_prob: bool = True,
        routed_scaling_factor: float = 1.5,
        num_hash_layers: int = 3,
        # Hyper-Connections
        hc_mult: int = 4,
        hc_sinkhorn_iters: int = 20,
        hc_eps: float = 1e-6,
        # Misc
        hidden_act: str = "silu",
        swiglu_limit: float = 10.0,
        rms_norm_eps: float = 1e-6,
        max_position_embeddings: int = 1_048_576,
        # RoPE
        rope_theta: float = 10000.0,
        compress_rope_theta: float = 160000.0,
        rope_scaling=None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        # MTP (multi-token prediction)
        num_nextn_predict_layers: int = 1,
        # Misc
        initializer_range: float = 0.02,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        bos_token_id: int = 0,
        eos_token_id: int = 1,
        pad_token_id: int = None,
        torch_dtype: str = "bfloat16",
        **kwargs,
    ):
        # YaRN defaults from the V4-Flash config
        if rope_scaling is None:
            rope_scaling = {
                "type": "yarn",
                "factor": 16.0,
                "original_max_position_embeddings": 65536,
                "beta_fast": 32,
                "beta_slow": 1,
            }
        if compress_ratios is None:
            # 43 layers in V4-Flash: first two ratios are 0 (sliding-window only),
            # then alternating 4/128, last is 0. Toy default: all zeros (handled by user).
            compress_ratios = [0] * num_hidden_layers

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.q_lora_rank = q_lora_rank
        self.o_lora_rank = o_lora_rank
        self.o_groups = o_groups
        self.sliding_window = sliding_window
        self.compress_ratios = list(compress_ratios)

        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk

        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size
        self.scoring_func = scoring_func
        self.topk_method = topk_method
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.num_hash_layers = num_hash_layers

        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.hc_eps = hc_eps

        self.hidden_act = hidden_act
        self.swiglu_limit = swiglu_limit
        self.rms_norm_eps = rms_norm_eps
        self.max_position_embeddings = max_position_embeddings

        self.rope_theta = rope_theta
        self.compress_rope_theta = compress_rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        self.num_nextn_predict_layers = num_nextn_predict_layers

        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.torch_dtype = torch_dtype

        if len(self.compress_ratios) != num_hidden_layers:
            raise ValueError(
                f"compress_ratios length {len(self.compress_ratios)} != num_hidden_layers {num_hidden_layers}"
            )

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            **kwargs,
        )
