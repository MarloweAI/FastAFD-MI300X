from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
from transformers import PretrainedConfig


@dataclass(frozen=True)
class RotaryConfig:
    head_dim: int
    rotary_dim: int
    max_position: int
    base: float
    scaling: Dict[str, Any] | None
    parameters: Dict[str, Any] | None = None
    partial_rotary_factor: float | None = None


@dataclass(frozen=True)
class QuantConfig:
    method: str
    activation_scheme: str
    weight_block_size: Tuple[int, int]


@dataclass(frozen=True)
class ModelConfig:
    num_layers: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    vocab_size: int
    intermediate_size: int
    rms_norm_eps: float
    rotary_config: RotaryConfig
    hidden_act: str
    tie_word_embeddings: bool
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    norm_topk_prob: bool
    use_routing_bias: bool
    scoring_func: str
    model_type: str
    architectures: list[str]
    quant: Optional[QuantConfig] = None
    use_qk_norm: bool = False
    # Per-layer attention type. gpt-oss alternates sliding_attention/full_attention and
    # gives a single `sliding_window`. A tuple, not a list, so the frozen dataclass stays
    # hashable. `attention/_build_layer_windows` reads these two off ModelConfig -- if
    # they are not carried here the sliding window silently becomes a no-op, because
    # getattr() just returns the default.
    layer_types: Tuple[str, ...] | None = None
    sliding_window: int = 0
    # q/k/v/o projection bias. Qwen3 has none; gpt-oss has all four.
    attention_bias: bool = False
    rope_parameters: Dict[str, Any] | None = None
    rope_scaling: Dict[str, Any] | None = None
    rope_theta: float = 10000.0
    partial_rotary_factor: float | None = None
    model_extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_moe(self) -> bool:
        return self.num_experts > 0

    @classmethod
    def from_hf(cls, config: PretrainedConfig) -> ModelConfig:
        if hasattr(config, "text_config") and config.text_config is not None:
            top = config
            config = config.text_config
            for attr in ("architectures", "rope_theta", "rope_scaling", "rope_parameters"):
                if not getattr(config, attr, None) and getattr(top, attr, None):
                    setattr(config, attr, getattr(top, attr))

        num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        tie_word_embeddings = getattr(config, "tie_word_embeddings", False)
        model_type = getattr(config, "model_type", "llama")
        n_routed_experts = getattr(config, "n_routed_experts", None)
        if n_routed_experts is None:
            num_experts = getattr(config, "num_local_experts", getattr(config, "num_experts", 0))
            n_routed_experts = num_experts
        else:
            num_experts = n_routed_experts
        num_experts_per_tok = getattr(config, "num_experts_per_tok", 0)
        moe_intermediate_size = getattr(config, "moe_intermediate_size", 0) or config.intermediate_size
        norm_topk_prob = getattr(config, "norm_topk_prob", False)
        use_routing_bias = getattr(config, "use_routing_bias", False)
        scoring_func = getattr(config, "scoring_func", "softmax")
        if model_type == "minimax_m2" and scoring_func == "sigmoid" and not hasattr(config, "norm_topk_prob"):
            # vLLM hardcodes MiniMax M2 sigmoid-routing weights to be
            # renormalized; the HF config currently omits norm_topk_prob.
            norm_topk_prob = True
        if model_type == "gpt_oss" or "GptOssForCausalLM" in getattr(
            config, "architectures", []
        ):
            # gpt-oss's router does topk(logits) -> softmax(top-k values), which is
            # algebraically identical to softmax-over-all -> topk -> renormalize (softmax
            # is monotonic so the selection matches, and renormalising a softmax over a
            # subset gives softmax over that subset's logits). The port's routing kernel
            # is the latter shape, so renormalize must be on -- but `norm_topk_prob` is
            # absent from gpt-oss's config, which would default it to False and silently
            # skip the renormalisation. Same fix shape as the MiniMax M2 case below.
            norm_topk_prob = True
        architectures = getattr(config, "architectures", ["LlamaForCausalLM"])
        use_qk_norm = getattr(config, "use_qk_norm", False)
        layer_types_raw = getattr(config, "layer_types", None)
        layer_types = tuple(layer_types_raw) if layer_types_raw else None
        sliding_window = int(getattr(config, "sliding_window", 0) or 0)
        attention_bias = bool(getattr(config, "attention_bias", False))
        model_extra: Dict[str, Any] = {}
        if layer_types is not None and sliding_window > 0:
            valid = {"sliding_attention", "full_attention"}
            unknown = sorted(set(layer_types) - valid)
            if unknown:
                # An unrecognised layer type would fall through to "full attention" and
                # silently widen the mask, so refuse rather than guess.
                raise ValueError(
                    f"unsupported layer_types {unknown}; expected a subset of {sorted(valid)}"
                )
        if model_type == "gpt_oss":
            # Clamp bound for the interleaved SwiGLU (layers/_triton_ops.gptoss_swiglu).
            model_extra["swiglu_limit"] = float(getattr(config, "swiglu_limit", 7.0))
        if model_type == "glm4_moe" or "Glm4MoeForCausalLM" in architectures:
            for attr, default in (
                ("n_shared_experts", 0),
                ("n_group", 0),
                ("topk_group", 0),
                ("first_k_dense_replace", 0),
                ("routed_scaling_factor", 1.0),
                ("attention_bias", False),
            ):
                value = getattr(config, attr, default)
                model_extra[attr] = default if value is None else value

        # Llama/Qwen: rope_theta is a direct attr; Mistral: it's inside rope_scaling dict
        # MiniMax M2 also uses partial rotary via rotary_dim.
        rope_scaling = getattr(config, "rope_scaling", None)
        rope_parameters = getattr(config, "rope_parameters", None)
        if rope_parameters is not None:
            rope_theta = rope_parameters.get("rope_theta", None) or getattr(config, "rope_theta", None)
            rope_scaling = rope_parameters
        else:
            rope_theta = getattr(config, "rope_theta", None)
            if rope_theta is None and rope_scaling is not None:
                rope_theta = rope_scaling["rope_theta"]
            if rope_theta is None:
                rope_theta = 10000.0
        partial_rotary_factor = getattr(config, "partial_rotary_factor", None)
        if partial_rotary_factor is None and isinstance(rope_parameters, dict):
            partial_rotary_factor = rope_parameters.get("partial_rotary_factor")
        if partial_rotary_factor is None and isinstance(rope_scaling, dict):
            partial_rotary_factor = rope_scaling.get("partial_rotary_factor")
        rotary_dim = getattr(config, "rotary_dim", None)
        if rotary_dim is None:
            rotary_dim = (
                int(head_dim * float(partial_rotary_factor))
                if partial_rotary_factor is not None
                else head_dim
            )

        quant: Optional[QuantConfig] = None
        quant_config = getattr(config, "quantization_config", None)
        if quant_config is not None:
            quant_dict = (
                quant_config
                if isinstance(quant_config, dict)
                else getattr(quant_config, "to_dict", lambda: {})()
            )
            quant_method = quant_dict.get("quant_method", "") or ""
            if "mxfp4" in quant_method:
                # gpt-oss. MXFP4 has no native datapath on gfx942/CDNA3 (that is CDNA 4 /
                # gfx950), so the weights are dequantised to bf16 at load and every GEMM
                # runs in bf16. `method="mxfp4"` therefore means "unpack at load", not
                # "compute in fp4" -- see kernel/mxfp4.py and
                # dev_log/gpt_oss_120b/00_README.md.
                quant = QuantConfig(
                    method="mxfp4",
                    activation_scheme="none",
                    weight_block_size=(1, 32),  # 32 values share one E8M0 scale
                )
            elif "fp8" in quant_method:
                block_size = quant_dict.get("weight_block_size", [128, 128])
                quant = QuantConfig(
                    method="fp8",
                    activation_scheme=quant_dict.get("activation_scheme", "dynamic"),
                    weight_block_size=(int(block_size[0]), int(block_size[1])),
                )
            elif (
                quant_method == "compressed-tensors"
                and quant_dict.get("format") == "float-quantized"
            ):
                # GLM FP8 checkpoints store per-output-channel FP8 weights plus
                # `.weight_scale`.  Runtime code repacks these channel-scale
                # tensors into DeepGEMM's 128x128 block-scale FP8 format once
                # after loading.
                quant = QuantConfig(
                    method="fp8_channel",
                    activation_scheme="dynamic",
                    weight_block_size=(1, 1),
                )

        return cls(
            num_layers=config.num_hidden_layers,
            num_qo_heads=config.num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            rms_norm_eps=config.rms_norm_eps,
            tie_word_embeddings=tie_word_embeddings,
            rotary_config=RotaryConfig(
                head_dim=head_dim,
                rotary_dim=rotary_dim,
                max_position=config.max_position_embeddings,
                base=rope_theta,
                scaling=rope_scaling,
                parameters=rope_parameters,
                partial_rotary_factor=partial_rotary_factor,
            ),
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            moe_intermediate_size=moe_intermediate_size,
            norm_topk_prob=norm_topk_prob,
            use_routing_bias=use_routing_bias,
            scoring_func=scoring_func,
            model_type=model_type,
            architectures=architectures,
            quant=quant,
            use_qk_norm=use_qk_norm,
            layer_types=layer_types,
            sliding_window=sliding_window,
            attention_bias=attention_bias,
            rope_parameters=rope_parameters,
            rope_scaling=rope_scaling,
            rope_theta=rope_theta,
            partial_rotary_factor=partial_rotary_factor,
            model_extra=model_extra,
        )
