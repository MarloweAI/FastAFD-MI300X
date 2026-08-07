import functools
import os
from typing import Any

from huggingface_hub import snapshot_download
from transformers import AutoConfig, PretrainedConfig

from minisgl.hf_support import DisabledTqdm, load_tokenizer, local_files_only, resolve_local_model_dir


@functools.cache
def _load_hf_config_cached(resolved_model_path: str, local_files_only: bool) -> Any:
    return AutoConfig.from_pretrained(
        resolved_model_path,
        local_files_only=local_files_only,
        trust_remote_code=True,
    )


def _load_hf_config(model_path: str) -> Any:
    local_files = local_files_only()
    resolved_model_path = resolve_local_model_dir(model_path, local_only=local_files)
    return _load_hf_config_cached(resolved_model_path, local_files)


def cached_load_hf_config(model_path: str) -> PretrainedConfig:
    config = _load_hf_config(model_path)
    return type(config)(**config.to_dict())


def download_hf_weight(model_path: str) -> str:
    if os.path.isdir(model_path):
        return model_path
    try:
        return snapshot_download(
            model_path,
            allow_patterns=["*.safetensors"],
            tqdm_class=DisabledTqdm,
        )
    except Exception as e:
        raise ValueError(
            f"Model path '{model_path}' is neither a local directory nor a valid model ID: {e}"
        )
