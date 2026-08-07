from __future__ import annotations

from typing import Callable

_PRE_PERMUTE_REGISTRY: dict[tuple[str, str], Callable] = {}
_POST_PERMUTE_REGISTRY: dict[tuple[str, str], Callable] = {}


def register_pre_permute(src: str, dst: str):
    def decorator(fn: Callable) -> Callable:
        if (src, dst) in _PRE_PERMUTE_REGISTRY:
            raise RuntimeError(f"pre-permute already registered for ({src!r}, {dst!r})")
        _PRE_PERMUTE_REGISTRY[(src, dst)] = fn
        return fn

    return decorator


def register_post_permute(src: str, dst: str):
    def decorator(fn: Callable) -> Callable:
        if (src, dst) in _POST_PERMUTE_REGISTRY:
            raise RuntimeError(f"post-permute already registered for ({src!r}, {dst!r})")
        _POST_PERMUTE_REGISTRY[(src, dst)] = fn
        return fn

    return decorator


def get_pre_permute(src: str, dst: str) -> Callable:
    if (src, dst) not in _PRE_PERMUTE_REGISTRY:
        raise RuntimeError(
            f"no pre-permute registered for ({src!r}, {dst!r}); "
            f"available: {sorted(_PRE_PERMUTE_REGISTRY.keys())}"
        )
    return _PRE_PERMUTE_REGISTRY[(src, dst)]


def get_post_permute(src: str, dst: str) -> Callable:
    if (src, dst) not in _POST_PERMUTE_REGISTRY:
        raise RuntimeError(
            f"no post-permute registered for ({src!r}, {dst!r}); "
            f"available: {sorted(_POST_PERMUTE_REGISTRY.keys())}"
        )
    return _POST_PERMUTE_REGISTRY[(src, dst)]


__all__ = [
    "register_pre_permute",
    "register_post_permute",
    "get_pre_permute",
    "get_post_permute",
]
