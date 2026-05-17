from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Tuple

import torch
import torch.nn as nn


DEFAULT_CKPT_KEYS = (
    "model",
    "state_dict",
    "model_state_dict",
    "teacher",
    "student",
    "backbone",
    "network",
)


DEFAULT_PREFIXES = (
    "module.",
    "model.",
    "backbone.",
    "student.",
    "teacher.",
    "_orig_mod.",
    "fc.",
    "classifier.",
    "head.",
)


def _is_tensor_like(v: Any) -> bool:
    return isinstance(v, (torch.Tensor, nn.Parameter))


def _is_state_dict_like(obj: Any) -> bool:
    return isinstance(obj, dict) and len(obj) > 0 and all(_is_tensor_like(v) for v in obj.values())


def extract_state_dict_from_checkpoint(
    checkpoint: Any,
    preferred_keys: Iterable[str] = DEFAULT_CKPT_KEYS,
) -> Tuple[Dict[str, torch.Tensor], str]:
    if _is_state_dict_like(checkpoint):
        return checkpoint, "<root>"

    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint is not a dict and cannot be interpreted as a state_dict.")

    for key in preferred_keys:
        if key in checkpoint and _is_state_dict_like(checkpoint[key]):
            return checkpoint[key], key

    for key, value in checkpoint.items():
        if _is_state_dict_like(value):
            return value, key

    raise ValueError("No state_dict-like entry found in checkpoint.")


def _strip_prefix(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    if not prefix:
        return state_dict
    return OrderedDict(
        (k[len(prefix) :], v) if k.startswith(prefix) else (k, v)
        for k, v in state_dict.items()
    )


def _add_prefix(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    if not prefix:
        return state_dict
    return OrderedDict((prefix + k, v) for k, v in state_dict.items())


def _strip_all_known_prefixes(
    state_dict: Dict[str, torch.Tensor],
    prefixes: Iterable[str],
) -> Dict[str, torch.Tensor]:
    prefixes = tuple(prefixes)
    out = OrderedDict()
    for key, value in state_dict.items():
        k = key
        changed = True
        while changed:
            changed = False
            for p in prefixes:
                if p and k.startswith(p):
                    k = k[len(p) :]
                    changed = True
        out[k] = value
    return out


def _build_candidate_state_dicts(
    state_dict: Dict[str, torch.Tensor],
    prefixes: Iterable[str],
) -> List[Tuple[str, Dict[str, torch.Tensor]]]:
    prefixes = tuple(prefixes)
    candidates: List[Tuple[str, Dict[str, torch.Tensor]]] = []

    base_variants: List[Tuple[str, Dict[str, torch.Tensor]]] = [("identity", state_dict)]
    for p in prefixes:
        base_variants.append((f"strip:{p}", _strip_prefix(state_dict, p)))
    base_variants.append(("strip_all_known", _strip_all_known_prefixes(state_dict, prefixes)))

    for base_name, base_sd in base_variants:
        candidates.append((base_name, base_sd))
        for p in prefixes:
            candidates.append((f"{base_name}|add:{p}", _add_prefix(base_sd, p)))

    return candidates


def adapt_state_dict_keys_for_model(
    module: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    prefixes: Iterable[str] = DEFAULT_PREFIXES,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    target_keys = set(module.state_dict().keys())
    candidates = _build_candidate_state_dicts(state_dict, prefixes=prefixes)

    best_sd = state_dict
    best_meta: Dict[str, Any] = {
        "transform": "identity",
        "matched_keys": 0,
        "missing_keys": len(target_keys),
        "unexpected_keys": len(state_dict),
        "coverage": 0.0,
    }

    for transform_name, cand in candidates:
        cand_keys = set(cand.keys())
        matched = len(target_keys & cand_keys)
        if matched == 0:
            continue

        missing = len(target_keys - cand_keys)
        unexpected = len(cand_keys - target_keys)
        coverage = matched / max(1, len(target_keys))

        score = (matched, -missing, -unexpected)
        best_score = (
            best_meta["matched_keys"],
            -best_meta["missing_keys"],
            -best_meta["unexpected_keys"],
        )
        if score > best_score:
            best_sd = cand
            best_meta = {
                "transform": transform_name,
                "matched_keys": matched,
                "missing_keys": missing,
                "unexpected_keys": unexpected,
                "coverage": coverage,
            }

    return best_sd, best_meta


def load_state_dict_with_prefix_compat(
    module: nn.Module,
    checkpoint: Any,
    strict: bool = True,
    prefixes: Iterable[str] = DEFAULT_PREFIXES,
) -> Dict[str, Any]:
    state_dict, source_key = extract_state_dict_from_checkpoint(checkpoint)
    adapted, meta = adapt_state_dict_keys_for_model(module, state_dict, prefixes=prefixes)
    load_msg = module.load_state_dict(adapted, strict=strict)
    return {
        "source_key": source_key,
        "strict": strict,
        "transform": meta["transform"],
        "matched_keys": meta["matched_keys"],
        "coverage": meta["coverage"],
        "missing_keys": list(load_msg.missing_keys),
        "unexpected_keys": list(load_msg.unexpected_keys),
    }


def load_partial_state_dict_with_prefix_compat(
    module: nn.Module,
    checkpoint: Any,
    prefixes: Iterable[str] = DEFAULT_PREFIXES,
) -> Dict[str, Any]:
    state_dict, source_key = extract_state_dict_from_checkpoint(checkpoint)
    adapted, meta = adapt_state_dict_keys_for_model(module, state_dict, prefixes=prefixes)

    module_state = module.state_dict()
    filtered = OrderedDict()
    skipped_shape = []
    for key, tensor in adapted.items():
        if key not in module_state:
            continue
        if tuple(module_state[key].shape) != tuple(tensor.shape):
            skipped_shape.append(key)
            continue
        filtered[key] = tensor

    load_msg = module.load_state_dict(filtered, strict=False)
    return {
        "source_key": source_key,
        "strict": False,
        "transform": meta["transform"],
        "matched_keys": meta["matched_keys"],
        "coverage": meta["coverage"],
        "loaded_keys": len(filtered),
        "skipped_shape_keys": skipped_shape,
        "missing_keys": list(load_msg.missing_keys),
        "unexpected_keys": list(load_msg.unexpected_keys),
    }
