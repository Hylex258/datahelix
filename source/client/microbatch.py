# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Client-side helpers for splitting one logical batch into microbatch requests."""

from __future__ import annotations

import importlib.util
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class MicroBatchIndexResponseGroup:
    """Index responses that together form one client-visible batch."""

    responses: tuple[Any, ...]
    chunk_sizes: tuple[int, ...]


def split_microbatch_indices(index: Any, microbatch_size: int | None, max_parts: int | None = None) -> list[Any]:
    """Split a list/tuple batch index into bounded microbatch index chunks."""

    if not isinstance(index, (list, tuple)):
        return [index]
    indices = list(index)
    if not indices:
        return [indices]
    if microbatch_size is None or microbatch_size <= 0 or len(indices) <= microbatch_size:
        return [indices]

    chunk_size = int(microbatch_size)
    if max_parts is not None and max_parts > 0:
        chunk_count = math.ceil(len(indices) / chunk_size)
        if chunk_count > max_parts:
            chunk_size = math.ceil(len(indices) / max_parts)

    return [indices[start : start + chunk_size] for start in range(0, len(indices), chunk_size)]


def merge_microbatch_results(results: list[Any], merge_fn: Callable[[list[Any]], Any] | None = None) -> Any:
    """Merge ordered microbatch payloads back into one logical batch."""

    if merge_fn is not None:
        return merge_fn(results)
    if len(results) == 1:
        return results[0]
    return _merge_values(results)


def _merge_values(values: list[Any]) -> Any:
    first = values[0]
    if _is_torch_tensor(first):
        torch = _require_torch()
        return torch.stack(values, dim=0) if first.dim() == 0 else torch.cat(values, dim=0)

    if isinstance(first, np.ndarray):
        return np.stack(values, axis=0) if first.ndim == 0 else np.concatenate(values, axis=0)

    if isinstance(first, Mapping):
        return type(first)((key, _merge_values([value[key] for value in values])) for key in first)

    if isinstance(first, tuple) and hasattr(first, "_fields"):
        return type(first)(*(_merge_values([value[i] for value in values]) for i in range(len(first))))

    if isinstance(first, tuple):
        return tuple(_merge_values([value[i] for value in values]) for i in range(len(first)))

    if isinstance(first, list):
        merged = []
        for value in values:
            merged.extend(value)
        return merged

    return list(values)


def _require_torch():
    import torch

    return torch


def _is_torch_tensor(value: Any) -> bool:
    if importlib.util.find_spec("torch") is None:
        return False

    import torch

    return isinstance(value, torch.Tensor)
