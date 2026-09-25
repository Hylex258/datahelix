# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
"""Runtime reader for normalized CPU scheduler cost profiles."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from distributed_dataloader.utils.logger import logger

DEFAULT_CPU_COST_S = 0.01


@dataclass(frozen=True)
class CostProfile:
    """Cost estimate for one profiled sample."""

    cpu_cost_s: float = DEFAULT_CPU_COST_S
    read_bytes: int = 0
    output_bytes: int = 0
    io_locality_hint: str = "unknown"


@dataclass(frozen=True)
class RequestCostProfile:
    """Aggregated cost estimate for one IndexRequest."""

    cpu_cost_s: float = DEFAULT_CPU_COST_S
    read_bytes: int = 0
    output_bytes: int = 0
    io_locality_hint: str = "unknown"
    sample_count: int = 1


class CostProfileStore:
    """Read normalized JSONL profiles produced by ``profiling/cpu_scheduler``."""

    def __init__(self, path: str | None = None):
        self._by_index: dict[int, CostProfile] = {}
        self._global_default = CostProfile()
        if path:
            self.load(path)

    def load(self, path: str) -> None:
        profile_path = Path(path)
        if not profile_path.exists():
            raise FileNotFoundError(f"Cost profile table does not exist: {path}")

        loaded = 0
        for row in self._read_jsonl(profile_path):
            profile = self._profile_from_row(row)
            index_value = row.get("index")
            if index_value is not None:
                self._by_index[int(index_value)] = profile
                loaded += 1
                continue
            if row.get("profile_key") in {"global:default", "default"}:
                self._global_default = profile
                loaded += 1

        logger.info(
            "Loaded CPU scheduler cost profile table: "
            f"path={path} exact={len(self._by_index)} loaded={loaded}"
        )

    def estimate_request(
        self,
        indices: Iterable[int],
    ) -> RequestCostProfile:
        profiles = [self._lookup_one(int(index)) for index in indices]
        if not profiles:
            profiles = [self._global_default]

        locality = self._merge_locality(profile.io_locality_hint for profile in profiles)
        return RequestCostProfile(
            cpu_cost_s=sum(profile.cpu_cost_s for profile in profiles),
            read_bytes=sum(profile.read_bytes for profile in profiles),
            output_bytes=sum(profile.output_bytes for profile in profiles),
            io_locality_hint=locality,
            sample_count=max(len(profiles), 1),
        )

    def _lookup_one(
        self,
        index: int,
    ) -> CostProfile:
        if index in self._by_index:
            return self._by_index[index]
        return self._global_default

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict]:
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL cost profile at {path}:{line_no}: {exc}") from exc
        return rows

    @staticmethod
    def _profile_from_row(row: dict) -> CostProfile:
        return CostProfile(
            cpu_cost_s=max(float(row.get("cpu_cost_s", DEFAULT_CPU_COST_S)), 0.0),
            read_bytes=max(int(row.get("read_bytes", 0)), 0),
            output_bytes=max(int(row.get("output_bytes", 0)), 0),
            io_locality_hint=str(row.get("io_locality_hint") or "unknown"),
        )

    @staticmethod
    def _merge_locality(localities: Iterable[str]) -> str:
        priority = {
            "local_shard": 0,
            "same_host_shared": 1,
            "remote_fs": 2,
            "object_store": 3,
            "unknown": 4,
        }
        selected = "local_shard"
        selected_priority = -1
        for locality in localities:
            current = locality or "unknown"
            current_priority = priority.get(current, priority["unknown"])
            if current_priority > selected_priority:
                selected = current
                selected_priority = current_priority
        return selected
