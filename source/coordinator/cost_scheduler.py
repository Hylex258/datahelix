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
"""Cost-aware DataServer selection for the Coordinator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from distributed_dataloader.coordinator.cost_profile import CostProfileStore, RequestCostProfile

SCHEDULER_LEAST_REQUEST = "least_request"
SCHEDULER_COST_AWARE = "cost_aware"
VALID_SCHEDULERS = {SCHEDULER_LEAST_REQUEST, SCHEDULER_COST_AWARE}

DEFAULT_REMOTE_MIN_GAIN = 0.0
DEFAULT_TCP_BANDWIDTH = 256 * 1024 * 1024
DEFAULT_FAST_BANDWIDTH = 2 * 1024 * 1024 * 1024
DEFAULT_SHM_BANDWIDTH = 8 * 1024 * 1024 * 1024
DEFAULT_UNKNOWN_BANDWIDTH = 128 * 1024 * 1024
DEFAULT_DDL_MISS_PENALTY = 10.0
DEFAULT_DYNAMIC_DDL_COST_FACTOR = 2.0
DEFAULT_DYNAMIC_DDL_MARGIN_S = 0.05
DEFAULT_ADAPTIVE_DDL_TARGET_ALL_MISS_RATE = 0.05
DEFAULT_ADAPTIVE_DDL_EWMA_ALPHA = 0.05
DEFAULT_ADAPTIVE_DDL_GROWTH_FACTOR = 0.10
DEFAULT_ADAPTIVE_DDL_SHRINK_FACTOR = 0.02
DEFAULT_ADAPTIVE_DDL_MARGIN_S = 0.02
BACKPRESSURE_PENDING_PER_WORKER = 4
BACKPRESSURE_PENALTY_FACTOR = 2.0
DDL_HARD_FACTOR = 2.0
MIN_BANDWIDTH = 1.0


@dataclass
class PeerLinkState:
    target_rank: int
    transport_mode: str = "unknown"
    bandwidth_bytes_per_s: float = 0.0
    avg_latency_s: float = 0.0
    inflight_forwards: int = 0
    inflight_bytes: float = 0.0
    contention_factor: float = 1.0


@dataclass
class DataServerRuntimeState:
    num_workers: int = 1
    active_tasks: int = 0
    pending_tasks: int = 0
    avg_task_latency_s: float = 0.01
    peer_links: dict[tuple[int, str], PeerLinkState] = field(default_factory=dict)


@dataclass
class AdaptiveDDLState:
    deadline_s: float
    profile_hint_deadline_s: float
    all_miss_ewma: float = 0.0
    updates: int = 0


@dataclass
class CandidateScore:
    ds_id: str
    rank: int
    score: float
    network_link_cost: float
    io_link_cost: float
    cpu_cost: float
    queue_wait_delay: float
    transport_mode: str
    io_locality: str
    predicted_finish_s: float
    ddl_profile_bucket: str | None
    profile_hint_ddl_deadline_s: float | None
    effective_ddl_deadline_s: float | None
    adaptive_ddl_all_miss_ewma: float | None
    adaptive_ddl_updates: int
    ddl_slack_s: float | None
    ddl_miss_s: float
    backpressure_penalty_s: float
    overloaded: bool
    deadline_bad_fit: bool


@dataclass
class SchedulerDecision:
    selected_ds_id: str
    candidates: list[CandidateScore]
    profile: RequestCostProfile
    local_ds_id: str | None
    reason: str

    @property
    def best_score_ds_id(self) -> str:
        return min(self.candidates, key=lambda candidate: candidate.score).ds_id


class CostAwareScheduler:
    """Score candidate DataServers using offline profiles and online feedback."""

    def __init__(
        self,
        profile_store: CostProfileStore | None = None,
        remote_min_gain: float = DEFAULT_REMOTE_MIN_GAIN,
        ddl_deadline_s: float | None = None,
        ddl_miss_penalty: float = DEFAULT_DDL_MISS_PENALTY,
        dynamic_ddl_enabled: bool = True,
        dynamic_ddl_cost_factor: float = DEFAULT_DYNAMIC_DDL_COST_FACTOR,
        dynamic_ddl_margin_s: float = DEFAULT_DYNAMIC_DDL_MARGIN_S,
        adaptive_ddl_enabled: bool = True,
        adaptive_ddl_target_all_miss_rate: float = DEFAULT_ADAPTIVE_DDL_TARGET_ALL_MISS_RATE,
        adaptive_ddl_ewma_alpha: float = DEFAULT_ADAPTIVE_DDL_EWMA_ALPHA,
        adaptive_ddl_growth_factor: float = DEFAULT_ADAPTIVE_DDL_GROWTH_FACTOR,
        adaptive_ddl_shrink_factor: float = DEFAULT_ADAPTIVE_DDL_SHRINK_FACTOR,
        adaptive_ddl_margin_s: float = DEFAULT_ADAPTIVE_DDL_MARGIN_S,
    ):
        self.profile_store = profile_store or CostProfileStore()
        self.remote_min_gain = remote_min_gain
        self.ddl_deadline_s = ddl_deadline_s if ddl_deadline_s and ddl_deadline_s > 0 else None
        self.ddl_miss_penalty = max(float(ddl_miss_penalty), 0.0)
        self.dynamic_ddl_enabled = bool(dynamic_ddl_enabled)
        self.dynamic_ddl_cost_factor = max(float(dynamic_ddl_cost_factor), 0.0)
        self.dynamic_ddl_margin_s = max(float(dynamic_ddl_margin_s), 0.0)
        self.adaptive_ddl_enabled = bool(adaptive_ddl_enabled)
        self.adaptive_ddl_target_all_miss_rate = min(max(float(adaptive_ddl_target_all_miss_rate), 0.0), 1.0)
        self.adaptive_ddl_ewma_alpha = min(max(float(adaptive_ddl_ewma_alpha), 0.0), 1.0)
        self.adaptive_ddl_growth_factor = max(float(adaptive_ddl_growth_factor), 0.0)
        self.adaptive_ddl_shrink_factor = min(max(float(adaptive_ddl_shrink_factor), 0.0), 1.0)
        self.adaptive_ddl_margin_s = max(float(adaptive_ddl_margin_s), 0.0)
        self._runtime_state: dict[str, DataServerRuntimeState] = {}
        self._adaptive_ddl_state: dict[str, AdaptiveDDLState] = {}

    def update_dataserver_metrics(self, ds_id: str, metrics: Any) -> None:
        if metrics is None:
            return
        state = self._runtime_state.setdefault(ds_id, DataServerRuntimeState())
        state.num_workers = max(int(_get_metric(metrics, "num_workers", state.num_workers) or 1), 1)
        state.active_tasks = max(int(_get_metric(metrics, "active_tasks", 0) or 0), 0)
        state.pending_tasks = max(int(_get_metric(metrics, "pending_tasks", 0) or 0), 0)
        state.avg_task_latency_s = max(float(_get_metric(metrics, "avg_task_latency_s", 0.01) or 0.01), 0.0)

        for peer in _get_repeated_metric(metrics, "peer_links"):
            target_rank = int(_get_metric(peer, "target_rank", -1))
            if target_rank < 0:
                continue
            transport_mode = str(_get_metric(peer, "transport_mode", "unknown") or "unknown")
            state.peer_links[(target_rank, transport_mode)] = PeerLinkState(
                target_rank=target_rank,
                transport_mode=transport_mode,
                bandwidth_bytes_per_s=max(float(_get_metric(peer, "bandwidth_bytes_per_s", 0.0) or 0.0), 0.0),
                avg_latency_s=max(float(_get_metric(peer, "avg_latency_s", 0.0) or 0.0), 0.0),
                inflight_forwards=max(int(_get_metric(peer, "inflight_forwards", 0) or 0), 0),
                inflight_bytes=max(float(_get_metric(peer, "inflight_bytes", 0.0) or 0.0), 0.0),
                contention_factor=max(float(_get_metric(peer, "contention_factor", 1.0) or 1.0), 1.0),
            )

    def score_candidates(
        self,
        index_request: Any,
        clients_and_data_servers: Any,
        candidate_ds_ids: list[str],
        local_ds_id: str | None = None,
    ) -> SchedulerDecision:
        indices = [int(index) for index in index_request.index]
        profile = self.profile_store.estimate_request(indices)
        total_data_servers = clients_and_data_servers.GetTotalDataServers()
        target_rank = self._resolve_target_rank(local_ds_id, total_data_servers)

        candidates = self._score_candidate_set(
            candidate_ds_ids,
            total_data_servers,
            profile,
            local_ds_id,
            target_rank,
        )
        if not candidates:
            raise RuntimeError("There is no valid dataserver candidate for cost-aware scheduling.")
        if self._update_adaptive_ddl_state(profile, candidates):
            candidates = self._score_candidate_set(
                candidate_ds_ids,
                total_data_servers,
                profile,
                local_ds_id,
                target_rank,
            )

        selected = self._select_with_local_threshold(candidates, local_ds_id)
        return SchedulerDecision(
            selected_ds_id=selected.ds_id,
            candidates=candidates,
            profile=profile,
            local_ds_id=local_ds_id,
            reason=self._selection_reason(selected.ds_id, candidates, local_ds_id),
        )

    def _score_candidate_set(
        self,
        candidate_ds_ids: list[str],
        total_data_servers: dict[str, Any],
        profile: RequestCostProfile,
        local_ds_id: str | None,
        target_rank: int | None,
    ) -> list[CandidateScore]:
        return [
            self._score_one(ds_id, total_data_servers[ds_id], profile, local_ds_id, target_rank)
            for ds_id in candidate_ds_ids
            if ds_id in total_data_servers
        ]

    def _score_one(
        self,
        ds_id: str,
        ds_info: dict[str, Any],
        profile: RequestCostProfile,
        local_ds_id: str | None,
        target_rank: int | None,
    ) -> CandidateScore:
        state = self._runtime_state.get(ds_id, DataServerRuntimeState())
        rank = int(ds_info.get("rank", -1))
        network_cost, transport_mode = self._network_cost(
            ds_id, ds_info, profile, local_ds_id, target_rank, state
        )
        io_cost = self._io_cost(profile)
        cpu_cost = self._cpu_cost(profile, state)
        queue_delay = self._queue_delay(state, profile)
        predicted_finish_s = network_cost + io_cost + cpu_cost + queue_delay
        overloaded = self._is_overloaded(state)
        backpressure_penalty = self._backpressure_penalty(state, overloaded)
        (
            ddl_profile_bucket,
            profile_hint_ddl_deadline_s,
            effective_ddl_deadline_s,
            adaptive_ddl_all_miss_ewma,
            adaptive_ddl_updates,
        ) = self._ddl_deadline_context(profile)
        ddl_slack_s, ddl_miss_s = self._ddl_slack_and_miss(predicted_finish_s, effective_ddl_deadline_s)
        deadline_bad_fit = self._deadline_bad_fit(predicted_finish_s, effective_ddl_deadline_s)
        score = predicted_finish_s + backpressure_penalty + self.ddl_miss_penalty * ddl_miss_s
        return CandidateScore(
            ds_id=ds_id,
            rank=rank,
            score=score,
            network_link_cost=network_cost,
            io_link_cost=io_cost,
            cpu_cost=cpu_cost,
            queue_wait_delay=queue_delay,
            transport_mode=transport_mode,
            io_locality=profile.io_locality_hint,
            predicted_finish_s=predicted_finish_s,
            ddl_profile_bucket=ddl_profile_bucket,
            profile_hint_ddl_deadline_s=profile_hint_ddl_deadline_s,
            effective_ddl_deadline_s=effective_ddl_deadline_s,
            adaptive_ddl_all_miss_ewma=adaptive_ddl_all_miss_ewma,
            adaptive_ddl_updates=adaptive_ddl_updates,
            ddl_slack_s=ddl_slack_s,
            ddl_miss_s=ddl_miss_s,
            backpressure_penalty_s=backpressure_penalty,
            overloaded=overloaded,
            deadline_bad_fit=deadline_bad_fit,
        )

    def _network_cost(
        self,
        ds_id: str,
        ds_info: dict[str, Any],
        profile: RequestCostProfile,
        local_ds_id: str | None,
        target_rank: int | None,
        state: DataServerRuntimeState,
    ) -> tuple[float, str]:
        if ds_id == local_ds_id:
            return 0.0, "local"

        peer = self._find_peer_link(state, target_rank)
        if peer is not None:
            bandwidth = peer.bandwidth_bytes_per_s or self._default_bandwidth(peer.transport_mode)
            contention = max(peer.contention_factor, self._contention_from_inflight(peer.inflight_forwards))
            effective = max(bandwidth / contention, MIN_BANDWIDTH)
            inflight_bytes = peer.inflight_bytes or profile.output_bytes * peer.inflight_forwards
            cost = (
                profile.output_bytes / effective
                + inflight_bytes / effective
                + peer.avg_latency_s
                + self._transport_overhead(peer.transport_mode)
            )
            return cost, peer.transport_mode

        if target_rank is None:
            transport_mode = "client_rpc"
            bandwidth = DEFAULT_TCP_BANDWIDTH
        elif ds_info.get("rank") == target_rank:
            transport_mode = "local"
            bandwidth = DEFAULT_SHM_BANDWIDTH
        else:
            transport_mode = "unknown_remote"
            bandwidth = DEFAULT_UNKNOWN_BANDWIDTH
        cost = profile.output_bytes / max(bandwidth, MIN_BANDWIDTH) + self._transport_overhead(transport_mode)
        return cost, transport_mode

    @staticmethod
    def _find_peer_link(state: DataServerRuntimeState, target_rank: int | None) -> PeerLinkState | None:
        if target_rank is None:
            return None
        links = [link for (rank, _), link in state.peer_links.items() if rank == target_rank]
        if not links:
            return None
        return max(links, key=lambda link: link.bandwidth_bytes_per_s)

    @staticmethod
    def _contention_from_inflight(inflight_forwards: int) -> float:
        return 1.0 + 0.5 * max(0, inflight_forwards - 1)

    @staticmethod
    def _default_bandwidth(transport_mode: str) -> float:
        if transport_mode in {"local", "SHM", "shm"}:
            return DEFAULT_SHM_BANDWIDTH
        if transport_mode in {"RDMA", "HCCL", "comm", "comm+grpc_inline"}:
            return DEFAULT_FAST_BANDWIDTH
        if transport_mode in {"TCP", "grpc_inline", "grpc_fallback", "client_rpc"}:
            return DEFAULT_TCP_BANDWIDTH
        return DEFAULT_UNKNOWN_BANDWIDTH

    @staticmethod
    def _transport_overhead(transport_mode: str) -> float:
        if transport_mode in {"local"}:
            return 0.0
        if transport_mode in {"SHM", "shm"}:
            return 0.0001
        if transport_mode in {"RDMA", "HCCL", "comm", "comm+grpc_inline"}:
            return 0.0005
        if transport_mode in {"TCP", "grpc_inline", "grpc_fallback", "client_rpc"}:
            return 0.001
        return 0.002

    @staticmethod
    def _io_cost(profile: RequestCostProfile) -> float:
        bandwidth_by_locality = {
            "local_shard": 2 * 1024 * 1024 * 1024,
            "same_host_shared": 1024 * 1024 * 1024,
            "remote_fs": 256 * 1024 * 1024,
            "object_store": 128 * 1024 * 1024,
            "unknown": 512 * 1024 * 1024,
        }
        penalty_by_locality = {
            "local_shard": 0.0,
            "same_host_shared": 0.0005,
            "remote_fs": 0.002,
            "object_store": 0.005,
            "unknown": 0.001,
        }
        locality = profile.io_locality_hint or "unknown"
        bandwidth = bandwidth_by_locality.get(locality, bandwidth_by_locality["unknown"])
        return profile.read_bytes / max(bandwidth, MIN_BANDWIDTH) + penalty_by_locality.get(locality, 0.001)

    @staticmethod
    def _cpu_cost(profile: RequestCostProfile, state: DataServerRuntimeState) -> float:
        load_degrade = max(1.0, state.active_tasks / max(state.num_workers, 1))
        return profile.cpu_cost_s * load_degrade

    @staticmethod
    def _queue_delay(state: DataServerRuntimeState, profile: RequestCostProfile) -> float:
        avg_latency = max(state.avg_task_latency_s, profile.cpu_cost_s)
        return state.pending_tasks * avg_latency / max(state.num_workers, 1)

    @staticmethod
    def _is_overloaded(state: DataServerRuntimeState) -> bool:
        return state.pending_tasks > max(state.num_workers, 1) * BACKPRESSURE_PENDING_PER_WORKER

    @staticmethod
    def _backpressure_penalty(state: DataServerRuntimeState, overloaded: bool) -> float:
        if not overloaded:
            return 0.0
        return BACKPRESSURE_PENALTY_FACTOR * max(state.avg_task_latency_s, 0.001)

    def _ddl_deadline_context(
        self,
        profile: RequestCostProfile,
    ) -> tuple[str | None, float | None, float | None, float | None, int]:
        if self.ddl_deadline_s is None:
            return None, None, None, None, 0
        if not self.dynamic_ddl_enabled:
            return None, None, self.ddl_deadline_s, None, 0
        bucket = self._ddl_profile_bucket(profile)
        profile_hint = self._profile_hint_deadline(profile)
        if not self.adaptive_ddl_enabled:
            return bucket, profile_hint, profile_hint, None, 0
        state = self._adaptive_ddl_state_for(bucket, profile_hint)
        return bucket, state.profile_hint_deadline_s, state.deadline_s, state.all_miss_ewma, state.updates

    def _profile_hint_deadline(self, profile: RequestCostProfile) -> float:
        if self.ddl_deadline_s is None:
            return 0.0
        profile_deadline = profile.cpu_cost_s * self.dynamic_ddl_cost_factor + self.dynamic_ddl_margin_s
        return max(self.ddl_deadline_s, profile_deadline)

    def _adaptive_ddl_state_for(self, bucket: str, profile_hint_deadline_s: float) -> AdaptiveDDLState:
        if self.ddl_deadline_s is None:
            raise RuntimeError("adaptive DDL state requires a base DDL deadline.")
        state = self._adaptive_ddl_state.get(bucket)
        if state is None:
            state = AdaptiveDDLState(
                deadline_s=self.ddl_deadline_s,
                profile_hint_deadline_s=max(profile_hint_deadline_s, self.ddl_deadline_s),
            )
            self._adaptive_ddl_state[bucket] = state
            return state
        state.profile_hint_deadline_s = max(state.profile_hint_deadline_s, profile_hint_deadline_s)
        state.deadline_s = min(max(state.deadline_s, self.ddl_deadline_s), state.profile_hint_deadline_s)
        return state

    def _update_adaptive_ddl_state(
        self,
        profile: RequestCostProfile,
        candidates: list[CandidateScore],
    ) -> bool:
        if not self._adaptive_ddl_active() or not candidates:
            return False

        bucket = self._ddl_profile_bucket(profile)
        state = self._adaptive_ddl_state_for(bucket, self._profile_hint_deadline(profile))
        before = state.deadline_s
        all_miss = all(candidate.ddl_miss_s > 0.0 for candidate in candidates)
        min_predicted_finish = min(candidate.predicted_finish_s for candidate in candidates)
        max_slack = max(
            (candidate.ddl_slack_s for candidate in candidates if candidate.ddl_slack_s is not None),
            default=0.0,
        )

        alpha = self.adaptive_ddl_ewma_alpha
        state.all_miss_ewma = (1.0 - alpha) * state.all_miss_ewma + alpha * (1.0 if all_miss else 0.0)
        state.updates += 1

        if all_miss or state.all_miss_ewma > self.adaptive_ddl_target_all_miss_rate:
            miss_gap = max(0.0, min_predicted_finish + self.adaptive_ddl_margin_s - state.deadline_s)
            pressure = max(
                state.all_miss_ewma - self.adaptive_ddl_target_all_miss_rate,
                alpha if all_miss else 0.0,
            )
            step_growth = max(
                state.deadline_s * self.adaptive_ddl_growth_factor * max(pressure, alpha),
                miss_gap * self.adaptive_ddl_growth_factor,
                self.adaptive_ddl_margin_s if all_miss else 0.0,
            )
            state.deadline_s = min(state.profile_hint_deadline_s, state.deadline_s + step_growth)
        elif (
            state.deadline_s > self.ddl_deadline_s
            and state.all_miss_ewma < self.adaptive_ddl_target_all_miss_rate * 0.25
            and max_slack > self.adaptive_ddl_margin_s
        ):
            target = max(self.ddl_deadline_s, min_predicted_finish + self.adaptive_ddl_margin_s)
            shrink = state.deadline_s * (1.0 - self.adaptive_ddl_shrink_factor)
            state.deadline_s = max(self.ddl_deadline_s, min(shrink, max(target, self.ddl_deadline_s)))

        return abs(state.deadline_s - before) > 1e-9

    def _adaptive_ddl_active(self) -> bool:
        return bool(self.ddl_deadline_s is not None and self.dynamic_ddl_enabled and self.adaptive_ddl_enabled)

    @staticmethod
    def _ddl_profile_bucket(profile: RequestCostProfile) -> str:
        sample_count = max(int(getattr(profile, "sample_count", 1) or 1), 1)
        cpu_per_sample = max(float(profile.cpu_cost_s), 0.0) / sample_count
        output_mb_per_sample = max(float(profile.output_bytes), 0.0) / sample_count / (1024.0 * 1024.0)
        output_bucket_mb = int(round(output_mb_per_sample / 8.0) * 8)
        return f"n{sample_count}:cpu{cpu_per_sample:.3f}:out{output_bucket_mb}mb"

    @staticmethod
    def _ddl_slack_and_miss(
        predicted_finish_s: float,
        effective_ddl_deadline_s: float | None,
    ) -> tuple[float | None, float]:
        if effective_ddl_deadline_s is None:
            return None, 0.0
        slack = effective_ddl_deadline_s - predicted_finish_s
        return slack, max(0.0, -slack)

    @staticmethod
    def _deadline_bad_fit(predicted_finish_s: float, effective_ddl_deadline_s: float | None) -> bool:
        return effective_ddl_deadline_s is not None and predicted_finish_s > effective_ddl_deadline_s * DDL_HARD_FACTOR

    def _select_with_local_threshold(
        self,
        candidates: list[CandidateScore],
        local_ds_id: str | None,
    ) -> CandidateScore:
        viable = [candidate for candidate in candidates if not self._guarded(candidate)] or candidates
        best = min(viable, key=self._candidate_priority)
        if local_ds_id is None:
            return best

        local = next((candidate for candidate in candidates if candidate.ds_id == local_ds_id), None)
        if local is None:
            return best
        if best.ds_id == local_ds_id:
            return local
        if self._guarded(local) and not self._guarded(best):
            return best
        if self.ddl_deadline_s is not None and local.ddl_miss_s > 0.0 and best.ddl_miss_s <= 0.0:
            return best

        local_priority = self._candidate_priority(local)
        best_priority = self._candidate_priority(best)
        if local_priority <= best_priority:
            return local

        gain = self._local_to_remote_gain(local, best)
        if gain > self.remote_min_gain:
            return best
        return local

    def _candidate_priority(self, candidate: CandidateScore) -> tuple:
        if self.ddl_deadline_s is None:
            return (self._guarded(candidate), candidate.score, candidate.predicted_finish_s)
        return (
            self._guarded(candidate),
            candidate.ddl_miss_s > 0.0,
            candidate.ddl_miss_s,
            candidate.predicted_finish_s,
            candidate.score,
        )

    @staticmethod
    def _guarded(candidate: CandidateScore) -> bool:
        return candidate.overloaded or candidate.deadline_bad_fit

    def _local_to_remote_gain(self, local: CandidateScore, remote: CandidateScore) -> float:
        if self.ddl_deadline_s is None:
            return local.score - remote.score
        if local.ddl_miss_s > 0.0 or remote.ddl_miss_s > 0.0:
            return local.ddl_miss_s - remote.ddl_miss_s
        return local.predicted_finish_s - remote.predicted_finish_s

    def _selection_reason(
        self,
        selected_ds_id: str,
        candidates: list[CandidateScore],
        local_ds_id: str | None,
    ) -> str:
        if local_ds_id is None:
            return "best_score_no_local_ds"
        if self.ddl_deadline_s is not None:
            selected = next((candidate for candidate in candidates if candidate.ds_id == selected_ds_id), None)
            local = next((candidate for candidate in candidates if candidate.ds_id == local_ds_id), None)
            if selected is not None and selected.ddl_miss_s <= 0.0 and local is not None and local.ddl_miss_s > 0.0:
                return "ddl_remote_meets_deadline"
            if selected is not None and selected.ddl_miss_s > 0.0:
                return "ddl_min_miss"
            if selected is not None and (selected.overloaded or selected.deadline_bad_fit):
                return "all_candidates_guarded"
        if selected_ds_id == local_ds_id:
            return "remote_gain_below_threshold_keep_local"
        local = next((candidate for candidate in candidates if candidate.ds_id == local_ds_id), None)
        selected = next((candidate for candidate in candidates if candidate.ds_id == selected_ds_id), None)
        if local is None or selected is None:
            return "best_score"
        return f"remote_gain_exceeds_threshold:{self._local_to_remote_gain(local, selected):.6f}"

    @staticmethod
    def _resolve_target_rank(local_ds_id: str | None, total_data_servers: dict[str, dict[str, Any]]) -> int | None:
        if local_ds_id is None:
            return None
        local_info = total_data_servers.get(local_ds_id)
        if local_info is None:
            return None
        rank = local_info.get("rank")
        return None if rank is None else int(rank)


def _get_metric(metrics: Any, name: str, default: Any = None) -> Any:
    if isinstance(metrics, dict):
        return metrics.get(name, default)
    return getattr(metrics, name, default)


def _get_repeated_metric(metrics: Any, name: str) -> list[Any]:
    value = _get_metric(metrics, name, [])
    if value is None:
        return []
    return list(value)
