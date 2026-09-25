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
"""Selective exchange optimization for DDLB stage-2 redistribution.

Core optimization: only transmit samples that actually need to be moved,
rather than full all_to_all exchange matrix.

Benefit: reduce transmission from O(N^2) to O(actual_routes).
"""

import os
from typing import Any


def should_use_selective_exchange(plan: dict, participant_world_size: int) -> bool:
    """Decide whether to use selective exchange based on route sparsity.

    Args:
        plan: Redistribution plan with routes
        participant_world_size: Number of participants in LB

    Returns:
        True if selective exchange is beneficial
    """
    if not os.environ.get("DDLB_SELECTIVE_EXCHANGE", "1") == "1":
        return False

    routes = plan.get("routes", [])
    if not routes:
        return False

    # Calculate route density
    max_possible_routes = participant_world_size * participant_world_size
    actual_routes = len(routes)
    density = actual_routes / max(max_possible_routes, 1)

    # Heuristic threshold: use selective when density < 70%
    # (saves at least 30% bandwidth)
    threshold = float(os.environ.get("DDLB_SELECTIVE_EXCHANGE_THRESHOLD", "0.7"))

    return density < threshold


def compute_selective_exchange_stats(plan: dict, participant_world_size: int) -> dict[str, Any]:
    """Compute statistics for selective exchange vs full all_to_all.

    Args:
        plan: Redistribution plan with routes
        participant_world_size: Number of participants

    Returns:
        Dictionary with transmission statistics
    """
    routes = plan.get("routes", [])
    actual_routes = len(routes)
    max_possible_routes = participant_world_size * participant_world_size

    # Count actual peer-to-peer transfers needed
    peer_pairs = set()
    for route in routes:
        src = route.get("src_ds")
        dst = route.get("target_ds")
        if src != dst:  # Skip self-routes
            peer_pairs.add((src, dst))

    actual_peer_transfers = len(peer_pairs)
    max_peer_transfers = participant_world_size * (participant_world_size - 1)

    density = actual_routes / max(max_possible_routes, 1)
    peer_density = actual_peer_transfers / max(max_peer_transfers, 1)

    bandwidth_saving = 1.0 - density

    return {
        "actual_routes": actual_routes,
        "max_possible_routes": max_possible_routes,
        "route_density": density,
        "actual_peer_transfers": actual_peer_transfers,
        "max_peer_transfers": max_peer_transfers,
        "peer_density": peer_density,
        "bandwidth_saving_ratio": bandwidth_saving,
        "use_selective": should_use_selective_exchange(plan, participant_world_size),
    }


def build_selective_send_map(plan: dict, ds_rank: int) -> dict[int, list[dict]]:
    """Build send map for selective exchange from this DS.

    Args:
        plan: Redistribution plan
        ds_rank: Current DS rank

    Returns:
        Dictionary mapping dst_rank -> list of routes to send
    """
    send_map = {}

    for route in plan.get("routes", []):
        if route["src_ds"] == ds_rank and route["target_ds"] != ds_rank:
            dst_rank = route["target_ds"]
            if dst_rank not in send_map:
                send_map[dst_rank] = []
            send_map[dst_rank].append(route)

    return send_map


def build_selective_recv_map(plan: dict, ds_rank: int) -> dict[int, list[dict]]:
    """Build receive map for selective exchange to this DS.

    Args:
        plan: Redistribution plan
        ds_rank: Current DS rank

    Returns:
        Dictionary mapping src_rank -> list of routes to receive
    """
    recv_map = {}

    for route in plan.get("routes", []):
        if route["target_ds"] == ds_rank and route["src_ds"] != ds_rank:
            src_rank = route["src_ds"]
            if src_rank not in recv_map:
                recv_map[src_rank] = []
            recv_map[src_rank].append(route)

    return recv_map
