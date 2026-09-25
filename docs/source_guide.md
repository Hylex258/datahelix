# Source guide

The directory layout follows responsibilities rather than the original
repository's packaging. Files are copied unchanged so their original imports
still reference `distributed_dataloader`; this is intentional.

| Included file | Design point |
| --- | --- |
| `source/client/send_index.py` | Asynchronous index dispatch, bounded in-flight work, microbatch fan-out. |
| `source/client/microbatch.py` | Logical-batch split/group/ordered merge semantics. |
| `source/coordinator/cost_profile.py` | Normalized per-sample cost metadata and request aggregation. |
| `source/coordinator/cost_scheduler.py` | Candidate scoring from offline cost and online DataServer/link state; optional deadline and local/remote selection. |
| `source/data_server/selective_exchange.py` | Route-based sparse exchange decision and send/receive map construction. |
| `source/cluster/tensor_extractor.py` | Separating tensors from nested object structure. |
| `source/cluster/tensor_serializer.py` | Metadata serialization and structure reconstruction from tensor references. |
| `protocol/*.proto` | Client, coordinator, DataServer, heartbeat, index and metadata message contracts. |

The service implementations are **not** included: no coordinator registration
or `AssignIndex` handler, DataServer worker/result loop, ordered delivery,
DDLB round manager, training-facing iterator, or inter-node payload transport.
The snapshot also omits generated protobuf code, tests, dependency lockfiles,
benchmark scripts/results, profiles, manifests, datasets, and machine-specific
configuration. Consequently, this is a design reference, not a complete
implementation or a testable release.

For a code-reading path, start with `protocol/index.proto` and
`protocol/metadata.proto`, then the two client files, the two coordinator
files, and finally the data-plane files. The [architecture note](architecture.md)
connects those pieces and explicitly marks the missing service boundaries.
