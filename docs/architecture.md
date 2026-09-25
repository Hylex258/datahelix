# Architecture and request flow

## Motivation

A training rank's local DataLoader workers may be unevenly occupied when
samples have different decode/transform costs or when CPU capacity is spread
across nodes. DataHelix separates *where a request is assigned* from *where
its tensors are delivered*. The coordinator sees lightweight cost and runtime
metadata; DataServers materialize samples and move results; training ranks
consume ordered batches.

This is a design for data preparation and delivery. It does not rebalance
model computation or gradient communication.

## Components

```text
training rank / client
    | index request                       ^ ready batch
    v                                     |
coordinator -- assignment --> executing DataServer
    ^       (metadata only)          | fetch / decode / transform / collate
    | heartbeat, queue and link state | 
    +---------------------------------+
                                      |
                         own local DS | or remote DS forwarding
                                      v
                         target rank's local DataServer
                                      |
                               local Queue / SHM
                                      v
                              training rank / client
```

Each training rank has one local DataServer as its delivery endpoint. A remote
DataServer is additional CPU capacity: it can perform the work, but forwards
the result to the target rank's local DataServer rather than bypassing that
endpoint. The coordinator selects an execution location; it does not relay
large tensor payloads.

## One logical batch

1. The client reads batch indices from its sampler. `IndexSenderThread` can
   split them into bounded microbatch requests and keep several requests in
   flight. `MicroBatchIndexResponseGroup` retains their logical grouping.
2. The coordinator selects a DataServer. `CostProfileStore` supplies estimated
   CPU work, read bytes, output bytes, and locality for the indices. The
   scheduler combines these with heartbeat-derived worker, queue, and peer-link
   state, then applies optional data-ready deadline and local/remote guards.
3. The selected DataServer prepares the samples. If it is not the target
   rank's local DataServer, it first sends task/tensor metadata to that local
   DataServer, then moves the tensor payload on the data plane.
4. The local DataServer delivers the result. The client collects microbatch
   results in request order and merges them into the original logical batch.

Steps 2--4 describe the complete system but their service and transport
implementations are deliberately **not** included here. The protobuf files
show the control-plane contract; `TensorExtractor` and `TensorSerializer`
illustrate how a nested object can be represented as tensor references plus
metadata. They are not, by themselves, a transport.

## Placement and balancing boundaries

The included scheduler estimates a candidate's finish time from network,
I/O, CPU, and queue components. A remote candidate is useful only when its
predicted gain can outweigh forwarding cost. Profile quality, queue estimates,
network contention, and deadline choice all affect that decision; the design
does not promise universal speedup.

DataServer-side redistribution is a separate stage after request placement.
The included `selective_exchange.py` demonstrates how a redistribution plan
can be reduced to actual cross-DS routes. The global round manager that
collects features, builds the plan, synchronizes participants, and guarantees
ordered delivery is intentionally omitted. Route-density statistics are
heuristics, not measured bandwidth savings.

## Trust and runtime assumptions

The original implementation uses Python object serialization for parts of the
data/control path. Only trusted peers and trusted serialized inputs should be
used; this snapshot is not a hardened public service. Correct execution also
requires the omitted RPC generation, process management, communication
backends, shared-memory lifecycle, and framework adapters.
