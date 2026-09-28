# DataHelix: distributed data loading design snapshot

DataHelix explores how to decouple CPU-side data preparation from synchronous
distributed training. This directory is a **selected source and design snapshot**,
not a runnable package or a reproduction artifact. It is intended to make the
client / coordinator / DataServer split, scheduling inputs, and data-transfer
contract inspectable without publishing the complete MindData repository.


## Read first

- [Architecture](docs/architecture.md): component responsibilities and the
  request-to-delivery sequence.
- [Source guide](docs/source_guide.md): why each file is included and what is
  deliberately absent.
- [Release checklist](docs/release_checklist.md): checks before making this
  directory public.

## Scope

`source/` contains representative, unmodified modules for client-side request
fan-out, cost/profile-based placement, sparse redistribution planning, and
tensor metadata extraction. `protocol/` contains the original protobuf
interface definitions, but no generated Python stubs.

This snapshot does **not** include the top-level `DistributedDataLoader`,
Coordinator or DataServer service implementations, worker lifecycle, concrete
communication backends, deployment scripts, datasets, profiles, experiment
outputs, model code, or the paper. Imports in the copied modules may therefore
not resolve. Do not treat this directory as installable software or its source
as a performance benchmark.
