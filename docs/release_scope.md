# Release scope and validation

The original export described below is version 0.1.0, commit `e08a443`.
Version 0.2.0 reorganizes role/report modules and moves tests out of runtime
packages. See [architecture](architecture.md) and
[`layout_refactor.json`](../provenance/layout_refactor.json) for current paths
and structural validation. Original export receipts remain historical records.

This repository was exported from the current `evolution_memory_phase1_dev_20261107`
development checkout, including the November 20, 2026 changes. The original
checkout, model weights, benchmark assets, and historical experiment results
were not edited. The complete local source/artifact snapshot is maintained
separately; it is not uploaded to GitHub.

## Included

- Current mechanism, Reviewer, Evolver, transactional online scheduling, ALMA
  transport, event contracts, isolation, evidence, and adoption implementation.
- First-party runtime benchmark adapters formerly imported from dated
  development directories, now under `benchmarks/`.
- Existing CPU test sources, a bounded default release suite, and new
  portability/registration/judge contracts.
- Install metadata, environment configuration, explicit asset preparation,
  upstream notices, an MIT license for original code, and CI configuration.

## Portability changes

- Checkpoint, endpoint, model name, data roots, and capability model roots are
  explicit environment settings rather than a developer's home-directory paths.
- Model registration is generated per installation. No historical freeze or
  service/process-registration file is distributed.
- The release does not import the developer machine's GPU1 lease/drain scripts;
  it only verifies and uses the explicitly configured local endpoint.
- Optional model availability is explicit and frozen; setting it to zero is
  a declared capability-catalog change, not a claim of identical old conditions.
- Linux `libseccomp` lookup uses the platform loader; the filter and generated
  language/resource restrictions remain enforced.
- Tests requiring a real tokenizer or pinned ALMA/assets skip when those
  separately installed resources are absent. They are not replaced by mock results.

The original bytes of the engine, Reviewer, Evolver, Evolution Memory,
acceptance rules, protocol, policy, mechanism program, ALMA backend, and online
search/decision/stream modules match the source snapshot. The MATH/APPS grading
function ASTs are unchanged after adapter relocation. See
[`semantic_preservation.json`](../provenance/semantic_preservation.json).

## What was checked

The default release suite checks checkpoint identity and immutability, benchmark
import boundaries, preserved scoring/prompt settings, local-service routing,
token accounting, event/worker contracts, and transactional online state using
CPU fixtures. A separate real CPU judge smoke check executes a synthetic program
and rejects a prohibited import. These checks make no model or task-success claim.

Python sources compile, CLI help works without downloading assets, and a wheel
build succeeds without fetching dependencies. The validation record is
[`release_validation.json`](../provenance/release_validation.json).

The full original research test collection was not completed as part of this
release. Many of its integration cases require historical private archives,
prepared native games, or an installed real tokenizer. Invoke those package
test directories explicitly in a configured research environment.

No new model inference, native benchmark rollout, or effectiveness experiment
was run for publication. A clean installation of every optional model/TextWorld
runtime and the GitHub-hosted CI environment has not yet been verified.

## Excluded from publication

Model weights, benchmark downloads, judging answers and private tests, generated
mechanism archives, execution traces, local credentials, host process records,
developer GPU scheduling scripts, Python environments, and historical run outputs
are not part of this Git repository. Upstream assets must be obtained separately
under their own licenses.

The method figures retained under `docs/figures/` are historical development
artwork. The current root README, event protocol, and runtime source are
authoritative if a historical figure differs.
