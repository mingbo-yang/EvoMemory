# Architecture and code map

The current layout separates role implementations, orchestration, reports,
benchmark adapters, and tests. This is a structural refactor; the model-facing
prompts, event contract, scoring, resource budgets, and adoption policy are
unchanged. The original development checkout and experiment archives are not edited.

```text
EvoMemory/
├── evomem/                         # Common infrastructure and ALMA/model transport
├── phase1_codegen/                 # Memory-program evolution engine
│   ├── roles/
│   │   ├── reviewer.py             # Fixed Reviewer and evidence-bound review contract
│   │   └── evolver.py              # Fixed Evolver and proposal contract
│   ├── reporting/
│   │   └── campaign.py             # Offline campaign report assembly
│   ├── engine.py                   # Orchestration; roles do not deploy their own edits
│   ├── acceptance.py               # Deployment eligibility
│   ├── experience.py               # Fixed Evolution Memory projection and retrieval
│   ├── memory.py, hooks.py         # Task Memory state and event lifecycle
│   ├── program.py, worker.py       # Bounded generated-code execution
│   ├── policy.py, isolation.py     # Language checks and syscall isolation
│   ├── native.py, textworld_native.py
│   └── campaign.py                 # Registered ALMA campaign entry point
├── math_apps_online/               # Online task scheduling and transactions
├── benchmarks/
│   ├── math_apps/                  # Public task adapters, host judge, one-turn Actor
│   └── alfworld/                   # Public CrossEp-Emb task mapping
├── scripts/                        # Explicit local model and asset preparation
├── tests/
│   ├── core/                       # Engine, roles, memory and runtime contracts
│   ├── online/                     # Stream and evaluator transactions
│   └── test_*.py                   # Release and import-boundary regressions
├── docs/                           # Canonical protocol and setup documentation
└── provenance/                     # Export and structural-validation records
```

## Main call paths

- `evomemory` / `python -m math_apps_online`: plans or runs the registered
  online task stream. `online.py` owns durable stream state; `search.py` owns
  per-item repair; `native.py` binds measured attempts and commits.
- `evomemory-campaign` / `python -m phase1_codegen`: dispatches registered
  ALMA campaign phases through `campaign.py` and the shared `engine.py`.
- `python -m phase1_codegen.reporting.campaign`: reads saved artifacts to
  assemble a report; it does not run an Executor or change deployed memory.

The Reviewer diagnoses and evaluates. The Evolver proposes code. The host engine
and acceptance policy decide deployment. Native adapters own measured outcomes;
reviewer opinions do not manufacture task rewards. `experience.py` retains the
fixed second-level memory implementation.

## Import and ownership rules

Runtime modules never import test helpers. Tests may share explicitly synthetic
fixtures under `tests/core/` and are excluded from installation packages.
Benchmark adapters use normal package imports. Importing an adapter does not
modify `sys.path`, download data, or rewrite another module's prompt dictionary.
The Actor owns its effective prompt, including the existing APPS judge constraints.
Official ALMA's own runtime module loading remains at the execution boundary.

The existing top-level package and public CLI names remain. Internal imports of
Reviewer/Evolver/report code now use `phase1_codegen.roles.reviewer`,
`phase1_codegen.roles.evolver`, and `phase1_codegen.reporting.campaign`.
The old internal module paths are removed rather than retained as forwarding layers.
Use a new run directory when bindings change; historical runs keep their frozen code.

## Documentation and tests

- [Online protocol](online_protocol.md) and [ALMA campaign](alma_campaign.md)
  are the sole detailed workflow documents. Package-local duplicate READMEs were removed.
- `python -m pytest -q` runs the bounded release suite.
- Run specific research contracts explicitly, for example
  `python -m pytest tests/core/test_review_contract.py tests/online/test_search.py`.
- `python -m pytest tests/core tests/online` requests the full development suite;
  native integrations and historical receipt cases require their separate assets.

`roles/reviewer.py` and `engine.py` still contain substantial evidence and
orchestration logic. This cleanup establishes their ownership boundaries; it does
not claim that all large modules have been decomposed or that research results improved.
