# Configuration

Run from an editable checkout. Copy `.env.example` to `.env`, edit it, and
`source .env`; environment files are not loaded implicitly.

| Variable | Meaning |
|---|---|
| `EVOMEMORY_MODEL_PATH` | Required local checkpoint directory; no weights are downloaded automatically |
| `EVOMEMORY_MODEL_NAME` | Exact model ID advertised by the local endpoint |
| `EVOMEMORY_BASE_URL` | Explicit loopback HTTP `/v1` endpoint, default port 8140 |
| `EVOMEMORY_MODEL_MANIFEST` | Manifest produced by `scripts.register_model` |
| `EVOMEMORY_DATA_ROOT` | Asset workspace, default `data/` |
| `EVOMEMORY_ASSET_ROOT` | ALFWorld archive root with `downloads/`, default `data/` |
| `EVOMEMORY_BENCHMARK_DATA` | MATH/APPS host-side asset directory, default `data/math_apps/` |
| `EVOMEMORY_ALFWORLD_METADATA` | Override the official `json_2.1.1_json.zip` location |
| `EVOMEMORY_CAPABILITY_MODEL_ROOT` | Optional auxiliary checkpoint root, default `models/` |
| `EVOMEMORY_ENABLE_OPTIONAL_MODELS` | `1` requires the registered embeddings/vision assets and vision helper; `0` exposes only the configured local text endpoint |

The historical model catalog remains the default (`1`) for compatibility.
The example configuration explicitly selects `0` for installations without
auxiliary models. Record this choice: different catalogs are different frozen
conditions and must not be silently compared as identical configurations.
The optional catalog never prescribes how the Evolver should design memory.

On Linux install `libseccomp2` (Debian/Ubuntu) or the equivalent distribution
package. The trusted syscall filter is resolved through the platform loader.
APPS judging and mechanism validation use CPU/memory/output limits. Their test
fixtures do not certify isolation on every kernel or Python distribution.
Use a dedicated research account/container when running untrusted submissions.

All source changes require a new output directory. Do not resume an old run
whose code, prompts, checkpoint registration, data hashes, or settings differ.
