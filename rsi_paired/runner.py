"""Run one engineering batch on an already resident model service."""
from copy import deepcopy
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing.util import Finalize
from tempfile import TemporaryDirectory
from threading import Thread
from contextlib import ExitStack, contextmanager
from pathlib import Path

import httpx

from evomem.common import file_hash
from phase1_codegen.capabilities import LocalModelCapabilities
from phase1_codegen.local_service import verify_model, verify_local_service
from phase1_codegen.tokenization import Tokenizer

from . import native as native_module
from . import roles as roles_module
from . import workflow
from . import coevolution_roles, coevolution_workflow
from . import witness
from .native import BASELINE_SOURCE, Native
from .panel import COEVOLUTION_MIXED_10X60_PROTOCOL, COEVOLUTION_PROTOCOLS, SEQUENTIAL_PROTOCOLS, save_registration
from .roles import INITIAL_STRATEGY, Roles


_worker_native = None


def _exit_with_coordinator():
    # A killed CLI cannot shut down its pool. Stop issuing further Actor requests.
    multiprocessing.parent_process().join()
    os._exit(1)


def _start_candidate_worker(root, cfg, tasks, phase, protocol, expected_native):
    """Construct an independent ALMA environment inside the spawned process."""
    global _worker_native
    Thread(target=_exit_with_coordinator, daemon=True).start()
    with ExitStack() as resources:
        layout = resources.enter_context(TemporaryDirectory(prefix="evomemory-candidate-"))
        tokenizer = Tokenizer(cfg["tokenizer_path"], model_name=cfg["model_name"])
        caps = LocalModelCapabilities(root / "capabilities", "alfworld", root / "capability_costs.jsonl",
                                      tokenizer, cfg["model_fingerprint"], cfg["base_url"], phase=phase)
        resources.callback(caps.close)
        native = Native(root / "native", tasks, tokenizer, cfg["model_fingerprint"],
                        base_url=cfg["base_url"], model_name=cfg["model_name"], capabilities=caps,
                        protocol=protocol, runtime_root=layout)
        resources.callback(native.close)
        # The fingerprint covers the full task manifest, model and capability catalog.
        if native.fingerprint != expected_native:
            raise RuntimeError("Candidate worker differs from the registered native evaluator")
        _worker_native = native
        Finalize(None, resources.pop_all().close, exitpriority=10)


def _evaluate_candidate(job_path, source, memory, tasks, seeds):
    _worker_native.set_job(job_path)
    return _worker_native.evaluate(source, memory, tasks, seeds)


class SmokeAdapter(Roles):
    def __init__(self, root, cfg, tokenizer, native, fingerprint):
        super().__init__(root, cfg, tokenizer=tokenizer)
        self.native = native
        self.fingerprint = fingerprint
        self.candidate_workers = cfg.get("candidate_workers", 1)
        self.candidate_pool = None

    def set_job(self, path):
        super().set_job(path)
        self.native.set_job(path)

    def learn(self, *args):
        return self.native.learn(*args)

    def diagnose(self, mechanism, memory, ledger, history):
        evidence = self.native.verified_learning_evidence(ledger)
        return super().diagnose(mechanism, memory, evidence, history)

    def evaluate(self, *args):
        return self.native.evaluate(*args)

    def evaluate_isolated(self, job_path, source, memory, tasks, seeds):
        return self.candidate_pool.submit(_evaluate_candidate, job_path, source, memory, tasks, seeds).result()

    def prepare(self, *args):
        return self.native.prepare(*args)


class CoevolutionAdapter(coevolution_roles.CoevolutionRolesMixin, SmokeAdapter):
    """Native evaluator with independently versioned diagnostic and proposal policies."""


def run_smoke(root, cfg, manifest, exercise_pair=False):
    if manifest.get("scope") != "engineering" or manifest.get("protocol") not in (workflow.PROTOCOL, *SEQUENTIAL_PROTOCOLS):
        raise ValueError("Expected a registered engineering panel")
    panels = manifest["panels"]
    if set(panels) != {"learning", "primary", "meta"} or any(len(rows) != 2 for rows in panels.values()):
        raise ValueError("The smoke entry point accepts two tasks per panel")
    if any(Path(t["archive_member"]).parts[1] != "train" for tasks in panels.values() for t in tasks):
        raise ValueError("Engineering smoke tasks must come from the training archive")
    all_tasks = [task for tasks in panels.values() for task in tasks]
    with open_stage(root, cfg, manifest, all_tasks, exercise_pair) as adapter:
        state = initial_state()
        return workflow.run_batch(root / "batch", state=state, panels=panels, adapter=adapter, protocol=manifest["protocol"],
                                  seed=manifest["seed"], force_pair=exercise_pair,
                                  budget={"episodes": 26, "infrastructure_attempts": 2})


def initial_state(arm=None):
    state = dict(mechanism=BASELINE_SOURCE, evolver=deepcopy(INITIAL_STRATEGY), memory={}, ledger=[], history=[])
    if arm is not None:
        if arm not in coevolution_workflow.ARM_COMPONENTS:
            raise ValueError(f"Unknown co-evolution arm: {arm}")
        state.update(arm=arm, reviewer=deepcopy(coevolution_roles.INITIAL_REVIEWER))
    return state


@contextmanager
def open_stage(root, cfg, manifest, tasks, exercise_pair=False, *, learning_archive=None, shared_evolution_memory=(), arm=None):
    coevolving = manifest.get("protocol") in COEVOLUTION_PROTOCOLS
    if coevolving != (arm is not None) or (coevolving and arm not in coevolution_workflow.ARM_COMPONENTS):
        raise ValueError("The registered campaign protocol and experimental arm do not match")
    if coevolving:
        from .coevolution_panel import validate_coevolution_manifest
        validate_coevolution_manifest(manifest)
    workers = cfg.get("candidate_workers", 1)
    if type(workers) is not int or workers < 1:
        raise ValueError("candidate_workers must be a positive integer")
    from scripts.alma_feasibility import LEGACY, META
    if (file_hash(LEGACY / "downloads/alfworld-tw-0.4.2.zip") != manifest["games_sha256"]
            or file_hash(META) != manifest["metadata_sha256"]):
        raise ValueError("ALFWorld archive identity changed since panel preparation")
    root = Path(root).resolve()
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    checkpoint = verify_model()
    with httpx.Client(trust_env=False, timeout=15) as client:
        identity = verify_local_service(cfg["base_url"], client, model_name=cfg["model_name"],
                                        model_path=cfg["tokenizer_path"])
        models = client.get(cfg["base_url"] + "/models")
        models.raise_for_status()
        row = next(row for row in models.json()["data"] if row["id"] == cfg["model_name"])
        if type(row.get("max_model_len")) is not int or row["max_model_len"] < cfg["context_tokens"]:
            raise ValueError("Registered model context does not fit the advertised service")
        # vLLM regenerates the model card's created timestamp on each request.
        # Freeze stable advertised identity and capacity so a client can resume.
        identity.pop("models_response_hash")
        identity["max_model_len"] = row["max_model_len"]
    model_hash = workflow.digest(checkpoint)
    cfg = dict(cfg, model_fingerprint=model_hash, seed=manifest["seed"])
    tokenizer = Tokenizer(cfg["tokenizer_path"], model_name=cfg["model_name"])
    dependencies = dict(model_identity=identity, checkpoint_hash=model_hash,
                        panel_hash=workflow.digest(manifest), context_tokens=cfg["context_tokens"],
                        software={name: file_hash(path) for name, path in {
                            "roles": roles_module.__file__, "workflow": workflow.__file__,
                            "native": native_module.__file__, "witness": witness.__file__, "runner": __file__
                        }.items()}, forced_engineering=exercise_pair,
                        model_config_hash=workflow.digest(cfg), role_limits=roles_module.ROLE_SETTINGS,
                        optional_models=os.environ.get("EVOMEMORY_ENABLE_OPTIONAL_MODELS", "1"))
    if coevolving:
        dependencies["role_limits"] = CoevolutionAdapter.role_settings
        dependencies["arm"] = arm
        dependencies["software"].update(coevolution_roles=file_hash(coevolution_roles.__file__),
                                        coevolution_workflow=file_hash(coevolution_workflow.__file__))
    if shared_evolution_memory:
        dependencies["shared_evolution_memory_hash"] = workflow.digest(shared_evolution_memory)
    # Store a JSON-normalized value so tuple-valued sampling settings compare on recovery.
    import json
    with ExitStack() as resources:
        phase = 'rsi_engineering' if manifest['scope'] == 'engineering' else 'rsi_formal'
        caps = LocalModelCapabilities(root / 'capabilities', 'alfworld', root / 'capability_costs.jsonl',
                                      tokenizer, model_hash, cfg['base_url'], phase=phase)
        resources.callback(caps.close)
        cfg["model_catalog"] = caps.describe()
        native = Native(root / 'native', tasks, tokenizer, model_hash, base_url=cfg["base_url"],
                        model_name=cfg["model_name"], capabilities=caps, learning_archive=learning_archive,
                        protocol=manifest["protocol"])
        resources.callback(native.close)
        dependencies.update(capability_fingerprint=caps.fingerprint, native_fingerprint=native.fingerprint)
        runtime = root / "runtime.json"
        dependencies = json.loads(workflow.canonical(dependencies))
        registered = dependencies
        recovery = root / "evidence-recovery.json"
        if recovery.is_file():
            before = workflow._read(runtime)
            binding = workflow._read(recovery)
            changed = {"roles", "runner"}
            if (manifest["protocol"] != COEVOLUTION_MIXED_10X60_PROTOCOL or arm != "D"
                    or root.name != "evolution-01" or root.parent.name != arm
                    or binding != {"previous_runtime_hash": workflow.digest(before),
                                   "software": {k: dependencies["software"][k] for k in changed}}
                    or before != dict(dependencies, software={
                        k: before["software"][k] if k in changed else v
                        for k, v in dependencies["software"].items()
                    })):
                raise ValueError("Evidence recovery changed the registered execution environment")
            save_registration(root / "runtime-evidence-recovery.json", dependencies)
            registered = before
        save_registration(runtime, registered)
        adapter_type = CoevolutionAdapter if coevolving else SmokeAdapter
        adapter = adapter_type(root, cfg, tokenizer, native, workflow.digest(registered))
        if coevolving:
            adapter.arm = arm
            adapter.protocol = manifest["protocol"]
            adapter.registered_batches = tuple(workflow.digest(batch) for batch in manifest["batches"])
        adapter.shared_evolution_memory = tuple(shared_evolution_memory)
        resources.callback(adapter.close)
        if workers > 1:
            adapter.candidate_pool = resources.enter_context(ProcessPoolExecutor(
                max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                initializer=_start_candidate_worker,
                initargs=(root, cfg, tasks, phase, manifest["protocol"], native.fingerprint)))
        yield adapter
