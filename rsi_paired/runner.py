"""Run one engineering batch on an already resident model service."""
import os
from contextlib import ExitStack
from pathlib import Path

import httpx

from evomem.common import file_hash
from phase1_codegen.capabilities import LocalModelCapabilities
from phase1_codegen.local_service import verify_model, verify_local_service
from phase1_codegen.tokenization import Tokenizer

from . import native as native_module
from . import roles as roles_module
from . import workflow
from .native import BASELINE_SOURCE, Native
from .panel import save_registration
from .roles import INITIAL_STRATEGY, Roles


class SmokeAdapter(Roles):
    def __init__(self, root, cfg, tokenizer, native, fingerprint):
        super().__init__(root, cfg, tokenizer=tokenizer)
        self.native = native
        self.fingerprint = fingerprint

    def set_job(self, path):
        super().set_job(path)
        self.native.set_job(path)

    def learn(self, *args):
        return self.native.learn(*args)

    def evaluate(self, *args):
        return self.native.evaluate(*args)

    def prepare(self, *args):
        return self.native.prepare(*args)


def run_smoke(root, cfg, manifest, exercise_pair=False):
    from scripts.alma_feasibility import LEGACY, META
    if manifest.get("scope") != "engineering" or manifest.get("protocol") != "rsi_original_parent_paired_v2":
        raise ValueError("Expected a registered engineering panel")
    panels = manifest["panels"]
    if set(panels) != {"learning", "primary", "meta"} or any(len(rows) != 2 for rows in panels.values()):
        raise ValueError("The smoke entry point accepts two tasks per panel")
    if any(Path(t["archive_member"]).parts[1] != "train" for tasks in panels.values() for t in tasks):
        raise ValueError("Engineering smoke tasks must come from the training archive")
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
    model_hash = workflow.digest(checkpoint)
    cfg = dict(cfg, model_fingerprint=model_hash, seed=manifest["seed"])
    tokenizer = Tokenizer(cfg["tokenizer_path"])
    dependencies = dict(model_identity=identity, checkpoint_hash=model_hash,
                        panel_hash=workflow.digest(manifest), context_tokens=cfg["context_tokens"],
                        software={name: file_hash(path) for name, path in {
                            "roles": roles_module.__file__, "workflow": workflow.__file__,
                            "native": native_module.__file__, "runner": __file__
                        }.items()}, forced_engineering=exercise_pair,
                        model_config_hash=workflow.digest(cfg), role_limits=roles_module.ROLE_SETTINGS,
                        optional_models=os.environ.get("EVOMEMORY_ENABLE_OPTIONAL_MODELS", "1"))
    # Store a JSON-normalized value so tuple-valued sampling settings compare on recovery.
    import json
    with ExitStack() as resources:
        caps = LocalModelCapabilities(root / 'capabilities', 'alfworld', root / 'capability_costs.jsonl',
                                      tokenizer, model_hash, cfg['base_url'], phase='rsi_engineering')
        resources.callback(caps.close)
        cfg["model_catalog"] = caps.describe()
        all_tasks = [task for tasks in panels.values() for task in tasks]
        native = Native(root / 'native', all_tasks, tokenizer, model_hash, base_url=cfg["base_url"],
                        model_name=cfg["model_name"], capabilities=caps)
        resources.callback(native.close)
        dependencies.update(capability_fingerprint=caps.fingerprint, native_fingerprint=native.fingerprint)
        save_registration(root / "runtime.json", json.loads(workflow.canonical(dependencies)))
        adapter = SmokeAdapter(root, cfg, tokenizer, native, workflow.digest(dependencies))
        resources.callback(adapter.close)
        state = dict(mechanism=BASELINE_SOURCE, evolver=INITIAL_STRATEGY, memory={}, ledger=[], history=[])
        return workflow.run_batch(root / "batch", state=state, panels=panels, adapter=adapter,
                                  seed=manifest["seed"], force_pair=exercise_pair,
                                  budget={"episodes": 26, "infrastructure_attempts": 2})
