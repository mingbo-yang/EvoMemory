"""Bind ALFWorld game assets to the published, ordered CrossEp-Emb test mapping."""
import hashlib
import json

from benchmarks.alfworld.panel import AUDIT, load_panel as load_assets

MAPPING = AUDIT / 'crossep_emb_mappings_test.json'
MAPPING_SHA256 = 'fd24a8525a4c0464fd1fae130763cd736d326d6a687dcb7d48bda40d93e83c00'
GAME_ARCHIVE_SHA256 = '5df77ea759f2211a4106082839ddbbb790f1ba4e7d097ed732cf453f72aa36cf'


def load_panel():
    raw = MAPPING.read_bytes()
    if hashlib.sha256(raw).hexdigest() != MAPPING_SHA256:
        raise ValueError('Published CrossEp-Emb test mapping content differs from the pinned file')
    mapping = json.loads(raw)
    audit = json.loads((AUDIT / 'crossep_emb_manifest_audit.json').read_bytes())
    if audit['archive_sha256'] != GAME_ARCHIVE_SHA256:
        raise ValueError('ALFWorld game archive identity differs from the verified asset release')
    tasks = load_assets()
    for entry, task in zip(mapping, tasks, strict=True):
        member = f"json_2.1.1/valid_train/{entry['task_type']}/{entry['task_id']}/game.tw-pddl"
        if (entry['item_id'] != task['benchmark_item_id'] or member != task['archive_member']
                or task['benchmark_split'] != 'test' or task['split'] != 'valid_train'):
            raise ValueError(f"Published CrossEp-Emb test mapping mismatch for item {entry['item_id']}")
    return tasks
