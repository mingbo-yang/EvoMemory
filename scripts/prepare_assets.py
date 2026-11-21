"""Download pinned public assets explicitly; never runs a model or benchmark task."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request
import zipfile

from evomem.common import DATA, LEGACY, ROOT, file_hash, write

GAME_SHA = '5df77ea759f2211a4106082839ddbbb790f1ba4e7d097ed732cf453f72aa36cf'
MAPPING_SHA = 'fd24a8525a4c0464fd1fae130763cd736d326d6a687dcb7d48bda40d93e83c00'
MAPPING_URL = 'https://raw.githubusercontent.com/DSAIL-Memory/EvoMemBench/3726cf1894d2b16698b9bc4a6186db91ad93849d/Cross-Episode-Execution/Embodied-AI/CROSSEP-EMB/agentenv-alfworld/configs/mappings_test.json'


def fetch(url, path, expected=None):
    path=Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True,exist_ok=True)
        partial=path.with_suffix(path.suffix+'.partial')
        with urllib.request.urlopen(url,timeout=120) as response, partial.open('wb') as output:
            while chunk:=response.read(2**20): output.write(chunk)
        if expected and file_hash(partial)!=expected:
            raise ValueError('Downloaded content hash mismatch: '+str(path))
        partial.replace(path)
    if expected and file_hash(path)!=expected: raise ValueError('Existing content hash mismatch: '+str(path))
    return path


def alma():
    from scripts.alma_feasibility import PIN, VENDOR, pinned
    if not VENDOR.exists():
        VENDOR.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run(['git','clone','https://github.com/zksha/alma.git',str(VENDOR)],check=True)
        subprocess.run(['git','-C',str(VENDOR),'checkout','--detach',PIN],check=True)
    pinned()
    print('Verified pinned ALMA checkout:',VENDOR)


def alfworld():
    from scripts.alma_feasibility import download_metadata, META
    alma()
    games=fetch('https://github.com/alfworld/alfworld/releases/download/0.4.2/alfworld-tw-0.4.2.zip',
                LEGACY/'downloads/alfworld-tw-0.4.2.zip',GAME_SHA)
    download_metadata()
    if not META.is_file(): raise FileNotFoundError(META)
    directory=DATA/'alfworld_panel'
    mapping=fetch(MAPPING_URL,directory/'crossep_emb_mappings_test.json',MAPPING_SHA)
    entries=json.loads(mapping.read_bytes());items=[]
    with zipfile.ZipFile(games) as archive:
        for row in entries:
            member=f"json_2.1.1/valid_train/{row['task_type']}/{row['task_id']}/game.tw-pddl"
            raw=archive.read(member)
            items.append(dict(item_id=row['item_id'],member=member,group=row['task_type'],
                family=row['task_type'].split('-',1)[0],game_sha256=hashlib.sha256(raw).hexdigest()))
    if len(items)!=200 or len({r['item_id'] for r in items})!=200: raise ValueError('Unexpected public panel')
    write(directory/'crossep_emb_manifest_audit.json',dict(source=MAPPING_URL,source_sha256=MAPPING_SHA,
        archive=str(games.resolve()),archive_sha256=GAME_SHA,panel_size=200,items=items,
        families=dict(Counter(r['family'] for r in items)),scope='public metadata only; no model outcomes'))
    print('Prepared the official 200-item mapping; no tasks executed.')


def apps():
    from benchmarks.math_apps import apps_prepare as prep, apps_identities as identity
    prep.HERE.mkdir(parents=True,exist_ok=True);prep.ARCHIVE.mkdir(exist_ok=True,mode=0o700)
    fetch('https://raw.githubusercontent.com/hendrycks/apps/362aedc3c71cd7d9bd2fc96a6c80e11dbc38c7a5/eval/testing_util.py',
          prep.ARCHIVE/'official_testing_util.py','9a4e58ff2634ef606c42597457c0733910862e0588ea750d901265bbfe65d36f')
    for split in prep.FILES: prep.fetch(split)
    prep.compose()
    rows,_=identity.index()
    fetch('https://people.eecs.berkeley.edu/~hendrycks/APPS.tar.gz',prep.ARCHIVE/'author-APPS.tar.gz',
          '6ef8e98ecca10b0159df0da4b524ecc1ca782a3b9473c57fc547ebccbbc2d0ca')
    identity.author_check(rows)
    print('Prepared APPS data and author identity verification; no solutions executed.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('asset',choices=('alma','alfworld','math','apps'))
    args=parser.parse_args()
    if args.asset=='alma': alma()
    elif args.asset=='alfworld': alfworld()
    elif args.asset=='apps': apps()
    else:
        from benchmarks.math_apps.math_tasks import prepare
        from benchmarks.math_apps.paths import ASSETS
        ASSETS.mkdir(parents=True,exist_ok=True)
        prepare(); print('Prepared eligible MATH rows; no model calls.')

if __name__=='__main__': main()
