"""Download immutable APPS mirrors and index the two official splits, without executing solutions."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import sys
import time
import requests

from .paths import ASSETS
HERE=ASSETS
ARCHIVE=HERE/'apps_archive'
REV='21e74ddf8de1a21436da12e3e653065c5213e9d1'
FILES={'train':(107101272,'45e82ef22ed8e7c0c04d881a21b923e9dd233157896b0b8d5b3493e887499cae'),
       'test':(1292436853,'5b003a65ac40feb47dd5eaec267a767a6fc435bdcfa68ff715fe869f948e760c')}

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(2**20),b''):h.update(chunk)
    return h.hexdigest()

def fetch(split):
    size,expected=FILES[split]
    path=ARCHIVE/(split+'.jsonl')
    if path.is_file():
        if sha(path)!=expected:raise ValueError('Unexpected '+split+' hash')
        return
    spans=ARCHIVE/(split+'_parts');spans.mkdir(exist_ok=True)
    chunk=8*2**20
    def part(start):
        end=min(size,start+chunk)-1; name=spans/str(start)
        if name.exists() and name.stat().st_size==end-start+1:return
        for attempt in range(5):
            try:
                host='huggingface.co' if attempt%2==0 else 'hf-mirror.com'
                url=f'https://{host}/datasets/codeparrot/apps/resolve/{REV}/{split}.jsonl'
                res=requests.get(url,params={'part':f'{start}-{end}'},headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'},timeout=(15,120))
                if res.status_code!=206 or res.headers['Content-Range']!=f'bytes {start}-{end}/{size}' or len(res.content)!=end-start+1:
                    raise ValueError(f'Unexpected range response {res.status_code} {res.headers.get("Content-Range")}')
                temp=name.with_suffix('.tmp');temp.write_bytes(res.content);temp.replace(name);return
            except (requests.RequestException,KeyError,ValueError) as exc:
                if attempt==4:raise
                print(split,start,'retry',attempt+1,str(exc)[:120],flush=True);time.sleep(1+attempt)
    offsets=list(range(0,size,chunk))
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        for ith,_ in enumerate(ex.map(part,offsets)):
            if ith%10==0:print(split,ith+1,'/',len(offsets),flush=True)
    tmp=ARCHIVE/(split+'.incoming')
    with tmp.open('wb') as out:
        for start in offsets:
            out.write((spans/str(start)).read_bytes())
    if sha(tmp)!=expected or tmp.stat().st_size!=size:raise ValueError('Bad assembled LFS')
    tmp.replace(path)
    for start in offsets:(spans/str(start)).unlink()
    spans.rmdir()
    print('verified',split,size,expected,flush=True)

def compose():
    from collections import Counter
    import re
    if hasattr(sys,'set_int_max_str_digits'):sys.set_int_max_str_digits(0)
    metadata={};offsets={}; excluded={}
    temp=HERE/'apps_public.incoming'
    with temp.open('w') as public:
        for split,(wanted_size,expected) in FILES.items():
            path=ARCHIVE/(split+'.jsonl')
            if path.stat().st_size!=wanted_size or sha(path)!=expected:raise ValueError('Incomplete pinned original corpus')
            tally=Counter();excluded[split]=Counter();difficulties=Counter();total=0;declared_ids=set()
            with path.open('rb') as f:
                while True:
                    start=f.tell();line=f.readline()
                    if not line:break
                    raw=json.loads(line);total+=1
                    native=raw['id']
                    if type(native) is not int or native in declared_ids:raise ValueError('Invalid native row ID')
                    declared_ids.add(native);question=raw['question']; starter=raw['starter_code'] or ''
                    try:io=json.loads(raw['input_output'] or '{}')
                    except (ValueError,TypeError):excluded[split]['invalid_private_json']+=1;continue
                    if not isinstance(io,dict) or not isinstance(io.get('inputs'),list) or not io['inputs'] or not isinstance(io.get('outputs'),list) or len(io['inputs'])!=len(io['outputs']):
                        excluded[split]['absent_or_unaligned_original_cases']+=1;continue
                    name=io.get('fn_name')
                    family='call_based' if name else 'stdio'
                    def string_form(x):return isinstance(x,str) or isinstance(x,list) and all(isinstance(i,str) for i in x)
                    if name:
                        if not (isinstance(name,str) and name.isidentifier() and not name.startswith('_') and starter.strip() and
                                re.search(r'\b'+re.escape(name)+r'\s*\(',starter) and all(isinstance(case,list) for case in io['inputs'])):
                            excluded[split]['unmatched_public_signature_or_unsupported_call_shape']+=1;continue
                    elif not all(string_form(x) for a in ('inputs','outputs') for x in io[a]):
                        excluded[split]['non_text_stdio_cases']+=1;continue
                    tid=f'apps/{split}/{native:04d}'
                    prompt=question if not starter.strip() else question+'\n\nStarter code:\n```python\n'+starter+'\n```'
                    item=dict(id=tid,family=family,split=split,prompt=prompt,difficulty=raw['difficulty'],
                              tests_total=len(io['inputs']),native_id=native)
                    public.write(json.dumps(item,ensure_ascii=False,separators=(',',':'))+'\n')
                    offsets[tid]=dict(split=split,offset=start,bytes=len(line),sha256=hashlib.sha256(line).hexdigest())
                    tally[family]+=1;difficulties[raw['difficulty']]+=1
            if total!=5000:raise ValueError('Original split is incomplete')
            metadata[split]=dict(original_size_bytes=wanted_size,raw_sha256=expected,rows=total,eligible=dict(tally),
                                 excluded=dict(excluded[split]),eligible_difficulty=dict(difficulties))
    temp.replace(HERE/'apps_public.jsonl')
    (ARCHIVE/'offsets.json').write_text(json.dumps(offsets,sort_keys=True,separators=(',',':'))+'\n')
    manifest=dict(dataset='APPS',author_repo='https://github.com/hendrycks/apps',author_commit='362aedc3c71cd7d9bd2fc96a6c80e11dbc38c7a5',
        author_grading_module_sha256=sha(ARCHIVE/'official_testing_util.py'),dataset_repo='https://huggingface.co/datasets/codeparrot/apps',
        revision=REV,splits=metadata,public_sha256=sha(HERE/'apps_public.jsonl'),private_offsets_sha256=sha(ARCHIVE/'offsets.json'),
        labels='stdio / call_based are the native APPS interfaces. They do not label algorithm families.',
        eligibility='Aligned nonempty original tests. For call_based require a matching public starter signature and positional argument lists; for stdio require string or string-list cases.',
        scope='All 10000 original rows frozen. Actor sees original public question and original starter only. Cases and reference solutions stay in the private raw archive. No test subsampling.')
    (HERE/'apps_manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n')
    return manifest

if __name__=='__main__':
    ARCHIVE.mkdir(exist_ok=True,mode=0o700)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        list(ex.map(fetch,FILES))
    print(json.dumps(compose()['splits'],indent=2),flush=True)
