"""MATH Algebra and Number Theory with fixed source files and exact integer grading."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import os
import re
import sys
import urllib.request

from .paths import ASSETS
HERE = ASSETS
REGISTRATION = Path(__file__).with_name('math_pool_registration.json')
REGISTRATION_SHA = "4e25ccf361c39e29aaaacba0da38351c4a1a348b1a055ac0a816c5cf96e03d45"
PUBLIC = HERE / "math_public.json"
PRIVATE = HERE / "math_private"
GOLD = PRIVATE / "gold.json"
MANIFEST = HERE / "math_manifest.json"
BOX = re.compile(r"\\boxed\b")
INT = re.compile(r"[+-]?(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)")
FAMILIES = {"algebra": "Algebra", "number_theory": "Number Theory"}
FORMAT = r"Give the single final integer in the form \boxed{integer}."


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def freeze(path,data,mode=0o644):
    if path.exists():
        if path.read_bytes()!=data:
            raise ValueError("Changed frozen file: "+str(path))
        return
    handle=os.open(path,os.O_WRONLY | os.O_CREAT | os.O_EXCL,mode)
    with os.fdopen(handle,"wb") as stream:
        stream.write(data)


def encode(value):
    return (json.dumps(value,ensure_ascii=False,indent=2)+"\n").encode()


def boxed_integer(response, *, unique=False):
    """Parse a single decimal integer in the last explicit box; never evaluate math."""
    boxes=list(BOX.finditer(response))
    if not boxes or (unique and len(boxes)!=1):
        return None
    tail=response[boxes[-1].end():].lstrip()
    if not tail.startswith("{"):
        return None
    end=tail.find("}")
    if end<0:
        return None
    token=tail[1:end].strip()
    if not INT.fullmatch(token):
        return None
    digits=token.lstrip("+-").replace(",","").lstrip("0") or "0"
    return ("-" if token.startswith("-") and digits!="0" else "") + digits


def question_hash(text):
    return sha(" ".join(text.split()).encode())


def source_file(distribution,row):
    family,split=row["family"],row["split"]
    p=PRIVATE/"source"/(family+"_"+split+".parquet")
    url="https://huggingface.co/datasets/{}/resolve/{}/{}/{}-00000-of-00001.parquet".format(
        distribution["repository"],distribution["revision"],family,split)
    if p.exists():
        data=p.read_bytes()
    else:
        with urllib.request.urlopen(url,timeout=60) as response:
            data=response.read()
    if len(data)!=row["bytes"] or sha(data)!=row["sha256"]:
        raise ValueError("Unexpected published bytes for "+family+"/"+split)
    freeze(p,data,0o600)
    return p


def eligible(row,duplicate):
    if row["level"] not in {f"Level {n}" for n in range(1,6)}:
        return "unregistered_level",None
    prompt=row["problem"].casefold()
    if not prompt.strip() or any(tag in prompt for tag in ("[asy]","<img",r"\includegraphics")):
        return "non_text_prompt",None
    if duplicate:
        return "duplicated_question",None
    gold=boxed_integer(row["solution"],unique=True)
    if gold is None:
        return "non_scalar_integer_reference",None
    return "eligible",gold


def prepare():
    """Rebuild or authenticate the complete four-shard pool. No model requests or task sampling."""
    import pyarrow
    import pyarrow.parquet as pq
    if sha(REGISTRATION.read_bytes())!=REGISTRATION_SHA:
        raise ValueError("Changed MATH registration")
    registration=read(REGISTRATION)
    dist=registration["publish"]
    PRIVATE.mkdir(mode=0o700,exist_ok=True)
    (PRIVATE/"source").mkdir(mode=0o700,exist_ok=True)
    shards={}
    occurrences=Counter()
    for row in dist["shards"]:
        p=source_file(dist,row)
        data=pq.read_table(p,columns=["problem","level","type","solution"])
        if data.num_rows!=row["rows"]:
            raise ValueError("Wrong number of published rows")
        rows=data.to_pylist()
        if not all(r["type"]==FAMILIES[row["family"]] for r in rows):
            raise ValueError("Unexpected subject label")
        shards[row["family"],row["split"]]=rows
        occurrences.update(question_hash(r["problem"]) for r in rows)
    public=[]
    gold={}
    stats=[]
    for (family,split),rows in shards.items():
        counts=Counter()
        for index,row in enumerate(rows):
            key=question_hash(row["problem"])
            status,answer=eligible(row,occurrences[key]>1)
            level=row["level"]
            counts[(level,status)]+=1
            if status!="eligible":
                continue
            task_id=f"math/{family}/{split}/{index}"
            public.append(dict(id=task_id,group_id=task_id,family=family,split=split,
                               level=int(level.split()[1]),prompt=row["problem"]+"\n\n"+FORMAT))
            gold[task_id]=answer
        for level in sorted({r["level"] for r in rows}):
            by_status={status:n for (label,status),n in counts.items() if label==level}
            stats.append(dict(family=family,split=split,official_level=level,
                              published=sum(by_status.values()),eligible=by_status.get("eligible",0),
                              filters=by_status))
    p_wire,g_wire=encode(public),encode(gold)
    freeze(PUBLIC,p_wire)
    freeze(GOLD,g_wire,0o600)
    manifest=dict(schema=1,dataset="MATH",family_names=FAMILIES,
                  title=registration["benchmark"],upstream=registration["upstream"],distribution=dist,
                  preregistration_sha256=REGISTRATION_SHA,implementation_sha256=sha(Path(__file__).read_bytes()),
                  public_file=PUBLIC.name,public_sha256=sha(p_wire),private_gold_sha256=sha(g_wire),
                  source_format="parquet",prepare_pyarrow=pyarrow.__version__,prepare_python=sys.version.split()[0],
                  subset="Whole exact boxed decimal-integer, text-only, nonduplicate pool in two named subjects and official levels 1 through 5",
                  duplicate_policy="Remove every copy of whitespace-normalized duplicate prompts across all four raw official shards",
                  score="0 or 1 using the last explicit boxed integer; no rounding or model grading",
                  published_rows=sum(s["rows"] for s in dist["shards"]),eligible_rows=len(public),strata=stats)
    freeze(MANIFEST,encode(manifest))
    return manifest


def checked_payload(path,key):
    expected=read(MANIFEST)[key]
    payload=path.read_bytes()
    if sha(payload)!=expected:
        raise ValueError("Changed frozen payload: "+str(path))
    return json.loads(payload)


def load_public():
    """Return only public task, subject, split, level and neutral formatting instruction."""
    return checked_payload(PUBLIC,"public_sha256")


def grade(task_id,completion):
    """Host-side exact grade. Return no references, solutions or private file contents."""
    values=checked_payload(GOLD,"private_gold_sha256")
    if task_id not in values:
        raise KeyError("Unknown MATH item")
    if not isinstance(completion,str):
        return dict(score=None,status="missing_completion")
    value=boxed_integer(completion)
    if value is None:
        return dict(score=0,status="invalid_final_integer")
    correct=value==values[task_id]
    return dict(score=int(correct),status="correct" if correct else "incorrect")


if __name__=="__main__":
    summary=prepare()
    print(f"Frozen {summary['eligible_rows']} / {summary['published_rows']} eligible problems; "
          f"manifest SHA-256 {sha(MANIFEST.read_bytes())}")

