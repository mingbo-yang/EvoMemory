"""Fixed lexical ranking for Evolution Memory; no Task Memory mechanism lives here."""
from collections import Counter
import math
import re


def terms(s): return re.findall(r'[a-z0-9_]+|[\u4e00-\u9fff]', s.lower())

def rank(query, documents, mode='bm25'):
    if not documents: return []
    qs = set(terms(query)); ds = [Counter(terms(x)) for x in documents]
    lens = [sum(d.values()) for d in ds]; avg = sum(lens)/len(lens) or 1
    scores=[]
    for d,n in zip(ds,lens):
        if mode=='overlap': score=len(qs & d.keys()) / (len(qs | d.keys()) or 1)
        else:
            score=0.0
            for t in qs & d.keys():
                df=sum(t in z for z in ds)
                idf=math.log(1+(len(ds)-df+.5)/(df+.5)); tf=d[t]
                score+=idf*tf*2.2/(tf+1.2*(.25+.75*n/avg))
        scores.append(score)
    return sorted(range(len(ds)), key=lambda i:(-scores[i],i))
