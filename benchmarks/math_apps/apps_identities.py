"""Offline source identity and original text joins; never imported by the Actor."""
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import re
import tarfile
import unicodedata
from urllib.parse import urlsplit,urlunsplit
from .apps_prepare import ARCHIVE,HERE,sha

def text_sha(text):
    return hashlib.sha256(text.encode()).hexdigest()

def normalized(text):
    return ' '.join(unicodedata.normalize('NFKC',text).split()).casefold()

def url_key(value):
    if not value:return ''
    try:
        u=urlsplit(value.strip()); host=(u.hostname or '').lower().removeprefix('www.')
    except ValueError:return ''
    if not host:return ''
    path=u.path.rstrip('/')
    if host in ('codeforces.com','codeforces.ru') or host.endswith('.codeforces.com'):
        m=re.fullmatch(r'/(?:problemset/problem/(\d+)/([a-z][a-z0-9]*)|contest/(\d+)/problem/([a-z][a-z0-9]*))',path,re.I)
        if m:
            cid,part=(m[1],m[2]) if m[1] else (m[3],m[4])
            return 'codeforces:'+cid+'/'+part.upper()
    return urlunsplit(('',host,path,u.query,'')).removeprefix('//')

def index():
    rows={};by=defaultdict(list);questions=defaultdict(list);nonempty=Counter()
    for split in ('train','test'):
        with (ARCHIVE/(split+'.jsonl')).open('rb') as stream:
            for line in stream:
                r=json.loads(line);k=f"apps/{split}/{r['id']:04d}"
                url=r['url'] or '';q=r['question'];key=url_key(url);qh=text_sha(normalized(q))
                rows[k]=dict(original_url=url,source_key=key,question_sha256=text_sha(q),normalized_question_sha256=qh)
                if key:by[key].append(k);nonempty[split]+=1
                questions[qh].append(k)
    src=ARCHIVE/'source_identities.json'
    src.write_text(json.dumps(rows,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n')
    def repeats(dic):
        return {k:sorted(v) for k,v in dic.items() if len({x.split('/')[1] for x in v})>1}
    report=dict(identity_index='apps_archive/source_identities.json',index_sha256=sha(src),
        count=len(rows),nonempty_source_keys=dict(nonempty),
        source_field='Original APPS url. Canonicalize Codeforces contest and problemset aliases; for other sites normalize host and trailing slash, preserve path/query.',
        text_identity='Original question excluding starter; exact SHA256 and NFKC whitespace collapse + casefold SHA256.',
        cross_split_same_url=repeats(by),cross_split_same_normalized_question=repeats(questions))
    path=ARCHIVE/'overlap.json'
    path.write_text(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
    return rows,report

def author_check(rows):
    counts=Counter();disagreements=[];seen=set()
    with tarfile.open(ARCHIVE/'author-APPS.tar.gz',mode='r|gz') as tar:
        for file in tar:
            m=re.fullmatch(r'(?:\./)?APPS/(train|test)/(\d+)/(metadata.json|question.txt)',file.name)
            if not m:continue
            split,no,field=m.groups();key=f'apps/{split}/{int(no):04d}'
            s=tar.extractfile(file).read().decode('utf-8')
            if field=='metadata.json':
                native=json.loads(s).get('url') or ''
                same=native==rows[key]['original_url']
            else:
                same=text_sha(s)==rows[key]['question_sha256']
            if not same:disagreements.append([key,field])
            if (key,field) in seen:raise ValueError('Duplicate archive field')
            seen.add((key,field));counts[split+'_'+field]+=1
    evidence=dict(source_url='https://people.eecs.berkeley.edu/~hendrycks/APPS.tar.gz',archive_sha256=sha(ARCHIVE/'author-APPS.tar.gz'),
        author_commit='362aedc3c71cd7d9bd2fc96a6c80e11dbc38c7a5',checked=dict(counts),
        index_sha256=sha(ARCHIVE/'source_identities.json'),diff_count=len(disagreements),diff=disagreements)
    if len(seen)!=20000:raise ValueError('Missing author archive index metadata')
    (ARCHIVE/'author_check.json').write_text(json.dumps(evidence,sort_keys=True,indent=2)+'\n')
    print('author',evidence['checked'],evidence['diff_count'],flush=True)

if __name__=='__main__':
    rows,report=index()
    print('index',report['index_sha256'],report['nonempty_source_keys'],
          'duplicate keys',len(report['cross_split_same_url']),len(report['cross_split_same_normalized_question']),flush=True)
    author_check(rows)
