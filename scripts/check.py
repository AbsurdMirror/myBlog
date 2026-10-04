"""Check source metadata, local links, figure hashes, and arithmetic fixtures."""
from __future__ import annotations
import argparse, hashlib, json, re
from pathlib import Path
from render import load_markdown
ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--published-only',action='store_true')
    args=parser.parse_args()
    mode='published' if args.published_only else 'preview'
    docs=ROOT/'docs'
    pages=json.loads((ROOT/'publication.json').read_text())[mode]
    errors=[]
    for rel in pages:
        p=docs/rel
        if not p.is_file(): errors.append('Missing page: '+rel); continue
        meta,text=load_markdown(p)
        if meta.get('status') not in {'draft','review','published'}: errors.append('Missing status: '+rel)
        if args.published_only and meta.get('status')!='published': errors.append('Draft in published list: '+rel)
        if text.count('$$')%2: errors.append('Unpaired display math: '+rel)
        for target in re.findall(r'\]\(([^\s)]+)',text):
            if re.match(r'(?:[a-z]+:|#)',target): continue
            clean=target.split('#')[0]
            dest=(p.parent/clean).resolve()
            if not dest.is_relative_to(docs.resolve()) or not dest.exists(): errors.append('Broken local reference: '+rel+' -> '+clean)
            if args.published_only and clean.endswith('.md') and dest.exists():
                if str(dest.relative_to(docs.resolve())) not in pages: errors.append('Published page links to unstaged page: '+clean)
    if True:
        for manifest in docs.rglob('figures.json'):
            for row in json.loads(manifest.read_text())['figures']:
                path=manifest.parent/row['file']
                if not path.is_file(): errors.append('Missing original PNG: '+str(path.relative_to(ROOT))); continue
                if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']: errors.append('Changed original PNG: '+row['file'])
    # Derived scalar-count tests, not a claim of model-level correctness.
    mhc=24*20480+24+3
    swa=5120*1280+1280+1280*32768+5120*512+512+64+8*4096*1024+8192*5120
    moe=385*3*5120*2304+384*5120+2*384
    assert (mhc,swa,moe)==(491547,126617408,13626901248)
    assert 2*mhc+2*5120+swa+moe==13754511990
    if errors: raise SystemExit('\n'.join(errors))
    print(f'PASS: {len(pages)} pages, relative resources, original hashes, arithmetic ({mode}).')

if __name__=='__main__': main()
