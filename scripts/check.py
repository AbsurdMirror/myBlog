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
    pages=json.loads((ROOT/'publication.json').read_text(encoding='utf-8'))[mode]
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
                if dest.relative_to(docs.resolve()).as_posix() not in pages: errors.append('Published page links to unstaged page: '+clean)
    if True:
        for manifest in docs.rglob('figures.json'):
            for row in json.loads(manifest.read_text(encoding='utf-8'))['figures']:
                path=manifest.parent/row['file']
                if not path.is_file(): errors.append('Missing original PNG: '+str(path.relative_to(ROOT))); continue
                if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']: errors.append('Changed original PNG: '+row['file'])
    # Derived scalar-count tests, not a claim of model-level correctness.
    mhc=24*20480+24+3
    rmsnorm=5120+(5120-1)+1+1+5120+5120
    rmsnorm_params=5120
    rmsnorm_param_bytes=rmsnorm_params*2
    swa=5120*1280+1280+1280*32768+5120*512+512+64+8*4096*1024+8192*5120
    moe=385*3*5120*2304+384*5120+2*384
    moe_router_projection=384*(5120+5119)
    moe_routed_expert=2*2304*(5120+5119)+3*2304+2304+2304+5120*(2304+2303)
    moe_shared_expert=moe_routed_expert-2304
    moe_scalar_ops=(moe_router_projection+384+384+(5+1+6)
                    +6*moe_routed_expert+moe_shared_expert+6*5120)
    moe_routed_matrix_bytes=5120*2304//2+5120*2304//32
    moe_shared_matrix_bytes=5120*2304+(5120//32)*(2304//32)
    moe_parameter_bytes=(384*3*moe_routed_matrix_bytes+3*moe_shared_matrix_bytes
                         +384*5120*2+2*384*4)
    assert (rmsnorm,2*rmsnorm_params,rmsnorm_param_bytes)==(20481,10240,10240)
    assert (mhc,swa,moe)==(491547,126617408,13626901248)
    assert (moe_routed_expert,moe_shared_expert,moe_scalar_ops)==(70780672,70778368,499425676)
    assert (moe_routed_matrix_bytes,moe_shared_matrix_bytes,moe_parameter_bytes)==(6266880,11808000,7258804992)
    assert 2*mhc+2*5120+swa+moe==13754511990
    if errors: raise SystemExit('\n'.join(errors))
    print(f'PASS: {len(pages)} pages, relative resources, original hashes, arithmetic ({mode}).')

if __name__=='__main__': main()
