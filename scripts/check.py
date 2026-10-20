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
    # Derived scalar-count tests; source correspondence is documented in the article.
    mhc=24*20480+24+3
    mhc_scalar_ops=1244901
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
    swa_fixed_ops=(1280*(5120+5119)+(4*1280+1)
                   +32768*(1280+1279)+512*(5120+5119)+(4*512+1)
                   +(64*32+32+64*32)*6
                   +8*1024*(4096+4095)+5120*(8192+8191)
                   +64-64*512)
    swa_ops_per_kv=64*(512+511+1)+2*64+64*512*2
    layer_reference_ops=2*mhc_scalar_ops+2*rmsnorm+swa_fixed_ops+moe_scalar_ops
    layer_optimized_ops=layer_reference_ops-5120*(4+3)
    mhc_parameter_bytes=2*(24*20480+24+3)*4
    layer_parameter_bytes=(mhc_parameter_bytes+2*rmsnorm_param_bytes
                           +160264704+moe_parameter_bytes)
    engram_rows=384006168
    engram_embed_params=engram_rows*256
    engram_wkv_params=25600*6144
    engram_qk_params=2*4*5120
    engram_params=engram_embed_params+engram_wkv_params+engram_qk_params
    engram_embed_bytes=engram_embed_params+engram_rows*(256//32)
    engram_wkv_bytes=engram_wkv_params+(25600//32)*(6144//32)
    engram_qk_bytes=engram_qk_params*2
    engram_parameter_bytes=engram_embed_bytes+engram_wkv_bytes+engram_qk_bytes
    engram_wkv_ops=25600*(6144+6143)
    engram_ops_per_token=(engram_wkv_ops+81928+61448+8+40960+4+24)
    layer1_params=2*mhc+2*5120+swa+moe+engram_params
    layer1_parameter_bytes=layer_parameter_bytes+engram_parameter_bytes
    layer1_reference_ops=layer_reference_ops+engram_ops_per_token
    engram_hash_shared_fixed=129280*8+2*4*8+2*3*8*8+2*24*8
    assert (rmsnorm,2*rmsnorm_params,rmsnorm_param_bytes)==(20481,10240,10240)
    assert (mhc,swa,moe)==(491547,126617408,13626901248)
    assert (moe_routed_expert,moe_shared_expert,moe_scalar_ops)==(70780672,70778368,499425676)
    assert (moe_routed_matrix_bytes,moe_shared_matrix_bytes,moe_parameter_bytes)==(6266880,11808000,7258804992)
    assert (swa_fixed_ops,swa_ops_per_kv)==(253182466,131200)
    assert (layer_reference_ops,layer_optimized_ops)==(755138906,755103066)
    assert (mhc_parameter_bytes,layer_parameter_bytes)==(3932376,7423022552)
    assert 2*(4*5120*2+4*4)==81952
    assert 2*mhc+2*5120+swa+moe==13754511990
    assert (engram_params,engram_parameter_bytes)==(98462906368,101535150272)
    assert (engram_wkv_ops,engram_ops_per_token)==(314547200,314731572)
    assert (layer1_params,layer1_parameter_bytes)==(112217418358,108958172824)
    assert layer1_reference_ops==1069870478
    assert engram_hash_shared_fixed==1035072
    if errors: raise SystemExit('\n'.join(errors))
    print(f'PASS: {len(pages)} pages, relative resources, original hashes, arithmetic ({mode}).')

if __name__=='__main__': main()
