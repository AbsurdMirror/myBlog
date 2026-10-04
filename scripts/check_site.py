"""Validate a built static site: local URLs, fragments, original PNGs, MathML."""
from __future__ import annotations
import argparse,hashlib,json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote,urlsplit
ROOT=Path(__file__).resolve().parents[1]
class Links(HTMLParser):
    def __init__(self):
        super().__init__(); self.links=[]; self.ids=set(); self.math=0; self.images=[]
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if 'id'in a:self.ids.add(a['id'])
        if tag=='math':self.math+=1
        for key in ('href','src'):
            if key in a:self.links.append(a[key])
        if tag=='img': self.images.append(a.get('src',''))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--site',default='site');args=ap.parse_args()
    site=(ROOT/args.site).resolve(); manifest=json.loads((site/'build-manifest.json').read_text())
    errors=[];parsed={}
    for rel in manifest['pages']:
        path=site/rel;p=Links();p.feed(path.read_text());parsed[path.resolve()]=p
    for path,p in parsed.items():
        for link in p.links:
            u=urlsplit(link)
            if u.scheme or u.netloc:continue
            clean=unquote(u.path)
            if clean.startswith('/myBlog/'):
                dest=site/clean[len('/myBlog/'):]
            elif clean.startswith('/'):
                errors.append('Unexpected root URL '+link);continue
            else:dest=(path.parent/clean) if clean else path
            if dest.is_dir():dest=dest/'index.html'
            dest=dest.resolve()
            if not dest.is_relative_to(site) or not dest.exists():errors.append(f'{path.name}: missing {link}');continue
            if u.fragment and dest in parsed and unquote(u.fragment) not in parsed[dest].ids:
                errors.append(f'{path.name}: missing anchor {link}')
    layer=site/'models/deepseek-v4.1-flash/02-layer0.html'
    fig_manifest=ROOT/'docs/models/deepseek-v4.1-flash/assets/figures.json'
    if layer.exists():
        p=parsed[layer.resolve()]
        if len(p.images)!=5:errors.append('Layer 0 must include five original diagrams')
        if p.math<60:errors.append('Missing MathML formulas')
        for f in json.loads(fig_manifest.read_text())['figures']:
            path=layer.parent/'assets'/f['file']
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=f['sha256']:
                errors.append('Missing/changed deployed image '+f['file'])
    if errors:raise SystemExit('\n'.join(errors))
    print(f'PASS: {len(parsed)} pages, local URLs and anchors, five SHA256-identical diagrams; '+str(sum(p.math for p in parsed.values()))+' MathML nodes')
if __name__=='__main__':main()
