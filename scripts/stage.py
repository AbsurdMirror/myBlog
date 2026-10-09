"""Stage only explicitly selected pages and their referenced local assets."""
from __future__ import annotations
import argparse, json, re, shutil
from pathlib import Path
from render import load_markdown
ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preview', action='store_true')
    args = parser.parse_args()
    mode = 'preview' if args.preview else 'published'
    pages = json.loads((ROOT/'publication.json').read_text(encoding='utf-8'))[mode]
    docs, staged = ROOT/'docs', ROOT/'.build/docs'
    if staged.exists(): shutil.rmtree(staged)
    staged.mkdir(parents=True)
    for rel in pages:
        src = (docs/rel).resolve()
        if not src.is_relative_to(docs.resolve()): raise ValueError('Unsafe page path')
        meta, text = load_markdown(src)
        if not args.preview and meta.get('status') != 'published':
            raise ValueError('Non-published page in publication allowlist: ' + rel)
        targets = [src]
        for link in re.findall(r'!\[[^\]]*\]\(([^\s)]+)', text):
            asset = (src.parent/link).resolve()
            if not asset.is_relative_to(docs.resolve()) or not asset.is_file():
                raise ValueError('Missing or unsafe asset: ' + link)
            targets.append(asset)
        for item in targets:
            dest = staged/item.relative_to(docs.resolve())
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item,dest)
    print(f'Staged {len(pages)} pages ({mode}); only referenced assets copied.')

if __name__ == '__main__': main()
