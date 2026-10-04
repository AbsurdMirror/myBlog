"""Build the same Markdown/theme into a complete static site without MkDocs.

Requires Python, PyYAML and Pandoc. No CDN, network fetch, bundled font, or
JavaScript math runtime is needed. This is also the canonical CI build command.
"""
from __future__ import annotations
import argparse
import hashlib
import html
import json
import posixpath
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlsplit
from render import load_markdown, render

ROOT = Path(__file__).resolve().parents[1]
REPO = 'https://github.com/AbsurdMirror/myBlog'

def local_link(current: str, target: str) -> str:
    return posixpath.relpath(target, posixpath.dirname(current) or '.')

def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--preview',action='store_true')
    ap.add_argument('--output',default='site')
    ap.add_argument('--require-legacy',action='store_true')
    args=ap.parse_args()
    output=(ROOT/args.output).resolve()
    # Never delete sources or write outside the repository by accident.
    if not output.is_relative_to(ROOT) or output in {ROOT,ROOT/'docs',ROOT/'archive',ROOT/'scripts',ROOT/'theme'}:
        raise SystemExit('Unsafe output directory')
    subprocess.run(['python',str(ROOT/'scripts/check.py')]+([] if args.preview else ['--published-only']),check=True)
    mode='preview' if args.preview else 'published'
    pages=json.loads((ROOT/'publication.json').read_text())[mode]
    page_info=[]
    for rel in pages:
        src=ROOT/'docs'/rel
        meta,markdown=load_markdown(src)
        page_info.append((rel,meta,markdown))
    old=ROOT/'archive/hexo-site'
    if args.require_legacy and not old.is_dir():
        raise SystemExit('Original archive/hexo-site is required for publication')
    if output.exists(): shutil.rmtree(output)
    output.mkdir(parents=True)
    legacy_count=0
    if old.is_dir():
        for source in old.rglob('*'):
            if source.is_file() and not source.is_symlink():
                dst=output/source.relative_to(old)
                dst.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(source,dst)
                legacy_count+=1
    (output/'_static').mkdir(exist_ok=True)
    for name in ['site.css','viewer.js']:
        shutil.copy2(ROOT/'theme'/name, output/'_static'/name)
    for rel,meta,markdown in page_info:
        dest_rel=str(Path(rel).with_suffix('.html')).replace('\\','/')
        content=render(markdown)
        source=ROOT/'docs'/rel
        for match in re.finditer(r'!\[[^\]]*\]\(([^\s)]+)',markdown):
            image=match.group(1)
            if urlsplit(image).scheme: raise ValueError('Remote image not allowed: '+image)
            path=(source.parent/unquote(image)).resolve()
            if not path.is_relative_to((ROOT/'docs').resolve()) or not path.is_file():
                raise ValueError('Missing or unsafe image: '+image)
            dst=output/path.relative_to(ROOT/'docs')
            dst.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,dst)
        nav=''.join('<a href="'+html.escape(local_link(dest_rel,str(Path(p).with_suffix('.html')).replace('\\','/')))+'">'+html.escape(m.get('title',Path(p).stem))+'</a>' for p,m,_ in page_info)
        title=html.escape(meta.get('title',Path(rel).stem))
        badge='<span class="badge">PREVIEW · 预览</span>' if args.preview else ''
        doc_url=REPO+'/blob/main/docs/'+rel
        page=f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="light"><title>{title} · 图解技术笔记</title><link rel="stylesheet" href="{local_link(dest_rel,'_static/site.css')}"></head>
<body><header class="topbar"><a class="brand" href="{local_link(dest_rel,'index.html')}">图解技术笔记</a><span class="topmeta">MODEL NOTES</span></header>
<aside class="sidebar"><p class="series-label">模型 · 算法 · 系统</p><strong>图解技术笔记</strong><p class="sidebar-note">结构与数学，逐层展开</p><details class="site-links" open><summary>文档目录</summary>{nav}</details><nav id="page-toc" aria-label="本页目录"></nav><div class="side-foot"><a href="{doc_url}">本页 Markdown</a> · <a href="{REPO}/tree/main">源码仓库</a></div></aside>
<main><div class="edition"><span>TECHNICAL NOTES</span>{badge}</div><details class="mobile-nav"><summary>文档目录</summary>{nav}</details><article class="doc">{content}</article><footer>由同一份 Markdown 构建 · 原图可放大查看 · {html.escape(str(meta.get('updated','')))}<br><a href="{doc_url}">查看本页 Markdown</a></footer></main><script src="{local_link(dest_rel,'_static/viewer.js')}"></script></body></html>'''
        target=output/dest_rel
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(page,encoding='utf-8')
    # Safety: pages which were not approved are never copied from docs/.
    (output/'.nojekyll').touch()
    (output/'404.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>页面未找到</title><h1>页面未找到</h1><p><a href="/myBlog/">返回图解技术笔记</a></p></html>')
    try:
        commit=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,text=True,capture_output=True,check=True).stdout.strip()
    except (subprocess.SubprocessError,FileNotFoundError): commit=None
    files=[p for p in output.rglob('*') if p.is_file()]
    manifest={'format_version':1,'mode':mode,'source_commit':commit,
              'built_utc':datetime.now(timezone.utc).isoformat(),'pages':[str(Path(p).with_suffix('.html')) for p in pages],
              'legacy_file_count':legacy_count,'files':{str(p.relative_to(output)):{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in files}}
    (output/'build-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(f'Built {len(pages)} pages, {len(files)} resources, {legacy_count} legacy files -> {output}')

if __name__=='__main__': main()
