"""Produce a self-contained HTML review copy from exactly one Markdown source."""
from __future__ import annotations
import argparse, base64, html, re
from pathlib import Path
from render import load_markdown, render

ROOT = Path(__file__).resolve().parents[1]

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='docs/models/deepseek-v4.1-flash/02-layer0.md')
    parser.add_argument('--output', default='.preview/layer0.html')
    args = parser.parse_args()
    source = (ROOT / args.source).resolve()
    if not source.is_relative_to(ROOT):
        raise SystemExit('Source must be inside repository.')
    meta, md = load_markdown(source)
    content = render(md)
    def embed(match: re.Match) -> str:
        url = html.unescape(match.group(1))
        if re.match(r'\w+:', url):
            raise ValueError('Offline preview requires local image assets: ' + url)
        path = (source.parent / url).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise ValueError('Missing or unsafe image path: ' + url)
        mime = {'.png':'image/png', '.jpg':'image/jpeg', '.jpeg':'image/jpeg', '.svg':'image/svg+xml'}.get(path.suffix)
        if not mime:
            raise ValueError('Unsupported image format: ' + str(path))
        return 'src="data:' + mime + ';base64,' + base64.b64encode(path.read_bytes()).decode('ascii') + '"'
    content = re.sub(r'src="([^"]+)"', embed, content)
    css = (ROOT / 'theme/site.css').read_text(encoding='utf-8')
    js = (ROOT / 'theme/viewer.js').read_text(encoding='utf-8')
    title = html.escape(meta.get('title', source.stem))
    page = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="light"><title>'''+title+'''</title><style>'''+css+'''</style></head><body>
    <header class="topbar"><span class="brand">图解技术笔记</span><span class="topmeta">MODEL NOTES · 审阅版</span></header>
    <aside class="sidebar"><p class="series-label">模型解析 / 02</p><strong>DeepSeek V4.1</strong><p class="sidebar-note">Layer 0 · 数学结构</p><nav id="page-toc" aria-label="本页目录"></nav><div class="side-foot">九张原图 · 离线可读<br>点击图片进入放大查看</div></aside>
    <main><div class="edition"><span>LAYER 0 / EXPLAINED</span><span class="badge">DRAFT · 待确认</span></div><article class="doc">'''+content+'''</article>
    <footer>由同一份 Markdown 生成 · 公式为 MathML · 原始配图未改动<br>2026-10-03 / 不代表已发布</footer></main><script>'''+js+'''</script></body></html>'''
    output = Path(args.output)
    if not output.is_absolute(): output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding='utf-8')
    print(f'Preview: {output} ({output.stat().st_size:,} bytes)')

if __name__ == '__main__': main()
