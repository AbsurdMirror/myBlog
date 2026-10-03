"""Common Markdown -> HTML5/MathML renderer; no network or font bundling."""
from __future__ import annotations
import re, subprocess
from pathlib import Path
import yaml

def load_markdown(path: Path) -> tuple[dict, str]:
    text = path.read_text(encoding='utf-8')
    match = re.match(r'\A---\s*\n(.*?)\n---\s*\n', text, re.S)
    if not match:
        return {}, text
    return yaml.safe_load(match.group(1)) or {}, text[match.end():]

def render(text: str) -> str:
    try:
        result = subprocess.run(
            ['pandoc', '--from=markdown+tex_math_dollars-implicit_figures',
             '--to=html5', '--mathml', '--no-highlight', '--wrap=none'],
            input=text, text=True, capture_output=True, check=True, timeout=60)
    except FileNotFoundError as exc:
        raise RuntimeError('Pandoc is required. Install Pandoc before rendering.') from exc
    if result.stderr.strip():
        raise RuntimeError('Pandoc reported a rendering problem: ' + result.stderr)
    def convert_link(match: re.Match) -> str:
        href = match.group(1)
        if not re.match(r'\w+://', href):
            href = re.sub(r'\.md(?=#|$)', '.html', href)
        return 'href="' + href + '"'
    return re.sub(r'href="([^"]+)"', convert_link, result.stdout)
