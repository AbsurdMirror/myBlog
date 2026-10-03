"""Use the same MathML renderer for MkDocs and standalone previews."""
from pathlib import Path
import importlib.util
_spec = importlib.util.spec_from_file_location('doc_render', Path(__file__).with_name('render.py'))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

def on_config(config):
    root = Path(config.config_file_path).resolve().parent
    docs = root / '.build/docs'
    pages = sorted(docs.rglob('*.md'), key=lambda p: (p.name != 'index.md', str(p)))
    config['nav'] = [{_mod.load_markdown(p)[0].get('title', p.stem): str(p.relative_to(docs))} for p in pages]
    return config

def on_page_content(html, page, config, files):
    return _mod.render(page.markdown)

def on_post_build(config):
    # Preserve original legacy URLs/assets by copying the whole original tree.
    # Existing new-site output must win at root index.html; legacy resources are disjoint.
    import shutil
    root = Path(config.config_file_path).resolve().parent
    old, out = root / 'archive/hexo-site', Path(config['site_dir'])
    if old.is_dir():
        for path in old.rglob('*'):
            if not path.is_file():
                continue
            rel = path.relative_to(old)
            target = out / rel
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
