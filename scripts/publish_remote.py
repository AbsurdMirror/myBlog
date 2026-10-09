"""Submit the complete approved source and deploy to existing GitHub Pages.

Use a normal GitHub authentication context (GH_TOKEN/GITHUB_TOKEN or gh auth).
Secrets are never written to disk, URLs, logs, or generated manifests.
Default is a local dry-run. --execute performs authorized repository changes.
A real publication is declared successful only after source and site SHA checks.
"""
from __future__ import annotations
import argparse,base64,hashlib,json,os,subprocess,sys,time
from pathlib import Path
from urllib.error import HTTPError,URLError
from urllib.parse import quote
from urllib.request import Request,urlopen
ROOT=Path(__file__).resolve().parents[1]
REPO='AbsurdMirror/myBlog'
API='https://api.github.com/repos/'+REPO
BRANCH='work/deepseek-v41-layer0'
LAYER='docs/models/deepseek-v4.1-flash/02-layer0.md'
TOKEN=''

def git_sha(data:bytes)->str:
    return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()

def request(method:str,path:str,data:dict|None=None,allow_missing:bool=False):
    headers={'Accept':'application/vnd.github+json','X-GitHub-Api-Version':'2022-11-28','User-Agent':'myBlog-publisher'}
    if TOKEN: headers['Authorization']='Bearer '+TOKEN
    body=None if data is None else json.dumps(data,ensure_ascii=False).encode()
    if body is not None:headers['Content-Type']='application/json'
    req=Request(API+path,data=body,headers=headers,method=method)
    try:
        with urlopen(req,timeout=90) as r:
            raw=r.read();return json.loads(raw) if raw else None
    except HTTPError as e:
        if e.code==404 and allow_missing:return None
        msg=e.read().decode(errors='replace')[:1600]
        raise RuntimeError(f'GitHub {method} {path}: HTTP {e.code}: {msg}') from None

def public_bytes(url:str)->bytes:
    with urlopen(Request(url,headers={'User-Agent':'myBlog-publication-check','Cache-Control':'no-cache'}),timeout=90) as r:return r.read()

def head(branch:str)->str:
    return request('GET','/git/ref/heads/'+quote(branch,safe='/'))['object']['sha']

def tree_at(sha:str)->dict:
    commit=request('GET','/git/commits/'+sha)
    tree=request('GET','/git/trees/'+commit['tree']['sha']+'?recursive=1')
    if tree.get('truncated'):raise RuntimeError('Refusing to modify a truncated tree')
    return {'base_tree':commit['tree']['sha'],'files':{x['path']:x for x in tree['tree']}}

def commit_files(branch:str,files:dict[str,bytes],message:str,remove:tuple[str,...]=())->str:
    parent=head(branch);base=tree_at(parent);entries=[]
    for path,data in sorted(files.items()):
        sha=git_sha(data)
        if base['files'].get(path,{}).get('sha')==sha:continue
        blob=request('POST','/git/blobs',{'content':base64.b64encode(data).decode('ascii'),'encoding':'base64'})
        if blob['sha']!=sha:raise RuntimeError('Blob mismatch: '+path)
        entries.append({'path':path,'mode':'100644','type':'blob','sha':sha})
        print('Prepared',path,len(data),'bytes',flush=True)
    for path in remove:
        if path in base['files']:entries.append({'path':path,'mode':'100644','type':'blob','sha':None})
    if not entries:return parent
    tree=request('POST','/git/trees',{'base_tree':base['base_tree'],'tree':entries})
    commit=request('POST','/git/commits',{'message':message,'tree':tree['sha'],'parents':[parent]})
    if head(branch)!=parent:raise RuntimeError('Branch moved during upload; no ref overwritten')
    request('PATCH','/git/refs/heads/'+quote(branch,safe='/'),{'sha':commit['sha'],'force':False})
    if head(branch)!=commit['sha']:raise RuntimeError('Ref read-back failed')
    return commit['sha']

def source_files()->dict[str,bytes]:
    out={}
    for base in ('docs','progress','scripts','theme','.github'):
        for p in (ROOT/base).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix not in {'.pyc'}:
                out[p.relative_to(ROOT).as_posix()]=p.read_bytes()
    for name in ('README.md','AUTHORING.md','.gitignore','publication.json','mkdocs.yml','requirements.txt'):
        p=ROOT/name
        if p.exists():out[name]=p.read_bytes()
    return out

def verify_originals(ref:str,manifest:dict)->None:
    remote=tree_at(ref)['files']
    for f in manifest['figures']:
        path='docs/models/deepseek-v4.1-flash/assets/'+f['file']
        expected=(ROOT/path).read_bytes()
        if remote.get(path,{}).get('sha')!=git_sha(expected):raise RuntimeError('Missing/changed tree entry '+path)
        body=public_bytes(f'https://raw.githubusercontent.com/{REPO}/{ref}/{path}')
        if hashlib.sha256(body).hexdigest()!=f['sha256']:raise RuntimeError('Remote image hash mismatch '+path)

def main()->None:
    global TOKEN
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--execute',action='store_true')
    ap.add_argument('--poll-seconds',type=int,default=600)
    args=ap.parse_args()
    subprocess.run([sys.executable,str(ROOT/'scripts/check.py'),'--published-only'],check=True)
    subprocess.run([sys.executable,str(ROOT/'scripts/check_site.py')],check=True)
    manifest=json.loads((ROOT/'docs/models/deepseek-v4.1-flash/assets/figures.json').read_text(encoding='utf-8'))
    source=source_files()
    print('Ready:',len(source),'source files;',len(manifest['figures']),'original PNGs; full static site')
    if not args.execute:
        print('DRY RUN: no repository or Pages changes performed');return
    TOKEN=os.environ.get('GH_TOKEN') or os.environ.get('GITHUB_TOKEN') or ''
    if not TOKEN:
        try:TOKEN=subprocess.run(['gh','auth','token'],capture_output=True,text=True,check=True).stdout.strip()
        except (FileNotFoundError,subprocess.SubprocessError):pass
    if not TOKEN:raise RuntimeError('No authenticated native GitHub context is present. No remote write attempted.')
    settings=request('GET','')
    if settings.get('full_name')!=REPO:raise RuntimeError('Unexpected repository')
    expected=json.loads((ROOT/'progress/remote-base.json').read_text(encoding='utf-8'))['expected_source_head']
    current=head(BRANCH)
    if current!=expected:raise RuntimeError('Source branch moved; fetch and reconcile before publishing: '+current)
    # The original archive stays in the base tree. Only approved source paths change.
    uploaded=commit_files(BRANCH,source,'docs: add original figures and verified static publication pipeline',
                          ('.github/workflows/asset-transfer.yml','scripts/fixtures/binary-upload-probe.png'))
    verify_originals(uploaded,manifest)
    # A normal merge preserves both histories; conflicts stop publication.
    request('POST','/merges',{'base':'main','head':uploaded,'commit_message':'Merge approved Layer 0 documentation and original figures'})
    main_sha=head('main');verify_originals(main_sha,manifest)
    # Use the already configured branch-based site for this initial deployment.
    # No protection rules, approval rules, or existing domain settings are removed.
    pages=request('GET','/pages')
    if pages.get('build_type') not in (None,'legacy') or pages.get('source',{}).get('branch')!='gh-pages' or pages.get('source',{}).get('path')!='/':
        raise RuntimeError('Source is safely merged. Pages configuration differs from expected gh-pages/root; inspect before updating.')
    public={p.relative_to(ROOT/'site').as_posix():p.read_bytes() for p in (ROOT/'site').rglob('*') if p.is_file()}
    # base_tree preserves existing legacy routes/resources; new site overrides index.
    deploy_sha=commit_files('gh-pages',public,'site: publish Layer 0 with original figures')
    site_url=pages.get('html_url') or 'https://absurdmirror.github.io/myBlog/'
    page_url=site_url.rstrip('/')+'/models/deepseek-v4.1-flash/02-layer0.html'
    deadline=time.monotonic()+args.poll_seconds;last_error='Waiting for Pages'
    while time.monotonic()<deadline:
        try:
            page=public_bytes(page_url).decode('utf-8')
            if 'Layer 0' not in page or 'hc_post' not in page:raise RuntimeError('Old/unexpected page')
            for f in manifest['figures']:
                data=public_bytes(site_url.rstrip('/')+'/models/deepseek-v4.1-flash/assets/'+f['file'])
                if hashlib.sha256(data).hexdigest()!=f['sha256']:raise RuntimeError('Pages image checksum mismatch: '+f['file'])
            result={'deployed':True,'source_commit':main_sha,'pages_commit':deploy_sha,'page_url':page_url,
                    'markdown_url':f'https://github.com/{REPO}/blob/main/{LAYER}',
                    'image_urls':[f'https://github.com/{REPO}/blob/main/docs/models/deepseek-v4.1-flash/assets/'+f['file'] for f in manifest['figures']]}
            (ROOT/'publication-result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
            print(json.dumps(result,ensure_ascii=False,indent=2));return
        except (HTTPError,URLError,RuntimeError) as e:last_error=str(e)
        time.sleep(15)
    raise RuntimeError('Source committed; Pages not verified within timeout: '+last_error)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        print(str(exc),file=sys.stderr)
        sys.exit(1)
