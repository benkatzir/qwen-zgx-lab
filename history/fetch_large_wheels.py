"""Download exact PyPI wheels selected by pip, with bounded HTTP range workers.

Reads only the pip log, checks every wheel against PyPI's SHA-256, and writes
an install argument file. Does not install packages or change network settings.
"""
import concurrent.futures as cf
import hashlib, json, os, re, socket, time, urllib.request
from pathlib import Path
from packaging.utils import parse_wheel_filename

root=Path('/home/ben/qwen-lab'); out=root/'wheels'; out.mkdir(exist_ok=True)
orig=socket.getaddrinfo
socket.getaddrinfo=lambda h,p,family=0,type=0,proto=0,flags=0:orig(h,p,socket.AF_INET,type,proto,flags)
log=(root/'logs/runtime-pip-ipv4.log').read_text()
names=sorted(set(re.findall(r'(?:Downloading|Using cached) ([A-Za-z0-9_.+\-]+\.whl)(?:\.metadata)?',log)))
selected=[]
for name in names:
    pkg,version,_,_=parse_wheel_filename(name)
    with urllib.request.urlopen(f'https://pypi.org/pypi/{pkg}/{version}/json',timeout=30) as r: data=json.load(r)
    meta=next((x for x in data['urls'] if x['filename']==name),None)
    if meta and meta['size']>=60000000: selected.append(meta)
print('Selected:',[(x['filename'],round(x['size']/1e6)) for x in selected],flush=True)
(out/'manifest.json').write_text(json.dumps(selected,indent=2))

def wheel(meta):
    name=meta['filename']; dest=out/name; size=meta['size']; chunk=8*1024*1024
    if dest.exists() and hashlib.file_digest(open(dest,'rb'),'sha256').hexdigest()==meta['digests']['sha256']:
        print('CACHED',name,flush=True); return dest
    part=out/(name+'.partial')
    fd=os.open(part,os.O_RDWR|os.O_CREAT,0o644); os.ftruncate(fd,size)
    def section(start):
        end=min(size,start+chunk)-1
        for attempt in range(4):
            try:
                req=urllib.request.Request(meta['url'],headers={'Range':f'bytes={start}-{end}'})
                with urllib.request.urlopen(req,timeout=120) as r:
                    if r.status!=206: raise RuntimeError(f'Range not honored: {r.status}')
                    data=r.read(end-start+2)
                if len(data)!=end-start+1: raise RuntimeError('Unexpected range length')
                offset=start
                while data:
                    n=os.pwrite(fd,data,offset); offset+=n; data=data[n:]
                return
            except Exception:
                if attempt==3: raise
                time.sleep(2*(attempt+1))
    try:
        with cf.ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(section,range(0,size,chunk)))
    finally: os.close(fd)
    with part.open('rb') as f: actual=hashlib.file_digest(f,'sha256').hexdigest()
    if actual!=meta['digests']['sha256']: raise RuntimeError('SHA256 mismatch '+name)
    part.rename(dest); print('VERIFIED',name,flush=True); return dest

with cf.ThreadPoolExecutor(max_workers=2) as pool:
    files=list(pool.map(wheel,selected))
(out/'large-requirements.txt').write_text('vllm==0.28.0\n'+'\n'.join(str(f) for f in files)+'\n')
print('COMPLETE',flush=True)
