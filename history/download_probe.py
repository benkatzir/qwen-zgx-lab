import json, time, urllib.request
j=json.load(urllib.request.urlopen('https://pypi.org/pypi/vllm/0.28.0/json',timeout=15))
url=next(x['url'] for x in j['urls'] if 'aarch64' in x['filename'])
t=time.monotonic()
try:
 r=urllib.request.urlopen(urllib.request.Request(url,headers={'Range':'bytes=0-2097151'}),timeout=15)
 data=r.read(2097152); dt=time.monotonic()-t
 print(json.dumps({'bytes':len(data),'seconds':dt,'MBps':len(data)/dt/1e6,'url_host':url.split('/')[2]}))
except Exception as e: print(type(e).__name__,str(e))
