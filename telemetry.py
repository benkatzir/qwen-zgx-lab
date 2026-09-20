import argparse, json, subprocess, time
from pathlib import Path

p=argparse.ArgumentParser(); p.add_argument('--out',required=True); p.add_argument('--seconds',type=int,default=14400); p.add_argument('--interval',type=float,default=2)
a=p.parse_args(); start=time.monotonic()
with open(a.out,'a',buffering=1) as f:
    while time.monotonic()-start<a.seconds:
        row={'unix_time':time.time()}
        row['memory_kib']={k.rstrip(':'):int(v.split()[0]) for k,v in (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines()) if k in ['MemTotal','MemAvailable','SwapTotal','SwapFree','Cached']}
        try:
            r=subprocess.run(['nvidia-smi','--query-gpu=temperature.gpu,power.draw,utilization.gpu,clocks.sm,clocks.mem,pstate','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
            row['gpu']=dict(zip(['temperature_c','power_w','utilization_pct','sm_clock_mhz','memory_clock_mhz','pstate'],r.stdout.strip().split(', ')))
            if r.returncode: row['gpu_error']=r.stderr.strip()
        except Exception as e: row['gpu_error']=str(e)
        f.write(json.dumps(row)+'\n'); time.sleep(a.interval)
