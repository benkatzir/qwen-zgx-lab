import inspect, json, importlib.metadata as md
import torch, flashinfer
info={'torch':torch.__version__,'cuda':torch.version.cuda,'cuda_available':torch.cuda.is_available()}
for pkg in ['vllm','flashinfer-python','transformers','compressed-tensors','nvidia-cutlass-dsl']:
    try: info[pkg]=md.version(pkg)
    except md.PackageNotFoundError: pass
if torch.cuda.is_available():
    info.update(device=torch.cuda.get_device_name(),capability=torch.cuda.get_device_capability(),memory=torch.cuda.mem_get_info())
    a=torch.randn((2048,2048),device='cuda',dtype=torch.bfloat16); b=a@a; torch.cuda.synchronize()
    info['bf16_matmul_finite']=bool(torch.isfinite(b).all().item())
for name in ['BatchDecodeWithPagedKVCacheWrapper','BatchPrefillWithPagedKVCacheWrapper']:
    c=getattr(flashinfer,name)
    info[name]={method:str(inspect.signature(getattr(c,method))) for method in ['__init__','plan','run']}
print(json.dumps(info,indent=2))
