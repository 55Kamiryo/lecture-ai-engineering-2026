"""Compare CPU/MPS backward for SiLU with a non-contiguous incoming gradient."""
import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--contiguous-grad',action='store_true')
args=parser.parse_args()
torch.set_num_threads(1)
torch.manual_seed(42)
results=[]
for channels in [4,16,128,6144]:
    # Qwen uses a depthwise causal convolution with a transposed input.
    x=torch.randn(1,263,channels).transpose(1,2)
    weight=torch.randn(channels,1,4)
    g=torch.randn(1,263,channels).transpose(1,2)
    runs={}
    for device in ['cpu','mps']:
        inp=x.to(device).detach().requires_grad_()
        out=F.silu(F.conv1d(inp,weight.to(device),padding=3,groups=channels)[:,:,:263])
        if args.contiguous_grad:
            out.register_hook(lambda grad:grad.contiguous())
        out.backward(g.to(device))
        runs[device]=(out.detach().cpu(),inp.grad.cpu())
    output_diff=((runs['cpu'][0]-runs['mps'][0]).norm()/runs['cpu'][0].norm()).item()
    grad_diff=((runs['cpu'][1]-runs['mps'][1]).norm()/runs['cpu'][1].norm()).item()
    row={'channels':channels,'output_relative_error':output_diff,'gradient_relative_error':grad_diff,'cpu_grad_norm':runs['cpu'][1].norm().item(),'mps_grad_norm':runs['mps'][1].norm().item()}
    results.append(row)
    print(row,flush=True)
work=Path(__file__).resolve().parent.parent/'.venv/lecture02-diagnostics'
work.mkdir(parents=True,exist_ok=True)
name='silu_contiguous_gradient.json' if args.contiguous_grad else 'silu_strided_gradient.json'
(work/name).write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
