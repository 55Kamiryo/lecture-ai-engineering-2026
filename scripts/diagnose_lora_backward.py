"""Compare the same full model gradient on CPU and MPS without weight updates."""
import argparse
import json
import time
from pathlib import Path

import torch
import transformers

p = argparse.ArgumentParser()
p.add_argument('--device', choices=['cpu', 'mps'], required=True)
p.add_argument('--chunk-size', type=int)
p.add_argument('--cpu-op', choices=['chunk', 'conv', 'rms'])
p.add_argument('--mps-silu-grad-contiguous', action='store_true')
args = p.parse_args()
if args.mps_silu_grad_contiguous:
    # Diagnostic only: this resolves the isolated SiLU case but does not fix
    # the full model's backward result on the tested environment.
    import torch.nn.functional as F
    original_silu = F.silu
    def silu(input, inplace=False):
        output = original_silu(input, inplace=inplace)
        if output.device.type == 'mps' and output.requires_grad:
            output.register_hook(lambda gradient: gradient.contiguous())
        return output
    F.silu = silu
torch.set_num_threads(4)
root = Path(__file__).resolve().parent.parent
work = root / '.venv/lecture02-diagnostics'
work.mkdir(parents=True, exist_ok=True)
notebook = json.loads((root / 'notebooks/lora_format_tuning.ipynb').read_text())
ns = {'__name__': '__main__'}
def source(marker):
    return next(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code' and marker in ''.join(c['source']))
exec(source('CONFIG = {').replace('/content/qlora_lab', str(work)), ns)
exec(source('FAQ = {'), ns)
transformers.set_seed(42)
exec(source('DEVICE = "cuda"').replace('DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")', 'DEVICE = ' + repr(args.device)), ns)
model = ns['model']
model.set_attn_implementation('sdpa')
setup = source('from peft import LoraConfig').split('t0 = time.time()')[0].replace('fp16=USE_AMP,', 'fp16=False, bf16=False, use_cpu=' + repr(args.device=='cpu') + ', loss_type="nll",')
exec(setup, ns)
trainer = ns['trainer']
model = trainer.model
batch = trainer.data_collator([trainer.train_dataset[0]])
batch = {k:v.to(args.device) for k,v in batch.items()}
model.eval()  # Disable dropout for matching comparisons; autograd remains enabled.
if args.cpu_op:
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    if args.cpu_op in ['chunk', 'conv']:
        name = {'chunk':'torch_chunk_gated_delta_rule', 'conv':'causal_conv1d_fn'}[args.cpu_op]
        original_op = getattr(qwen, name)
        def on_cpu(*a, **kw):
            device = a[0].device
            cpu = lambda x:x.to('cpu') if isinstance(x, torch.Tensor) else x
            out = original_op(*(cpu(x) for x in a), **{k:cpu(v) for k,v in kw.items()})
            back = lambda x:x.to(device) if isinstance(x, torch.Tensor) else x
            return tuple(back(x) for x in out) if isinstance(out,tuple) else back(out)
        setattr(qwen, name, on_cpu)
    else:
        def rms(self,x):
            xc = x.to('cpu')
            return (xc * torch.rsqrt(xc.pow(2).mean(-1,keepdim=True)+self.eps)).to(x.device)
        qwen.Qwen3_5RMSNorm._norm = rms
if args.chunk_size:
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    original = qwen.torch_chunk_gated_delta_rule
    def changed_chunk(*a, **kw):
        kw['chunk_size'] = args.chunk_size
        return original(*a, **kw)
    qwen.torch_chunk_gated_delta_rule = changed_chunk
t0 = time.monotonic()
loss = trainer.compute_loss(model, batch)
print('FORWARD', float(loss.detach()), flush=True)
loss.backward()
norms = {name:float(p.grad.norm().item()) for name,p in model.named_parameters() if p.requires_grad and p.grad is not None}
global_norm = sum(v*v for v in norms.values())**.5
result = {'device':args.device, 'cpu_op':args.cpu_op, 'chunk_size':args.chunk_size, 'mps_silu_grad_contiguous':args.mps_silu_grad_contiguous, 'loss':loss.item(), 'grad_norm':global_norm, 'seconds':time.monotonic()-t0, 'shape':list(batch['input_ids'].shape), 'gradient_norms':norms}
out = work / (args.device + ('_chunk'+str(args.chunk_size) if args.chunk_size else '') + ('_cpu_'+args.cpu_op if args.cpu_op else '') + ('_silu_contiguous' if args.mps_silu_grad_contiguous else '') + '_backward.json')
out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
print('SUMMARY', {k:v for k,v in result.items() if k!='gradient_norms'}, flush=True)
print('LARGEST', sorted(norms.items(), key=lambda x:x[1], reverse=True)[:8], flush=True)
