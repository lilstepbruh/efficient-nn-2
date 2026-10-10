"""Native-bf16 XSA vs fp32-projection ablation; paired block timings."""
import json,time,statistics
from pathlib import Path
import torch
from .model import Config,Block
from .train import ROOT,atomic_json

def main():
 torch.set_num_threads(4);torch.manual_seed(99);rows=[]
 # Initialize CUDA/BLAS workspaces outside measured memory baselines.
 q=torch.randn(256,256,device='cuda',dtype=torch.bfloat16,requires_grad=True)
 (q@q).float().square().mean().backward();del q;torch.cuda.synchronize()
 for width,head_dim in [(512,128),(1536,256)]:
  for length in [512,2048,8192]:
   for backward in [False,True]:
    modes=['SA','XSA','XSA_fp32'];values={m:[] for m in modes};mems={m:[] for m in modes};event_times={m:[] for m in modes}
    for round_id in range(5):
     for mode in (modes if round_id%2==0 else modes[::-1]):
      torch.cuda.synchronize();base=torch.cuda.memory_allocated();torch.manual_seed(99)
      c=Config(width=width,heads=width//head_dim,head_dim=head_dim,xsa=mode!='SA',projection_fp32=mode=='XSA_fp32')
      block=Block(c).cuda().bfloat16();x=torch.randn(4,length,width,device='cuda',dtype=torch.bfloat16,requires_grad=backward)
      angle=torch.outer(torch.arange(length,device='cuda').float(),10000.**(-torch.arange(0,head_dim,2,device='cuda').float()/head_dim))
      co,si=angle.cos().bfloat16(),angle.sin().bfloat16();del angle
      def iteration():
       block.zero_grad(set_to_none=True);x.grad=None
       with torch.set_grad_enabled(backward):
        y=block(x,co,si)
        if backward:y.float().square().mean().backward()
      for _ in range(10):iteration()
      torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
      begin,end=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
      tick=time.perf_counter();begin.record()
      for _ in range(20):iteration()
      end.record();torch.cuda.synchronize()
      values[mode].append((time.perf_counter()-tick)*1000/20);event_times[mode].append(begin.elapsed_time(end)/20)
      mems[mode].append((torch.cuda.max_memory_allocated()-base)/2**20)
      del block,x,co,si,begin,end
    for mode in modes:rows.append({'width':width,'head_dim':head_dim,'heads':width//head_dim,'length':length,'batch':4,'backward':backward,'mode':mode,'wall_ms':statistics.median(values[mode]),'cuda_event_ms':statistics.median(event_times[mode]),'round_wall_ms':values[mode],'round_event_ms':event_times[mode],'peak_incremental_mib':max(mems[mode])})
    print(json.dumps({'width':width,'length':length,'backward':backward,'ms':{m:statistics.median(values[m]) for m in modes}}),flush=True)
 out=ROOT/'results/fineweb_v2';atomic_json(out/'benchmark.json',{'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'dtype':'bf16; XSA_fp32 uses fp32 projection accumulation','scope':'one block, batch4, eager; cached RoPE; automatic SDPA backend; 10 warmup, 5x20 iterations','rows':rows})
if __name__=='__main__':main()
