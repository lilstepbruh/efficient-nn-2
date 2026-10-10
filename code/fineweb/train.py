"""Paired training with common target streams and matched-target held-out evaluation."""
import argparse,hashlib,json,math,os,platform,random,time
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from .model import Config,LM
ROOT=Path(__file__).resolve().parents[1]

def atomic_json(path,obj):
 temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(obj,indent=2));temp.replace(path)

def data_stream(split):return np.memmap(ROOT/f'data/fineweb/{split}.bin',dtype='<u2',mode='r')

def training_batch(data,rng,context,global_tokens,common_context=2048):
 """Same RNG draws and target-token order for both context lengths and both methods."""
 starts=rng.integers(0,len(data)-common_context,size=global_tokens//common_context)
 chunks=np.stack([data[s:s+common_context+1] for s in starts]).astype(np.int64)
 x=torch.from_numpy(chunks[:,:-1].copy()).reshape(-1,context)
 y=torch.from_numpy(chunks[:,1:].copy()).reshape(-1,context)
 return x,y

def evaluation_starts(n,count,common_context=2048):
 # Shared absolute targets, nonoverlapping 2048-token windows, deterministic selection.
 candidates=np.arange(0,n-common_context,common_context+1,dtype=np.int64)
 if len(candidates)<count:raise ValueError('Evaluation corpus too short for requested independent windows')
 ids=np.linspace(0,len(candidates)-1,count,dtype=np.int64)
 return candidates[ids]

@torch.no_grad()
def evaluate(model,data,context,count=128,target_tokens=256):
 model.eval();starts=evaluation_starts(len(data),count);losses=[]
 for start in starts:
  block=np.asarray(data[start:start+2049],dtype=np.int64)
  x=torch.from_numpy(block[2048-context:2048].copy()).unsqueeze(0).cuda()
  y=torch.from_numpy(block[2049-context:2049].copy()).unsqueeze(0).cuda()
  with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(x,y,loss_start=context-target_tokens)
  losses.append(float(loss))
 model.train()
 return {'loss':float(np.mean(losses)),'perplexity':math.exp(float(np.mean(losses))),'window_losses':losses,'windows':count,'target_tokens_per_window':target_tokens,'evaluation':'same absolute last-256 targets of fixed disjoint 2048-token windows'}

def parse_args():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--mode',choices=['SA','XSA'],required=True)
 p.add_argument('--seed',type=int,default=17);p.add_argument('--context',type=int,choices=[512,2048],default=2048)
 p.add_argument('--tokens',type=int,default=150_994_944);p.add_argument('--global-tokens',type=int,default=16384)
 p.add_argument('--micro-tokens',type=int,default=4096);p.add_argument('--lr',type=float,default=6e-4)
 p.add_argument('--eval-every',type=int,default=512);p.add_argument('--eval-windows',type=int,default=128)
 p.add_argument('--resume',action='store_true');p.add_argument('--tuning',action='store_true')
 p.add_argument('--width',type=int,default=512);p.add_argument('--layers',type=int,default=8)
 return p.parse_args()

def main():
 a=parse_args();torch.set_num_threads(4)
 assert a.tokens%a.global_tokens==0 and a.global_tokens%2048==0
 assert a.global_tokens%a.micro_tokens==0 and a.micro_tokens%a.context==0
 out=ROOT/a.out;out.mkdir(parents=True,exist_ok=True)
 if (out/'result.json').exists():raise RuntimeError('Completed run exists; refusing to overwrite')
 if (out/'last.pt').exists() and not a.resume:raise RuntimeError('Checkpoint exists; pass --resume')
 torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
 cfg=Config(width=a.width,layers=a.layers,xsa=a.mode=='XSA')
 model=LM(cfg).cuda()
 groups=[{'params':[v for v in model.parameters() if v.ndim>=2],'weight_decay':.1},
         {'params':[v for v in model.parameters() if v.ndim<2],'weight_decay':0.}]
 opt=torch.optim.AdamW(groups,lr=a.lr,betas=(.9,.95),eps=1e-8,fused=True)
 rng=np.random.default_rng(a.seed+1000)
 train=data_stream('train');val=data_stream('validation')
 manifest=json.loads((ROOT/'data/fineweb/manifest.json').read_text())
 args=vars(a).copy();args.pop('resume')
 meta={'args':args,'config':asdict(cfg),'parameters':sum(p.numel() for p in model.parameters()),'nonembedding_parameters':sum(p.numel() for n,p in model.named_parameters() if n!='emb.weight'),
       'torch':torch.__version__,'python':platform.python_version(),'numpy':np.__version__,'gpu':torch.cuda.get_device_name(0),'cuda':torch.version.cuda,'training':'fp32 weights, bf16 autocast, native-dtype XSA projection, fused AdamW','data_manifest':manifest,'source_sha256':{name:hashlib.sha256((ROOT/'fineweb'/name).read_bytes()).hexdigest() for name in ['model.py','train.py']}}
 start_step=0;curve=[];train_seconds=0.;eval_seconds=0.
 if a.resume:
  ck=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
  if ck['meta']['args']!=args or ck['meta']['source_sha256']!=meta['source_sha256']:raise ValueError('Resume configuration/code mismatch')
  model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);rng.bit_generator.state=ck['numpy_rng']
  torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng'])
  start_step=ck['step'];curve=ck['curve'];train_seconds=ck['train_seconds'];eval_seconds=ck['eval_seconds']
 atomic_json(out/'environment.json',meta)
 steps=a.tokens//a.global_tokens;warmup=max(1,round(.01*steps));microbatch=a.micro_tokens//a.context;accum=a.global_tokens//a.micro_tokens
 def save(step):
  obj={'model':model.state_dict(),'optimizer':opt.state_dict(),'step':step,'curve':curve,'train_seconds':train_seconds,'eval_seconds':eval_seconds,'numpy_rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'meta':meta}
  torch.save(obj,out/'last.pt.tmp');(out/'last.pt.tmp').replace(out/'last.pt')
 if not curve:
  t=time.perf_counter();ev=evaluate(model,val,a.context,a.eval_windows);eval_seconds+=time.perf_counter()-t
  curve.append({'step':0,'tokens':0,'train_seconds':0,'validation_loss':ev['loss']});atomic_json(out/'curve.json',curve)
 segment=[];wall_start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
 for step in range(start_step,steps):
  rate=a.lr*(step+1)/warmup if step<warmup else a.lr*(.1+.45*(1+math.cos(math.pi*(step-warmup)/max(1,steps-warmup-1))))
  for group in opt.param_groups:group['lr']=rate
  torch.cuda.synchronize();tick=time.perf_counter()
  xx,yy=training_batch(train,rng,a.context,a.global_tokens)
  opt.zero_grad(set_to_none=True);total=torch.zeros((),device='cuda')
  for j in range(0,len(xx),microbatch):
   x=xx[j:j+microbatch].cuda();y=yy[j:j+microbatch].cuda()
   with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(x,y)/accum
   loss.backward();total+=loss.detach()
  norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
  if not torch.isfinite(total) or not torch.isfinite(norm):raise RuntimeError(f'Nonfinite loss or gradient at step {step}')
  opt.step();torch.cuda.synchronize();train_seconds+=time.perf_counter()-tick
  segment.append(float(total))
  if (step+1)%64==0:print(json.dumps({'run':out.name,'step':step+1,'steps':steps,'tokens':(step+1)*a.global_tokens,'train_loss_recent':float(np.mean(segment[-64:])),'train_seconds':round(train_seconds,1)}),flush=True)
  if (step+1)%a.eval_every==0 or step+1==steps:
   t=time.perf_counter();ev=evaluate(model,val,a.context,a.eval_windows);eval_seconds+=time.perf_counter()-t
   entry={'step':step+1,'tokens':(step+1)*a.global_tokens,'train_seconds':train_seconds,'validation_loss':ev['loss'],'train_loss_mean_since_eval':float(np.mean(segment)),'lr':rate,'grad_norm_last':float(norm)}
   curve.append(entry);segment=[];atomic_json(out/'curve.json',curve);save(step+1)
   print(json.dumps({'evaluation':entry,'run':out.name}),flush=True)
 if start_step==steps:ev=evaluate(model,val,a.context,a.eval_windows)
 torch.save({'model':model.state_dict(),'config':asdict(cfg),'meta':meta},out/'model.pt')
 # Tuning never opens the test file.
 test=None
 if not a.tuning:test=evaluate(model,data_stream('test'),a.context,512)
 result={'meta':meta,'validation':ev,'test':test,'curve':curve,'train_seconds':train_seconds,'eval_seconds':eval_seconds,'this_process_wall_seconds':time.perf_counter()-wall_start,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30}
 atomic_json(out/'result.json',result)
 print(json.dumps({'complete':str(out),'validation_loss':ev['loss'],'test_loss':test['loss'] if test else None}),flush=True)

if __name__=='__main__':main()
