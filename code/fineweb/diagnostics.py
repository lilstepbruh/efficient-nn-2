"""Layer diagnostics on validation; seed 17 chosen in advance, not by quality."""
import json,math
import numpy as np
import torch
from torch.nn import functional as F
from .model import Config,LM,project
from .train import ROOT,data_stream,evaluation_starts,atomic_json

@torch.no_grad()
def main():
 torch.set_num_threads(4);out=ROOT/'results/fineweb_v2';data=data_stream('validation');results={}
 for mode in ['SA','XSA']:
  path=out/f'c2048_s17_{mode.lower()}'/'model.pt'
  if not path.exists():continue
  ck=torch.load(path,map_location='cpu',weights_only=False);model=LM(Config(**ck['config'])).cuda();model.load_state_dict(ck['model']);model.eval()
  accum=[[] for _ in model.blocks];hooks=[]
  def hook(layer):
   def capture(attn,inputs):
    x,cos,sin=inputs;b,t,_=x.shape;c=attn.c
    q,k,v=attn.qkv(x).view(b,t,3,c.heads,c.head_dim).permute(2,0,3,1,4).unbind(0)
    def rotate(a):
     u,w=a[...,::2],a[...,1::2]
     return torch.stack((u*cos-w*sin,u*sin+w*cos),-1).flatten(-2)
    q,k=rotate(q),rotate(k);y=F.scaled_dot_product_attention(q,k,v,is_causal=True);z=project(y,v,c.projection_fp32) if c.xsa else y
    vf=v.float();yf=y.float();zf=z.float();before=F.cosine_similarity(yf[:,:,1:],vf[:,:,1:],dim=-1)
    after=F.cosine_similarity(zf[:,:,1:],vf[:,:,1:],dim=-1);valid=zf[:,:,1:].norm(dim=-1)>1e-6
    absolute=after[valid].abs()
    # FP32 projection on the same BF16-generated Y,V isolates arithmetic error.
    exact=project(yf,vf);orth=(exact*vf).sum(-1).abs()
    vn=F.normalize(vf,dim=-1)
    pair_cos=((vn.sum(-2).square().sum(-1)-vn.square().sum((-2,-1)))/(t*(t-1))).mean()
    # Disable autocast for diagnostic score calculation explicitly.
    with torch.autocast('cuda',enabled=False):
     scores=q.float()@k.float().transpose(-2,-1)/math.sqrt(c.head_dim)
     mask=torch.ones(t,t,device='cuda',dtype=torch.bool).triu(1);scores.masked_fill_(mask,float('-inf'))
     diagonal=scores.softmax(-1).diagonal(dim1=-2,dim2=-1)[:,:,1:].mean()
    accum[layer].append({'cos_before_mean':float(before.mean()),'cos_after_signed_mean':float(after.mean()),'cos_after_absolute_mean':float(absolute.mean()) if len(absolute) else 0.,'cos_after_absolute_p95':float(torch.quantile(absolute,.95)) if len(absolute) else 0.,'cos_after_absolute_max':float(absolute.max()) if len(absolute) else 0.,'nonzero_fraction':float(valid.float().mean()),'fp32_dot_error_max':float(orth.max()),'value_pair_cos_mean':float(pair_cos),'attention_diagonal_mean':float(diagonal)})
   return capture
  for i,block in enumerate(model.blocks):hooks.append(block.attn.register_forward_pre_hook(hook(i)))
  for s in evaluation_starts(len(data),32):
   x=torch.from_numpy(np.asarray(data[s:s+2048],dtype=np.int64).copy()).unsqueeze(0).cuda()
   with torch.autocast('cuda',dtype=torch.bfloat16):model.features(x)
  for h in hooks:h.remove()
  results[mode]=[]
  for i,rows in enumerate(accum):
   stats={k:(max(r[k] for r in rows) if k.endswith('_max') else float(np.mean([r[k] for r in rows]))) for k in rows[0]}
   results[mode].append({'layer':i+1,**stats})
  del model,ck;torch.cuda.empty_cache()
 atomic_json(out/'diagnostics.json',{'seed':17,'context':2048,'validation_windows':32,'first_position_excluded_from_output_cosine':True,'p95':'mean of per-window p95 over valid nonzero positions','results':results})
 print(json.dumps({'diagnostics_modes':list(results)}),flush=True)
if __name__=='__main__':main()
