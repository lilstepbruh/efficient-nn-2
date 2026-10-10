"""nanoGPT-style pre-LN Transformer with RoPE, embedding LN, and XSA."""
import math
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F

@dataclass
class Config:
    vocab_size: int = 50257
    width: int = 512
    layers: int = 8
    heads: int = 4
    head_dim: int = 128
    max_length: int = 2048
    xsa: bool = False
    projection_fp32: bool = False

def project(y,v,fp32=False):
    work=torch.float32 if fp32 and y.dtype in (torch.float16,torch.bfloat16) else y.dtype
    vn=F.normalize(v.to(work),dim=-1,eps=1e-12)
    yy=y.to(work)
    return (yy-(yy*vn).sum(-1,keepdim=True)*vn).to(y.dtype)

class Attention(nn.Module):
    def __init__(self,c):
        super().__init__(); self.c=c
        self.qkv=nn.Linear(c.width,3*c.heads*c.head_dim,bias=False)
        self.out=nn.Linear(c.heads*c.head_dim,c.width,bias=False)
    def forward(self,x,cos,sin):
        b,t,_=x.shape
        q,k,v=self.qkv(x).view(b,t,3,self.c.heads,self.c.head_dim).permute(2,0,3,1,4).unbind(0)
        def rotate(a):
            u,w=a[...,::2],a[...,1::2]
            return torch.stack((u*cos-w*sin,u*sin+w*cos),dim=-1).flatten(-2)
        q,k=rotate(q),rotate(k)
        y=F.scaled_dot_product_attention(q,k,v,is_causal=True)
        z=project(y,v,self.c.projection_fp32) if self.c.xsa else y
        return self.out(z.transpose(1,2).reshape(b,t,-1))

class Block(nn.Module):
    def __init__(self,c):
        super().__init__(); self.ln1=nn.LayerNorm(c.width);self.ln2=nn.LayerNorm(c.width)
        self.attn=Attention(c)
        self.ffn=nn.Sequential(nn.Linear(c.width,4*c.width),nn.GELU(),nn.Linear(4*c.width,c.width))
    def forward(self,x,cos,sin):
        x=x+self.attn(self.ln1(x),cos,sin)
        return x+self.ffn(self.ln2(x))

class LM(nn.Module):
    def __init__(self,c):
        super().__init__();self.config=c
        assert c.head_dim%2==0
        self.emb=nn.Embedding(c.vocab_size,c.width);self.emb_ln=nn.LayerNorm(c.width)
        self.blocks=nn.ModuleList([Block(c) for _ in range(c.layers)])
        self.norm=nn.LayerNorm(c.width)
        # Shared LM output uses F.linear, so embedding is initialized exactly once.
        self.apply(self._init)
        for b in self.blocks:
            nn.init.normal_(b.attn.out.weight,std=.02/math.sqrt(2*c.layers))
            nn.init.normal_(b.ffn[2].weight,std=.02/math.sqrt(2*c.layers))
        angles=torch.outer(torch.arange(c.max_length).float(),10000.**(-torch.arange(0,c.head_dim,2).float()/c.head_dim))
        self.register_buffer('rope_cos',angles.cos(),persistent=False)
        self.register_buffer('rope_sin',angles.sin(),persistent=False)
    @staticmethod
    def _init(m):
        if isinstance(m,(nn.Linear,nn.Embedding)):
            nn.init.normal_(m.weight,std=.02)
            if isinstance(m,nn.Linear) and m.bias is not None:nn.init.zeros_(m.bias)
    def features(self,tokens):
        x=self.emb_ln(self.emb(tokens))
        dtype=torch.get_autocast_dtype('cuda') if x.is_cuda and torch.is_autocast_enabled('cuda') else x.dtype
        cos=self.rope_cos[:tokens.size(1)].to(dtype);sin=self.rope_sin[:tokens.size(1)].to(dtype)
        for b in self.blocks:x=b(x,cos,sin)
        return self.norm(x)
    def forward(self,tokens,targets=None,loss_start=0):
        h=self.features(tokens)[:,loss_start:]
        logits=F.linear(h,self.emb.weight)
        if targets is None:return logits
        return F.cross_entropy(logits.reshape(-1,self.config.vocab_size),targets[:,loss_start:].reshape(-1))
