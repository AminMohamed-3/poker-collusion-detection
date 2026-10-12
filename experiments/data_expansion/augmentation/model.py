"""Small V4.1-inspired causal encoder/decoder for controlled local experiments.

mHC equations and lagged pre-mixing follow DeepSeek's MIT-licensed V4.1
reference (pinned sources and license in references/). This original PyTorch
implementation omits production kernels, sparse indexing, quantization,
vision, Engram, and speculative decoding. It is not a DeepSeek checkpoint.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Config:
    input_features: int = 897
    tokens: int = 24
    width: int = 48
    heads: int = 4
    layers: int = 4
    expert_hidden: int = 64
    experts: int = 4
    topk: int = 2
    streams: int = 4
    sinkhorn_steps: int = 20
    eps: float = 1e-6
    window: int = 8
    dropout: float = 0.1
    route_scale: float = 1.5
    balance_rate: float = 0.001
    connection: str = 'single_pass'


class RMSNorm(nn.Module):
    def __init__(self, width, eps=1e-6):
        super().__init__(); self.weight=nn.Parameter(torch.ones(width)); self.eps=eps

    def forward(self,x):
        return (x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+self.eps)).to(x.dtype)*self.weight


def sinkhorn(logits, steps=20, eps=1e-6):
    """Exact reference normalization order; finite iterations are approximate."""
    p=logits.float().softmax(-1)+eps
    p=p/(p.sum(-2,keepdim=True)+eps)
    for _ in range(steps-1):
        p=p/(p.sum(-1,keepdim=True)+eps)
        p=p/(p.sum(-2,keepdim=True)+eps)
    return p


class HyperConnection(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.n=cfg.streams; self.steps=cfg.sinkhorn_steps; self.eps=cfg.eps
        size=self.n*self.n+2*self.n
        self.projection=nn.Parameter(torch.empty(size,self.n*cfg.width))
        nn.init.normal_(self.projection,std=1/math.sqrt(self.n*cfg.width))
        self.scale=nn.Parameter(torch.full((3,),0.01))
        base=torch.zeros(size)
        base[:self.n]=math.log((1/self.n)/(1-1/self.n))
        # A mildly identity-favouring residual prior; this is our initialization.
        base[2*self.n:]=(torch.eye(self.n)*4).flatten()
        self.base=nn.Parameter(base)

    def coefficients(self,x):
        flat=x.flatten(-2).float()
        values=F.linear(flat,self.projection)*torch.rsqrt(flat.square().mean(-1,keepdim=True)+self.eps)
        n=self.n
        pre=(values[...,:n]*self.scale[0]+self.base[:n]).sigmoid()+self.eps
        post=2*(values[...,n:2*n]*self.scale[1]+self.base[n:2*n]).sigmoid()
        residual=sinkhorn((values[...,2*n:]*self.scale[2]+self.base[2*n:]).unflatten(-1,(n,n)),self.steps,self.eps)
        return pre,post,residual

    @staticmethod
    def collapse(x,pre): return (x*pre.unsqueeze(-1)).sum(-2)

    @staticmethod
    def merge(x,update,post,residual):
        # Official reference stores comb[input_stream, output_stream].
        return residual.transpose(-1,-2)@x + post.unsqueeze(-1)*update.unsqueeze(-2)


class SwiGLU(nn.Module):
    def __init__(self,width,hidden):
        super().__init__(); self.up=nn.Linear(width,2*hidden,bias=False); self.down=nn.Linear(hidden,width,bias=False)

    def forward(self,x):
        gate,up=self.up(x).chunk(2,-1)
        return self.down(F.silu(gate.clamp(max=10))*up.clamp(-10,10))


class MoE(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.topk=cfg.topk; self.route_scale=cfg.route_scale; self.balance_rate=cfg.balance_rate
        self.gate=nn.Linear(cfg.width,cfg.experts,bias=False)
        self.experts=nn.ModuleList([SwiGLU(cfg.width,cfg.expert_hidden) for _ in range(cfg.experts)])
        self.shared=SwiGLU(cfg.width,cfg.expert_hidden)
        self.register_buffer('correction',torch.zeros(cfg.experts))
        self.register_buffer('last_load',torch.zeros(cfg.experts),persistent=False)

    def forward(self,x):
        scores=F.softplus(self.gate(x).float()).sqrt()
        ids=(scores+self.correction).topk(self.topk,dim=-1).indices
        weights=scores.gather(-1,ids); weights=self.route_scale*weights/(weights.sum(-1,keepdim=True)+1e-20)
        dispatch=torch.zeros_like(scores).scatter(-1,ids,weights)
        # Dense evaluation is efficient at four tiny experts; routing is top-k.
        outputs=torch.stack([expert(x) for expert in self.experts],-2)
        result=self.shared(x)+(outputs*dispatch.unsqueeze(-1)).sum(-2)
        if self.training:
            with torch.no_grad(): self.last_load.copy_(F.one_hot(ids,num_classes=len(self.experts)).float().sum(tuple(range(ids.ndim))))
        return result

    @torch.no_grad()
    def balance(self):
        # Selection-only bias, separate from differentiable routing weights.
        self.correction.add_(self.balance_rate*torch.sign(self.last_load.mean()-self.last_load))
        self.correction.sub_(self.correction.mean())


class Attention(nn.Module):
    def __init__(self,cfg,owns_global):
        super().__init__(); self.heads=cfg.heads; self.width=cfg.width; self.window=cfg.window
        self.query=nn.Linear(cfg.width,cfg.width,bias=False)
        self.local_kv=nn.Linear(cfg.width,2*cfg.width,bias=False)
        self.global_kv=nn.Linear(cfg.width,2*cfg.width,bias=False) if owns_global else None
        self.output=nn.Linear(cfg.width,cfg.width,bias=False)

    def project_memory(self,x):
        assert self.global_kv is not None
        k,v=self.global_kv(x).chunk(2,-1)
        return self.split(k),self.split(v)

    def split(self,x): return x.unflatten(-1,(self.heads,self.width//self.heads)).transpose(1,2)

    def forward(self,x,memory):
        q=self.split(self.query(x)); local_k,local_v=self.local_kv(x).chunk(2,-1)
        k=torch.cat([self.split(local_k),memory[0]],-2); v=torch.cat([self.split(local_v),memory[1]],-2)
        t=x.shape[1]; pos=torch.arange(t,device=x.device); delta=pos[:,None]-pos[None,:]
        allowed=torch.cat([(delta>=0)&(delta<self.window),delta>=0],-1)
        h=F.scaled_dot_product_attention(q,k,v,attn_mask=allowed,dropout_p=0.)
        return self.output(h.transpose(1,2).flatten(-2))


class Layer(nn.Module):
    def __init__(self,cfg,index):
        super().__init__(); self.index=index; self.encoder_layers=cfg.layers//2
        self.attn_norm=RMSNorm(cfg.width,cfg.eps); self.ffn_norm=RMSNorm(cfg.width,cfg.eps)
        self.attention=Attention(cfg,owns_global=index<=self.encoder_layers)
        self.moe=MoE(cfg); self.dropout=nn.Dropout(cfg.dropout)


class TinyCED(nn.Module):
    def __init__(self,cfg):
        super().__init__(); assert cfg.layers%2==0 and cfg.width%cfg.heads==0
        assert cfg.connection in ['residual','mhc','single_pass']
        self.cfg=cfg; self.chunk=math.ceil(cfg.input_features/cfg.tokens)
        self.input=nn.Linear(self.chunk,cfg.width)
        self.position=nn.Parameter(torch.randn(1,cfg.tokens+1,cfg.width)*.02)
        self.query_token=nn.Parameter(torch.randn(1,1,cfg.width)*.02)
        self.layers=nn.ModuleList([Layer(cfg,i) for i in range(cfg.layers)])
        self.final_norm=RMSNorm(cfg.width,cfg.eps); self.head=nn.Linear(cfg.width,4)
        # Instantiate after the backbone: a shared seed matches its initial weights.
        self.connections=nn.ModuleList([HyperConnection(cfg) for _ in range(2*cfg.layers)]) if cfg.connection!='residual' else nn.ModuleList()

    def encode_tokens(self,tokens):
        cfg=self.cfg; h=tokens; previous=None; memory=None; boundary=None
        if cfg.connection!='residual':
            h=h.unsqueeze(-2).expand(*h.shape[:-1],cfg.streams,cfg.width)
            previous=h.new_zeros(*h.shape[:-1]); previous[...,0]=1
        for i,layer in enumerate(self.layers):
            for sub in range(2):
                if cfg.connection=='residual': z=h
                else:
                    mixer=self.connections[2*i+sub]; pre,post,comb=mixer.coefficients(h)
                    z=mixer.collapse(h,previous if cfg.connection=='single_pass' else pre)
                if sub==0:
                    z=layer.attn_norm(z)
                    if i<cfg.layers//2: memory=layer.attention.project_memory(z)
                    elif i==cfg.layers//2:
                        # All decoder global KV derive from final causal encoder states.
                        boundary=z
                        memory=layer.attention.project_memory(boundary)
                    update=layer.attention(z,memory)
                else: update=layer.moe(layer.ffn_norm(z))
                update=layer.dropout(update)
                if cfg.connection=='residual': h=h+update
                else: h=mixer.merge(h,update,post,comb); previous=pre
        if cfg.connection!='residual': h=HyperConnection.collapse(h,previous)
        return self.final_norm(h)

    def forward(self,pair):
        padding=self.cfg.tokens*self.chunk-self.cfg.input_features
        chunks=F.pad(pair,(0,padding)).reshape(len(pair),self.cfg.tokens,self.chunk)
        h=torch.cat([self.input(chunks),self.query_token.expand(len(pair),-1,-1)],1)+self.position
        return self.head(self.encode_tokens(h)[:,-1])

    @torch.no_grad()
    def update_routing(self):
        for layer in self.layers: layer.moe.balance()

    def parameter_count(self): return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def constraint_diagnostic(self,pair):
        if not self.connections: return {}
        h=self.input(F.pad(pair,(0,self.cfg.tokens*self.chunk-self.cfg.input_features)).reshape(len(pair),self.cfg.tokens,self.chunk))
        x=h.unsqueeze(-2).expand(*h.shape[:-1],self.cfg.streams,self.cfg.width)
        values=[]
        for m in self.connections:
            _,_,b=m.coefficients(x)
            values.append({'row_error':float((b.sum(-1)-1).abs().max()),'column_error':float((b.sum(-2)-1).abs().max()),'minimum':float(b.min())})
        return {'probe':'Every learned mixer on the same normalized input-token probe; not a full forward trajectory.','max_row_error':max(v['row_error'] for v in values),'max_column_error':max(v['column_error'] for v in values),'minimum_entry':min(v['minimum'] for v in values)}
