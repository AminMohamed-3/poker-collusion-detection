"""Small position-free contextual residual ranker; no identifiers as inputs."""
import torch
from torch import nn
from torch.nn import functional as F


class ContextReranker(nn.Module):
    def __init__(self,features,config):
        super().__init__()
        width=config['width']
        self.input=nn.Linear(features,width)
        self.blocks=nn.ModuleList([nn.TransformerEncoderLayer(width,config['heads'],config['feedforward'],
            dropout=config['dropout'],activation='gelu',batch_first=True,norm_first=True) for _ in range(config['layers'])])
        self.norm=nn.LayerNorm(width)
        self.head=nn.Linear(width,1)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
        self.scale=config['residual_scale']

    def forward(self,x,baseline,mask):
        hidden=self.input(x)
        for block in self.blocks: hidden=block(hidden,src_key_padding_mask=~mask)
        correction=self.scale*self.head(self.norm(hidden)).squeeze(-1)
        correction=correction.masked_fill(~mask,0)
        return baseline+correction,correction


def ranking_loss(score,relevance,mask,objective):
    positive=(relevance>0)&mask
    if objective=='pairwise':
        comparisons=positive[:,:,None]&((relevance<=0)&mask)[:,None,:]
        loss=F.softplus(-(score[:,:,None]-score[:,None,:]))
        counts=comparisons.sum((1,2)); valid=counts>0
        per_query=(loss*comparisons).sum((1,2))/counts.clamp_min(1)
    elif objective=='listwise':
        counts=positive.sum(1); valid=counts>0
        target=positive.to(score.dtype)/counts.clamp_min(1)[:,None]
        per_query=-(target*F.log_softmax(score.masked_fill(~mask,-10000),dim=1)).sum(1)
    else: raise ValueError(objective)
    assert valid.any()
    return per_query[valid].mean()
