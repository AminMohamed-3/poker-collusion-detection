import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from mil_pair_model import PairMIL,objective


def test_padding_and_permutation_do_not_change_pair_scores():
    torch.manual_seed(93)
    model=PairMIL(7,9).eval()
    x=torch.randn(3,8,7); pair=torch.randn(3,9)
    mask=torch.arange(8)[None,:]<torch.tensor([5,8,3])[:,None]
    order=torch.tensor([7,0,5,2,6,1,3,4])
    with torch.no_grad():
        a,e=model(x,mask,pair)
        b,f=model(x[:,order],mask[:,order],pair)
        c,_=model(torch.cat([x,torch.randn(3,4,7)*1000],1),torch.cat([mask,torch.zeros(3,4,dtype=torch.bool)],1),pair)
    torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-6)
    torch.testing.assert_close(a,c,atol=2e-6,rtol=2e-6)
    torch.testing.assert_close(e[:,order],f)


def test_auxiliary_ignores_normal_annotations_and_empty_target_views():
    torch.manual_seed(12)
    logits=torch.randn(3,4,requires_grad=True); events=torch.randn(3,5,3,requires_grad=True)
    mask=torch.ones(3,5,dtype=torch.bool); y=torch.tensor([0,1,3]); relevance=torch.zeros(3,5)
    a=objective(logits,events,mask,y,relevance,False)
    relevance[0]=1
    b=objective(logits,events,mask,y,relevance,True)
    torch.testing.assert_close(a,b)
    relevance[2,1]=1
    c=objective(logits,events,mask,y,relevance,True)
    assert c>a
    c.backward()
    assert torch.isfinite(events.grad).all()
    assert events.grad[:2].abs().sum()==0
    assert events.grad[2,:,2].abs().sum()>0
    assert events.grad[2,:,:2].abs().sum()==0
