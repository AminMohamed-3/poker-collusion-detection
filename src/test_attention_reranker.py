"""Tests of the contextual ranker's invariants and training loss semantics."""
import unittest
import torch
from attention_reranker_model import ContextReranker,ranking_loss

CONFIG={'width':16,'heads':4,'layers':2,'feedforward':32,'dropout':.1,'residual_scale':.5}


class AttentionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2); torch.manual_seed(5319)

    def inputs(self):
        x=torch.randn(2,8,7); base=torch.randn(2,8)
        mask=torch.tensor([[1,1,1,1,1,0,0,0],[1,1,1,1,1,1,1,1]],dtype=torch.bool)
        return x,base,mask

    def test_zero_head_exactly_preserves_first_stage(self):
        model=ContextReranker(7,CONFIG).eval(); x,base,mask=self.inputs()
        score,correction=model(x,base,mask)
        self.assertTrue(torch.equal(score,base)); self.assertTrue(torch.equal(correction,torch.zeros_like(correction)))

    def test_nonzero_model_permutation_equivariance(self):
        model=ContextReranker(7,CONFIG).eval(); torch.nn.init.normal_(model.head.weight)
        x,base,mask=self.inputs(); permutation=torch.tensor([4,2,7,1,5,3,0,6]); inverse=permutation.argsort()
        actual=model(x[:,permutation],base[:,permutation],mask[:,permutation])[0][:,inverse]
        expected=model(x,base,mask)[0]
        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)

    def test_padding_cannot_influence_real_hands(self):
        model=ContextReranker(7,CONFIG).eval(); torch.nn.init.normal_(model.head.weight)
        x,base,mask=self.inputs(); changed=x.clone(); changed[~mask]=1000
        torch.testing.assert_close(model(changed,base,mask)[0][mask],model(x,base,mask)[0][mask],rtol=1e-5,atol=1e-6)

    def test_losses_raise_positive_and_lower_negative_scores(self):
        for objective in ['pairwise','listwise']:
            score=torch.tensor([[0.,1.,1000.]],requires_grad=True)
            mask=torch.tensor([[True,True,False]]); truth=torch.tensor([[1.,0.,1.]])
            loss=ranking_loss(score,truth,mask,objective); loss.backward()
            self.assertLess(score.grad[0,0].item(),0); self.assertGreater(score.grad[0,1].item(),0)
            self.assertEqual(score.grad[0,2].item(),0)
            equivalent=ranking_loss(score[:,:2],truth[:,:2],mask[:,:2],objective)
            torch.testing.assert_close(loss,equivalent)

    def test_gradient_reaches_nonzero_head_and_is_finite(self):
        model=ContextReranker(7,CONFIG); x,base,mask=self.inputs()
        truth=torch.zeros_like(base); truth[:,0]=1
        loss=ranking_loss(model(x,base,mask)[0],truth,mask,'pairwise'); loss.backward()
        self.assertTrue(torch.isfinite(model.head.weight.grad).all())
        self.assertGreater(model.head.weight.grad.abs().sum().item(),0)


if __name__=='__main__': unittest.main()
