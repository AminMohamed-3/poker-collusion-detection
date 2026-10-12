"""Architecture invariants, independent contractions and a learning smoke test."""
import io
import json
import math
from pathlib import Path
import time
import unittest
import torch
from torch import nn
from torch.nn import functional as F
from model import Config,HyperConnection,MoE,TinyCED,sinkhorn

torch.set_num_threads(2)

class ArchitectureTests(unittest.TestCase):
    def setUp(self): torch.manual_seed(81127)

    def test_conservation_and_residual_norm(self):
        p=sinkhorn(torch.randn(12,4,4),20)
        self.assertLess(float((p.sum(-1)-1).abs().max()),2e-5)
        self.assertLess(float((p.sum(-2)-1).abs().max()),2e-5)
        self.assertGreater(float(p.min()),0)
        x=torch.randn(12,4,13); y=p.transpose(-1,-2)@x
        torch.testing.assert_close(x.mean(-2),y.mean(-2),atol=2e-5,rtol=2e-5)
        self.assertTrue(torch.all(y.norm(dim=(-2,-1))<=x.norm(dim=(-2,-1))+2e-5))
        product=torch.eye(4).repeat(12,1,1)
        for _ in range(50): product=p@product
        self.assertLess(float((product.sum(-1)-1).abs().max()),2e-4)

    def test_mixer_matches_literal_reference_and_has_gradients(self):
        cfg=Config(width=12,heads=3); mixer=HyperConnection(cfg)
        x=torch.randn(2,3,4,12,requires_grad=True); update=torch.randn(2,3,12,requires_grad=True)
        pre,post,comb=mixer.coefficients(x)
        expected=torch.zeros_like(x)
        for i in range(4):
            for j in range(4): expected[:,:,j]+=comb[:,:,i,j,None]*x[:,:,i]
        expected+=post[...,None]*update[:,:,None]
        actual=mixer.merge(x,update,post,comb)
        torch.testing.assert_close(actual,expected)
        actual.square().mean().backward()
        for p in [x,update,mixer.projection,mixer.base,mixer.scale]:
            self.assertIsNotNone(p.grad); self.assertTrue(torch.isfinite(p.grad).all()); self.assertGreater(float(p.grad.abs().sum()),0)

    def test_single_pass_consumes_previous_pre(self):
        cfg=Config(input_features=20,tokens=4,width=12,heads=3,layers=2,expert_hidden=12,dropout=0.)
        model=TinyCED(cfg).eval(); observed=[]
        hook=model.layers[0].attn_norm.register_forward_pre_hook(lambda module,args: observed.append(args[0].detach().clone()))
        tokens=torch.randn(2,5,12)
        with torch.no_grad(): model.encode_tokens(tokens)
        hook.remove(); torch.testing.assert_close(observed[0],tokens)
        # Change current pre coefficients: first attention input must stay unchanged.
        with torch.no_grad(): model.connections[0].base[:4].add_(3.)
        observed=[]; hook=model.layers[0].attn_norm.register_forward_pre_hook(lambda module,args: observed.append(args[0].detach().clone()))
        with torch.no_grad(): model.encode_tokens(tokens)
        hook.remove(); torch.testing.assert_close(observed[0],tokens)

    def test_all_variants_are_causal_and_reload_exact(self):
        for mode in ['residual','mhc','single_pass']:
            cfg=Config(input_features=20,tokens=4,width=12,heads=3,layers=4,expert_hidden=12,dropout=0.,connection=mode)
            model=TinyCED(cfg).eval(); x=torch.randn(2,7,12); z=x.clone(); z[:,4:]+=100*torch.randn_like(z[:,4:])
            with torch.no_grad(): a=model.encode_tokens(x); b=model.encode_tokens(z)
            torch.testing.assert_close(a[:,:4],b[:,:4],atol=1e-6,rtol=1e-5)
            buffer=io.BytesIO(); torch.save(model.state_dict(),buffer); buffer.seek(0)
            clone=TinyCED(cfg).eval(); clone.load_state_dict(torch.load(buffer,weights_only=True))
            with torch.no_grad(): self.assertTrue(torch.equal(model(x.flatten(1)[:,:20]),clone(x.flatten(1)[:,:20])))

    def test_decoder_reuses_encoder_memory_and_moe_topk(self):
        cfg=Config(input_features=20,tokens=4,width=12,heads=3,layers=4,expert_hidden=12,dropout=0.)
        model=TinyCED(cfg).eval(); memories=[]; hooks=[]
        for layer in model.layers[2:]: hooks.append(layer.attention.register_forward_pre_hook(lambda m,args: memories.append(args[1])))
        with torch.no_grad(): model(torch.randn(2,20))
        for hook in hooks: hook.remove()
        self.assertIs(memories[0],memories[1])
        moe=MoE(cfg); before=moe.correction.clone(); moe(torch.randn(2,5,12)); self.assertEqual(float(moe.last_load.sum()),2*5*cfg.topk)
        moe.balance(); self.assertTrue(torch.isfinite(moe.correction).all())

    def test_common_initial_backbone_and_toy_learning(self):
        models=[]
        for mode in ['residual','mhc','single_pass']:
            torch.manual_seed(55193); cfg=Config(input_features=12,tokens=3,width=12,heads=3,layers=2,expert_hidden=12,dropout=0.,connection=mode)
            models.append(TinyCED(cfg))
        state=models[0].state_dict()
        for other in models[1:]:
            for name,value in state.items(): self.assertTrue(torch.equal(value,other.state_dict()[name]),name)
        model=models[-1]; x=torch.randn(24,12); y=(x[:,0]>0).long()+2*(x[:,1]>0).long(); optimizer=torch.optim.AdamW(model.parameters(),lr=.004)
        initial=float(F.cross_entropy(model(x),y).detach())
        for _ in range(90):
            optimizer.zero_grad(); loss=F.cross_entropy(model(x),y); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),2.); optimizer.step(); model.update_routing()
        final=float(F.cross_entropy(model(x),y).detach()); self.assertLess(final,.35*initial)

if __name__=='__main__':
    began=time.perf_counter(); suite=unittest.defaultTestLoader.loadTestsFromTestCase(ArchitectureTests); result=unittest.TextTestRunner(verbosity=2).run(suite)
    receipt={'passed':result.wasSuccessful(),'tests':result.testsRun,'seconds':time.perf_counter()-began,'torch':torch.__version__,'device':'cpu'}
    (Path(__file__).parent/'tests.json').write_text(json.dumps(receipt,indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
