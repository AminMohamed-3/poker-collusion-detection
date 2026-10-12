"""Bounded synthetic GPU parity and training probe, without poker labels."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
from dataclasses import replace
import json
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from model import Config, TinyCED

torch.set_num_threads(2)
torch.set_num_interop_threads(2)
torch.use_deterministic_algorithms(True)
assert torch.cuda.is_available()
records = []
for mode in ['residual', 'mhc', 'single_pass']:
    torch.manual_seed(80132)
    cfg = replace(Config(), connection=mode, dropout=0.)
    cpu = TinyCED(cfg).eval()
    gpu = TinyCED(cfg).cuda().eval()
    gpu.load_state_dict(cpu.state_dict())
    x = torch.randn(64, cfg.input_features)
    y = torch.arange(64) % 4
    with torch.no_grad():
        a = cpu(x)
        b = gpu(x.cuda()).cpu()
    torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-4)
    gpu.train()
    xg, yg = x.cuda(), y.cuda()
    opt = torch.optim.AdamW(gpu.parameters(), lr=.001)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    losses = []
    for step in range(12):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(gpu(xg), yg)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(gpu.parameters(), 2.)
        assert torch.isfinite(loss) and torch.isfinite(norm)
        opt.step()
        gpu.update_routing()
        losses.append(float(loss.detach()))
    torch.cuda.synchronize()
    row = {'variant': mode, 'cpu_gpu_max_logit_difference': float((a-b).abs().max()),
           'steps': 12, 'seconds_per_step': (time.perf_counter()-start)/12,
           'first_loss': losses[0], 'last_loss': losses[-1],
           'peak_allocated_bytes': torch.cuda.max_memory_allocated()}
    records.append(row)
    print(json.dumps(row), flush=True)
    del cpu, gpu, opt, xg, yg
    torch.cuda.empty_cache()
receipt = {'torch': torch.__version__, 'cuda': torch.version.cuda,
           'gpu': torch.cuda.get_device_name(), 'passed': True, 'records': records}
(Path(__file__).parent / 'runtime-check.json').write_text(json.dumps(receipt, indent=2))
