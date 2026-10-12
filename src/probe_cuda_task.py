"""Bounded CUDA availability check, run under an external timeout."""
import torch
print({'torch':torch.__version__,'stage':'imported'},flush=True)
value=torch.ones(1,device='cuda')
print({'stage':'cuda_tensor_succeeded','value':value.item()},flush=True)
