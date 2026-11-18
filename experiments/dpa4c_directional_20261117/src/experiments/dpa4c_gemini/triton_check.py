from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"dpa4c_acvr"))
import torch
from rotary_cuda import rotated_factors
from triton.runtime import driver
driver.active.get_current_target()
with torch.cuda.device(0):
    q=torch.randn(28,1,32,device="cuda",requires_grad=True)
    k=torch.randn_like(q,requires_grad=True)
    phase=torch.randn(28,4,16,device="cuda",requires_grad=True)
    w=torch.ones(4,device="cuda")
    a,b=rotated_factors(q,k,phase,w)
    loss=a.square().mean()+b.square().mean()
    g=torch.autograd.grad(loss,(q,k,phase),create_graph=True)
    d=torch.autograd.grad(sum(x.square().sum() for x in g),(q,k,phase))
    assert all(x.isfinite().all() for x in (*g,*d))
    print("TRITON_DRIVER_AND_SECOND_DERIVATIVES_OK",torch.cuda.get_device_name())
