"""Small-N dense oracle is test-only. Check gradients, angular maps, active-arm physics."""
import itertools
import json
import math
from pathlib import Path
import sys

import torch

from transport import ARMS, PATHS, DirectionalTransport, stf_basis

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

def grid():
    # Degree-11 Lebedev grid, with quadrature normalized to one.
    pts=[]
    weights=[]
    for v,w in [((1.,0.,0.),.012698412698412974),((0.,2**-.5,2**-.5),.022574955908288955),
                ((3**-.5,)*3,.02109374999999996),((1/math.sqrt(11),1/math.sqrt(11),3/math.sqrt(11)),.02017333553791896)]:
        orbit=sorted({tuple(a*b for a,b in zip(p,s)) for p in itertools.permutations(v)
                       for s in itertools.product((-1.,1.), repeat=3)})
        pts.extend(orbit);weights.extend([w]*len(orbit))
    return torch.tensor(pts),torch.tensor(weights)

def factors(r,q,k,n,w):
    phi=(r@n.T).unsqueeze(-1)*torch.linspace(0.,.45,q.shape[-1]//2)
    def rot(x):
        a,b=x[...,0::2].unsqueeze(2),x[...,1::2].unsqueeze(2)
        return torch.stack((a*phi.cos()-b*phi.sin(),a*phi.sin()+b*phi.cos()),-1).flatten(-2)
    return (rot(q)*w[None,None,:,None]).flatten(-2),rot(k).flatten(-2)

def oracle(module,src,r,q,k,n,w):
    """Use relative pair cos / sin times content dot, and Cartesian rank coupling.
    For the full antipodal grid phase terms containing q cross k drop by parity.
    Calls no code from the production factorized kernel.
    """
    b,N=r.shape[:2]; modes=q.shape[-1]//2
    phase=torch.einsum('bijc,gc->bijg',r[:,:,None]-r[:,None,:],n)[...,None]*torch.linspace(0.,.45,modes)
    qk=q.reshape(b,N,modes,2);kk=k.reshape(b,N,modes,2)
    content=torch.einsum('bifz,bjfz->bijf',qk,kk)
    diag=(~torch.eye(N,dtype=torch.bool))[None,:,:,None]
    score={l:torch.einsum('bijgf,bijf->bijg',fun(phase),content)*w*diag for l,fun in ((0,torch.cos),(1,torch.sin))}
    out=[torch.zeros_like(v) for v in src]
    D=n[:,:,None]*n[:,None,:]-torch.eye(3)[None]/3; bas=stf_basis()
    for path in module.paths:
        s,l,t=path;x=torch.einsum('bijg,bjsc->bigsc',score[l%2],src[s])
        if l==0: y=x.sum(2)
        elif (s,t)==(0,1): y=torch.einsum('bigc,ga->biac',x.squeeze(-2),n)
        elif (s,t)==(1,0): y=torch.einsum('bigac,ga->bic',x,n).unsqueeze(-2)
        elif (s,t)==(0,2): y=torch.einsum('bigc,gxy,pxy->bipc',x.squeeze(-2),D,bas)
        elif (s,t)==(2,0): y=torch.einsum('bigpc,pxy,gx,gy->bic',x,bas,n,n).unsqueeze(-2)
        elif (s,t)==(1,1): y=torch.einsum('bigac,gda->bidc',x,D)
        elif (s,t)==(1,2): y=torch.einsum('bigac,gx,pax->bipc',x,n,bas)
        elif (s,t)==(2,1): y=torch.einsum('bigpc,pad,gd->biac',x,bas,n)
        else:
            tensors=torch.einsum('bigpc,pax->bigcax',x,bas)
            u=tensors@D[None,None,:,None]
            y=torch.einsum('bigcad,pad->bipc',(u+u.mT)/2,bas)
        out[t]=out[t]+module.gain[PATHS.index(path)]*y
    return out

def run():
    rng=torch.Generator().manual_seed(130370)
    def rand(*sz): return torch.randn(*sz,generator=rng)
    n,w=grid();b,N=2,6;width=(6,5,4);off=(0,6,21,41)
    r=rand(b,N,3).requires_grad_();q=rand(b,N,8).requires_grad_();k=rand(b,N,8).requires_grad_()
    raw=rand(b,N,41).requires_grad_();L,R=factors(r,q,k,n,w)
    report={}; common_weights=None
    for arm in ARMS[1:]:
        m=DirectionalTransport(arm,width,off,n,22).double()
        state={s:v.clone() for s,v in m.named_parameters()}
        if common_weights is None: common_weights=state
        assert state.keys()==common_weights.keys()
        for s in state: torch.testing.assert_close(state[s],common_weights[s],rtol=0,atol=0)
        uu,_=m(raw,L,R);assert torch.count_nonzero(uu)==0
        with torch.no_grad():m.out.weight.copy_(rand(1,48)*.03)
        u,messages=m(raw,L,R)
        src,_=m.value_features(raw); expected=oracle(m,src,r,q,k,n,w)
        max_value=max((a-e).abs().max().item() for a,e in zip(messages,expected))
        projections=[rand(*s.shape) for s in messages]
        def grad(v):return torch.autograd.grad(sum((a*p).sum() for a,p in zip(v,projections)),(r,raw,q,k),create_graph=True,retain_graph=True)
        ga,gb=grad(messages),grad(expected)
        max_first=max((a-e).abs().max().item() for a,e in zip(ga,gb))
        ga2=torch.autograd.grad(ga[0].square().sum(),m.source[2].weight,retain_graph=True)[0]
        gb2=torch.autograd.grad(gb[0].square().sum(),m.source[2].weight,retain_graph=True)[0]
        max_mixed=(ga2-gb2).abs().max().item()
        assert max_value<1e-10 and max_first<1e-9 and max_mixed<1e-8,(arm,max_value,max_first,max_mixed)
        i1,j1=factors(r[:,:1],q[:,:1],k[:,:1],n,w);single,v=m(raw[:,:1],i1,j1)
        assert single.abs().max()<1e-13 and max(x.abs().max() for x in v)<1e-12
        # Reflect and rotate the molecule while keeping the integration grid fixed;
        # any residual is true finite-grid quadrature variation, not changing q/k.
        rot=torch.linalg.qr(rand(3,3)).Q; rot[:,0]*=-torch.det(rot)
        setr=raw.split((6,15,20),-1); B=stf_basis()
        vr=raw.new_empty(raw.shape)
        vr[:,:,:6]=setr[0]
        vr[:,:,6:21]=torch.einsum('uv,bnvc->bnuc',rot,setr[1].reshape(b,N,3,5)).flatten(-2)
        vr[:,:,21:]=torch.einsum('pad,ax,dy,qxy,bnqc->bnpc',B,rot,rot,B,setr[2].reshape(b,N,5,4)).flatten(-2)
        lr,rr=factors(r@rot.T+.7,q,k,n,w);ur,_=m(vr,lr,rr)
        equiv=(ur-u).abs().max().item()
        assert equiv<2e-5,(arm,equiv)
        report[arm]=dict(paths=len(m.paths),value_dense_error=max_value,grad_dense_error=max_first,
                         mixed_grad_dense_error=max_mixed,active_R_improper_O3_energy_error=equiv)
    return report

if __name__=='__main__':
    out=run()
    if len(sys.argv)>1: Path(sys.argv[1]).write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
