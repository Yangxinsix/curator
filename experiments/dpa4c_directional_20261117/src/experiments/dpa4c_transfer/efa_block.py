"""PyTorch form of the authors' molecular EFA operator (l<=2, L_Y=0)."""
from math import pi,sqrt
import torch
from torch import nn
from curator.layer.efa.core import _lebedev_grid

class OriginalEFA(nn.Module):
    def __init__(self, rep):
        super().__init__()
        self.widths=tuple(rep.degree_channels[:3]); self.offsets=tuple(rep.moment_offsets)
        def projections(n):
            return nn.ModuleList(nn.Linear(w,n,bias=l==0) for l,w in enumerate(self.widths))
        self.query=projections(32);self.key=projections(32);self.value=projections(16)
        self.out=nn.ModuleList(nn.Linear(16,w,bias=False) for w in self.widths)
        for a in self.out:nn.init.zeros_(a.weight)
        u,w=_lebedev_grid(50);self.register_buffer('grid',u);self.register_buffer('weights',w)
        self.register_buffer('freq',torch.linspace(0,pi,16,dtype=torch.float64)/15)

    @staticmethod
    def gated(x):
        # e3x.nn.gelu(approximate=True): the even scalar gates ALL m and l.
        a=x[:,:,:1];return x*(.5*(1+torch.tanh(sqrt(2/pi)*(a+.044715*a**3))))

    def attention(self,q,k,v,pos):
        # input [batch,n,9,features]. Molecular segments remain completely separate.
        q,k=self.gated(q),self.gated(k)
        phase=(pos@self.grid.T)[:,:,:,None,None]*self.freq
        c,s=phase.cos(),phase.sin()
        def rotate(a):
            a0,a1=a[:,:,None,:,0::2],a[:,:,None,:,1::2]
            return torch.stack((a0*c-a1*s,a0*s+a1*c),-1).flatten(2)
        # Lebedev weights repeat over component and Q/K channel, never over atoms.
        qrot=rotate(q).reshape(*q.shape[:2],50,-1)
        left=(qrot*(self.weights/(q.shape[-2]*q.shape[-1])**.5)[None,None,:,None]).flatten(2)
        right=rotate(k)
        return (left@(right.transpose(1,2)@v.flatten(2))).reshape_as(v)

    def moments(self,raw,pos):
        b,n=pos.shape[:2]
        blocks=[raw[:,self.offsets[l]:self.offsets[l+1]].reshape(b,n,2*l+1,w) for l,w in enumerate(self.widths)]
        def proj(layers):return torch.cat([layer(x) for layer,x in zip(layers,blocks)],dim=2)
        y=self.attention(proj(self.query),proj(self.key),proj(self.value),pos)
        updates=[a(chunk).reshape(b*n,-1) for a,chunk in zip(self.out,y.split([1,3,5],dim=2))]
        return torch.cat([raw[:,:self.offsets[3]]+torch.cat(updates,dim=-1),raw[:,self.offsets[3]:]],-1)

    def energy(self,rep,d,p):
        count=d[p.n_atoms].numel();raw=d[p.node_embedding];pos=d[p.positions].reshape(count,-1,3)
        updated=self.moments(raw,pos)
        def pieces(t):
            inv,scalar,quartic=rep._invariants(t);return scalar,torch.cat([*inv,quartic],dim=-1)
        s0,i0=pieces(raw);s1,i1=pieces(updated)
        desc=d[p.node_feat];embed=rep.element_embedding.weight.shape[-1]
        difference=torch.cat((torch.zeros_like(desc[:,:embed]),s1-s0,torch.zeros_like(desc[:,:2]),i1-i0),-1)
        # Preserve the trained native descriptor including its stored calibration.
        atom=rep._atomic_readout(desc+difference*rep.calibration_inv_std,d[p.atomic_types])
        return atom.reshape(count,-1).sum(1)
