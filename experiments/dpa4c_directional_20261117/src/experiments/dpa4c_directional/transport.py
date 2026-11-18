"""Directional transport of geometric values through the existing molecular ERoPE factors.

All public interfaces use packed DPA4C moments, ordered (component, channel). The
source/receiver chemical projections do not use q, a fragment id or a torsion.
The only input growing with atom count in transport is O(N*G*D), never O(N**2).
"""
import math

import torch
from torch import nn

ARMS = ('B0', 'B1', 'T20', 'T02', 'T22', 'V11', 'O1', 'J2', 'B1H', 'J2H')
# (source rank, displacement rank, output rank), in checkpoint-stable order
BASE = ((0,0,0), (1,0,1), (2,0,2))
EVEN = ((2,2,0), (0,2,2), (2,2,2), (1,2,1))
ODD = ((0,1,1), (1,1,0), (1,1,2), (2,1,1))
PATHS = BASE + EVEN + ODD
ACTIVE = {'B0': (), 'B1': BASE, 'B1H': BASE, 'J2': PATHS, 'J2H': PATHS,
          'O1': BASE + ODD, **{a: BASE + (p,) for a,p in zip(('T20','T02','T22','V11'), EVEN)}}


def stf_basis(dtype=torch.float64):
    """Isometry from DPA4C packed l=2 coefficients to Cartesian STF tensors."""
    b = torch.zeros(5,3,3,dtype=dtype)
    f = 1/math.sqrt(2)
    for p,i,j in ((0,0,1),(1,1,2),(3,0,2)):
        b[p,i,j] = b[p,j,i] = f
    b[2] = torch.diag(torch.tensor([-1.,-1.,2.],dtype=dtype))/math.sqrt(6)
    b[4] = torch.diag(torch.tensor([f,-f,0.],dtype=dtype))
    return b


def angular_matrices(n):
    """Per-direction real O(3) maps, normalization built from Cartesian contractions."""
    g = len(n)
    eye = torch.eye(3,dtype=n.dtype,device=n.device)
    D = n[:,:,None]*n[:,None,:] - eye/3
    B = stf_basis(n.dtype).to(n.device)
    d = torch.einsum('gij,pij->gp',D,B)
    Tn = torch.einsum('pij,gj->gpi',B,n)
    return {
        **{(l,0,l): torch.eye(2*l+1,dtype=n.dtype,device=n.device).expand(g,-1,-1) for l in range(3)},
        (0,2,2):d[:,None,:], (2,2,0):d[:,:,None], (1,2,1):D,
        # Isometric projection of STF((D T + T D)/2).
        (2,2,2):torch.einsum('gij,pjk,qki->gpq',D,B,B),
        (0,1,1):n[:,None,:], (1,1,0):n[:,:,None],
        (1,1,2):Tn.transpose(1,2), (2,1,1):Tn,
    }


def quarter_turn(q):
    # q=(a,b) -> (b,-a); q'·k at opposite directions gives sin(phi_i-phi_j)
    # times the unrotated content dot product, after antisymmetric angular weighting.
    return torch.stack((q[...,1::2],-q[...,0::2]),dim=-1).flatten(-2)


class DirectionalTransport(nn.Module):
    def __init__(self, arm, widths, offsets, grid, seed, channels=4):
        super().__init__()
        if arm not in ACTIVE or arm=='B0':
            raise ValueError(f'{arm}: a directional module is only installed on non-B0 arms')
        if grid.ndim != 2 or grid.shape[1] != 3 or grid.shape[0]<2:
            raise ValueError('molecular grid [G,3] required')
        self.arm = arm
        self.widths, self.offsets = tuple(widths[:3]), tuple(offsets[:4])
        self.channels = channels
        self.ng = len(grid)
        self.paths = ACTIVE[arm]
        # All arms draw the very same parameters on the CPU (including inactive gains).
        # Do not change a global CUDA generator or seed a CPU generator after .to(cuda).
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(seed))
            self.source = nn.ModuleList(nn.Linear(w, channels, bias=False) for w in self.widths)
            self.receiver = nn.ModuleList(nn.Linear(w, channels, bias=False) for w in self.widths)
            self.gain = nn.Parameter(torch.ones(len(PATHS), channels))
            self.hidden = nn.Linear(6 * channels, 48)
            self.out = nn.Linear(48,1,bias=False)
            nn.init.zeros_(self.out.weight)
        # Fused readout contraction: compute the matrix acting on [G,9] only once.
        # This solution uses the restored parent frequency weights in Q/K, not a
        # separate learned spectral envelope. Gains and chemical projections learn.
        directions = grid.detach().cpu().double()
        couplings = angular_matrices(directions)
        for parity in (0,1):
            operators = []
            group_paths = tuple(p for p in self.paths if p[1]%2==parity and p[1]>0)
            for s,L,t in group_paths:
                a = directions.new_zeros(self.ng,9,2*t+1)
                a[:,s*s:(s+1)**2,:] = couplings[s,L,t]
                operators.append(a.reshape(self.ng*9,-1))
            self.register_buffer(f'op_{parity}',torch.cat(operators,dim=-1) if operators else
                                 directions.new_empty(self.ng*9,0),persistent=False)
            setattr(self,f'extra_{parity}',group_paths)

    def value_features(self, raw):
        moments=tuple(raw[...,self.offsets[l]:self.offsets[l+1]].reshape(
            *raw.shape[:-1],2*l+1,self.widths[l]) for l in range(3))
        return tuple(w(x) for w,x in zip(self.source,moments)), tuple(w(x) for w,x in zip(self.receiver,moments))

    def transport(self, values, left, right):
        """[B,N,(1,3,5),C], [B,N,G*D] -> tuple of the same three value shapes."""
        b,n,c = values[0].shape[0],values[0].shape[1],self.channels
        x=torch.cat(values,dim=-2).reshape(b,n,9*c)
        if left.shape != right.shape or left.shape[:2] != (b,n) or left.shape[-1]%(2*self.ng):
            raise ValueError('wrong molecular Q/K flattened factor layout')
        # Same-rank identity part is exactly the original fast transport (self-excluded).
        # When angular paths are active, reuse their K-transformed source for L=0.
        q=left.reshape(b,n,self.ng,-1).transpose(1,2)
        k=right.reshape(b,n,self.ng,-1).transpose(1,2)
        if not self.extra_0 and not self.extra_1:
            plain=left@(right.mT@x)-(left*right).sum(-1,keepdim=True)*x
        else:
            kv=k.mT@x[:,None]  # [B,G,D,9*C], independent of N
            ev=q@kv-(q*k).sum(-1,keepdim=True)*x[:,None]
            plain=ev.sum(1)
        msg=list(plain.reshape(b,n,9,c).split((1,3,5),dim=-2))
        msg=[v*self.gain[l] for l,v in enumerate(msg)]
        for parity in (0,1):
            routes=getattr(self,f'extra_{parity}')
            if not routes: continue
            if parity:
                qs=quarter_turn(q)
                field=qs@kv-(qs*k).sum(-1,keepdim=True)*x[:,None]
            else:
                field=ev
            # Angular map is linear, so we can sum over the ERoPE feature width D
            # before applying it. The per-direction G axis is retained throughout.
            field=field.reshape(b,self.ng,n,9,c).permute(0,2,4,1,3).reshape(b,n,c,-1)
            op=getattr(self,f'op_{parity}').to(field)
            routed=(field@op).transpose(-1,-2).split(tuple(2*t+1 for s,L,t in routes),dim=-2)
            for path,contribution in zip(routes,routed):
                t=path[2]
                msg[t]=msg[t]+contribution*self.gain[PATHS.index(path)]
        return tuple(msg)

    def forward(self, raw, left, right):
        src,local=self.value_features(raw)
        msg=self.transport(src,left,right)
        # Remote-zero subtraction ensures the residual cannot learn a second purely
        # local energy. Independent receiver projections + nonlinear local conditioning
        # avoid imposing the linear T20/T02 transpose-energy identity.
        loc=local[0].squeeze(-2)
        remote=torch.cat((msg[0].squeeze(-2), (msg[1]*local[1]).sum(-2),
                          (msg[2]*local[2]).sum(-2), msg[1].square().sum(-2).log1p(),
                          msg[2].square().sum(-2).log1p()),dim=-1)
        f=torch.cat((loc,remote),-1)
        f0=torch.cat((loc,torch.zeros_like(remote)),-1)
        delta=self.out(torch.nn.functional.silu(self.hidden(f))-
                       torch.nn.functional.silu(self.hidden(f0))).sum((-2,-1))
        return delta,msg
