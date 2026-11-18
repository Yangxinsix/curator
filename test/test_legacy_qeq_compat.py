import copy
import torch
from curator.data import properties
from curator.layer import PairwiseDistance
from curator.layer._atomwise_nn import MACEAtomwiseNN
from curator.layer._legacy_charge_equilibration import LegacyChargeEquilibration
from curator.model.legacy_qeq import append_legacy_charge_readout


def test_legacy_readout_preserves_pretrained_energy_and_shared_trunk():
    torch.manual_seed(5)
    original=MACEAtomwiseNN(num_interactions=2,hidden_irreps='4x0e+4x1o',MLP_irreps='2x0e',out_features=1,heads=['energy']).double()
    expanded=append_legacy_charge_readout(original)
    features=torch.randn(3,20,dtype=torch.float64,requires_grad=True)
    before=original._compute(features)[:,0]
    after=expanded._compute(features)
    torch.testing.assert_close(before,after[:,0],rtol=1e-12,atol=1e-12)
    a=torch.autograd.grad(before.sum(),features,retain_graph=True)[0]
    b=torch.autograd.grad(after[:,0].sum(),features,retain_graph=True)[0]
    torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)
    assert after[:,1].square().sum()>0
    assert expanded.readouts[-1] is expanded.readout_mlp


def test_legacy_energy_force_and_residual_match_explicit_formula():
    torch.manual_seed(9)
    layer=LegacyChargeEquilibration(num_features=4,compute_residual_forces_mode='always').double()
    pos=torch.tensor([[0.3,0.5,0.7],[2.1,1.5,1.2],[1.2,2.5,2.2]],dtype=torch.float64,requires_grad=True)
    raw=torch.sin(pos[:,0]) * .1
    features=torch.randn(3,4,dtype=torch.float64)
    data={properties.positions:pos,properties.atomic_numbers:torch.tensor([1,8,1]),properties.node_embedding:features,
          properties.atomic_charge:raw,properties.image_idx:torch.zeros(3,dtype=torch.long),properties.n_atoms:torch.tensor([3]),
          properties.total_charge:torch.tensor([.2],dtype=torch.float64),properties.cell:torch.eye(3,dtype=torch.float64)*6,
          properties.pbc:torch.ones(1,3,dtype=torch.bool),properties.energy:pos.square().sum().reshape(1),properties.forces:-2*pos}
    data=PairwiseDistance(compute_neighbor_list=True,compute_distance_from_R=True,compute_forces=True,cutoff=6.0)(data)
    q=raw+(.2-raw.sum())/3
    chi=layer.electronegativity_mlp._compute(features).squeeze().square()
    eta=layer.hardness_mlp._compute(features).squeeze().square()
    local=(chi*q+eta*q.square()).sum()
    ewald=layer.ewald({**data,properties.atomic_charge:q}).sum()
    fixed=layer.ewald({**data,properties.atomic_charge:q.detach()}).sum()
    fixed_grad=torch.autograd.grad(fixed,pos,create_graph=True,retain_graph=True)[0]
    response=torch.autograd.grad(local+ewald,pos,create_graph=True,retain_graph=True)[0]-fixed_grad
    result=layer(data)
    torch.testing.assert_close(result[properties.atomic_charge],q)
    torch.testing.assert_close(result[properties.energy],data[properties.energy]+local+ewald)
    torch.testing.assert_close(result[properties.forces],-2*pos-fixed_grad)
    torch.testing.assert_close(result[properties.residual_forces],response)
    loss=result[properties.forces].square().mean()+result[properties.residual_forces].square().mean()
    grads=torch.autograd.grad(loss,list(layer.parameters()),allow_unused=True)
    assert all(torch.isfinite(g).all() for g in grads if g is not None)
