"""DPA4C-only vs DPA4C + the original molecular EFA operator. No ACVR head."""
import argparse,csv,json,sys,time
from pathlib import Path
import numpy as np
import torch
from ase.io.trajectory import Trajectory
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'dpa4c_tensor'))
import train_runner as old
from equilibrium import model_inputs
from curator.data import properties as p
from scan import angle_tangent
from efa_block import OriginalEFA

def predict(model,d,create=False):
    d=d.copy();d[p.positions].requires_grad_(True)
    for layer in model.input_modules:d=layer(d)
    d=model.representation(d)
    e=model.output_modules[0].energy(model.representation,d,p) if model.output_modules else d[p.energy]
    f,=torch.autograd.grad(-e.sum(),d[p.positions],create_graph=create)
    return e,f

def score(model,ds,split,out,step):
    model.double().eval()
    for fold in ('validation','test'):
        accum=np.zeros(4); collate=ds.val_dataloader().collate_fn; idx=split[fold]
        for j in range(0,len(idx),16):
            items=[ds.datasets[fold][t] for t in range(j,min(j+16,len(idx)))]
            assert [int(x.meta['source_index']) for x in items]==idx[j:j+len(items)]
            b=collate(items);d=model_inputs(b,'cuda');e,f=predict(model,d);lab=b['variational']['labels']
            de=(e.detach().cpu()-lab['condition_energy']['values'].flatten())/b[p.n_atoms]
            df=f.detach().cpu().reshape(-1)-lab['condition_forces']['values'].reshape(-1)
            accum+=np.array([de.square().sum().item(),len(de),df.square().sum().item(),df.numel()])
        old.write_json(out/f'{fold}-{step}.json',dict(frames=int(accum[1]),energy_RMSE_meV_atom=1000*np.sqrt(accum[0]/accum[1]),force_RMSE_meV_A=1000*np.sqrt(accum[2]/accum[3])))
    rows=[]
    with Trajectory(str(old.DEFAULT_SCAN)) as traj:
        assert len(traj)==1000
        for i in range(0,len(traj),16):
            at=[traj[j] for j in range(i,min(i+16,len(traj)))];b=len(at);n=len(at[0])
            d={p.positions:torch.as_tensor(np.concatenate([a.positions for a in at]),device='cuda'),
               p.atomic_numbers:torch.tensor(np.concatenate([a.numbers for a in at]),device='cuda'),
               p.n_atoms:torch.full((b,),n,device='cuda',dtype=torch.long),
               p.image_idx:torch.arange(b,device='cuda').repeat_interleave(n),
               p.cell:torch.zeros(b*3,3,dtype=torch.float64,device='cuda'), p.pbc:torch.zeros(b,3,dtype=torch.bool,device='cuda')}
            e,f=predict(model,d);e=e.detach().cpu().numpy();f=f.detach().cpu().numpy().reshape(b,n,3)
            for j,a in enumerate(at):
                assert a.info['source_frame']==i+j and not a.pbc.any()
                angle,t=angle_tangent(a.numbers,a.positions)
                rows.append(dict(index=i+j,angle_deg=angle,E_pred=e[j],E_ref=a.get_potential_energy(),torque_pred=(f[j]*t).sum(),torque_ref=(a.get_forces()*t).sum()))
    hi=[r for r in rows if 150<=r['angle_deg']<=175];assert len(hi)==277
    with (out/f'scan-{step}.csv').open('w',newline='') as fp:
        w=csv.DictWriter(fp,rows[0]);w.writeheader();w.writerows(rows)
    old.write_json(out/f'torsion-{step}.json',dict(frames=len(hi),scan_sha256=old.sha(old.DEFAULT_SCAN),
        torque_RMSE_eV_rad=float(np.mean([(r['torque_pred']-r['torque_ref'])**2 for r in hi])**.5),
        barrier_eV=float(rows[-1]['E_pred']-rows[0]['E_pred']),DFT_barrier_eV=float(rows[-1]['E_ref']-rows[0]['E_ref'])))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=('pure','original_efa'),required=True);ap.add_argument('--out-root',type=Path,required=True)
    ap.add_argument('--steps',type=int,default=40000);ap.add_argument('--parent',type=Path,default=old.DEFAULT_SOURCE/'cumulene/3407/pure')
    a=ap.parse_args();old.configure();ds,split,digest=old.load_data('cumulene');out=a.out_root/a.arm;out.mkdir(parents=True,exist_ok=True)
    ck=a.parent/'ckpt-020000.pt';cp=torch.load(ck,weights_only=False,map_location='cpu');st=torch.load(a.parent/'resume.pt',weights_only=False,map_location='cpu')
    ci=cp['identity'];assert (ci['system'],ci['arm'],ci['seed'],ci['split_sha256'],cp['joint_step'],st['step'])==('cumulene','pure',3407,digest,20000,20000)
    assert ci==st['identity'] and old.sha(ck)==json.loads((a.parent/'test_final.json').read_text())['checkpoint_sha256']
    assert list(cp['model'])==list(st['model']) and all(torch.equal(v,st['model'][k]) for k,v in cp['model'].items() if torch.is_tensor(v))
    model=old.build('cumulene',list(ds.sources[0].species),'pure',3407,'B2').cuda().float();model.load_state_dict(cp['model'],strict=True)
    torch.manual_seed(3407)
    model.output_modules=torch.nn.ModuleList([OriginalEFA(model.representation).cuda().float()] if a.arm=='original_efa' else [])
    opt=torch.optim.Adam(model.representation.parameters(),lr=1e-4)
    group=st['adam']['param_groups'][0];opt.load_state_dict(dict(param_groups=[group],state={k:v for k,v in st['adam']['state'].items() if k in group['params']}))
    assert group['lr']==1e-4 and all(stv['exp_avg'].shape==pr.shape for pr,stv in opt.state.items() if 'exp_avg' in stv)
    if model.output_modules:opt.add_param_group(dict(params=model.output_modules.parameters(),lr=1e-5))
    ident=dict(arm=a.arm,split_sha256=digest,source_checkpoint_sha256=old.sha(ck),source_resume_sha256=old.sha(a.parent/'resume.pt'),joint_start=20000,add_steps=a.steps,batch=4,seed=3407,loss='200 * (per_atom_energy_mse + component_force_mse)')
    old.write_json(out/'config.json',ident);st={};cp={};stream=old.Stream(ds,3407+300000)
    done=0;elapsed=0
    if (out/'resume.pt').is_file():
        state=torch.load(out/'resume.pt',map_location='cpu',weights_only=False);assert state['identity']==ident
        model.load_state_dict(state['model']);opt.load_state_dict(state['adam']);done=state['step'];elapsed=state['seconds']
    initial_e=None
    if done==0 and model.output_modules:
        d=model_inputs(stream.get(20000)[0],'cuda')
        e1,f1=predict(model,d);head=model.output_modules;model.output_modules=torch.nn.ModuleList()
        e0,f0=predict(model,d);model.output_modules=head
        torch.testing.assert_close(e1,e0,rtol=3e-6,atol=2e-5);torch.testing.assert_close(f1,f0,rtol=3e-5,atol=2e-5)
        initial_e=dict(max_E=float((e1-e0).detach().abs().max()),max_F=float((f1-f0).detach().abs().max()));old.write_json(out/'start_parity.json',initial_e)
    t0=time.monotonic();model.train()
    for j in range(done+1,a.steps+1):
        opt.zero_grad(set_to_none=True)
        batches=stream.get(20000+j-1);assert len(batches)==1
        b=batches[0];d=model_inputs(b,'cuda');e,f=predict(model,d,True);lab=b['variational']['labels']
        de=(e-lab['condition_energy']['values'].flatten().to(e))/d[p.n_atoms]
        df=f.reshape(-1)-lab['condition_forces']['values'].flatten().to(f)
        loss=200*(de.square().mean()+df.square().mean());loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True);opt.step()
        if j==1 or j%100==0:
            torch.cuda.synchronize();old.write_json(out/'status.json',dict(joint_step=20000+j,physical_loss=loss.item(),training_seconds=elapsed+time.monotonic()-t0))
        if j in (20000,a.steps):
            old.atomic_save(dict(identity=ident,joint_step=20000+j,model=model.state_dict()),out/f'ckpt-{20000+j}.pt')
        if j%1000==0 or j==a.steps:
            old.atomic_save(dict(identity=ident,step=j,model=model.state_dict(),adam=opt.state_dict(),seconds=elapsed+time.monotonic()-t0),out/'resume.pt')
    for step in sorted({20000+a.steps,40000}):
        path=out/f'ckpt-{step}.pt'
        if path.exists():
            model.double();model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model']);score(model,ds,split,out,step)
    old.write_json(out/'complete.json',dict(identity=ident,final=20000+a.steps))
if __name__=='__main__':main()
