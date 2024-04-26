import numpy as np
from ase.visualize import view
import matplotlib.pyplot as plt
from ase.io import read, write, Trajectory
from ase.parallel import paropen
from gpaw import GPAW, Davidson, FermiDirac
from ase.dft.bee import BEEFEnsemble
from ase import Atoms
from ase.optimize import BFGS, QuasiNewton
import toml
from ase.io import read

params_GPAW = {}
params_GPAW['mode']        = {'name': 'pw', 'ecut': 520}                    # The used plane wave energy cutoff
params_GPAW['kpts']        = {'size': (4, 4, 4),          # The k-point mesh
                              'gamma': True}
params_GPAW['xc']          = 'PBE'                   # The used exchange-correlation functional
params_GPAW['occupations'] = FermiDirac(0.05, fixmagmom=True)    # The smearing with fixed magnetic moments
params_GPAW['convergence'] = {'eigenstates': 1,      # eV^2 / electron
                              'energy':      1.0e-5,      # eV / electron
                              'density':     1}
params_GPAW['setups']      = {'Fe': ':d,4.3'}             # U=4.3 applied to d orbitals
params_GPAW['eigensolver'] = {'name': 'dav', 'niter':1} # The used eigenvalue solver

# We save the GPAW parameters to a toml file
toml.dump(params_GPAW, open('params_GPAW.toml', 'w'))

lifepo4 = read('LiFePO4_olivine.cif')

n_Li = len(lifepo4.symbols=='Li')
fmax = 0.03
# Loop over all Li atoms and remove each one at a time
for i in range(n_Li+1):# +1 to take account for fully lithiated structure
    if i == 0 or i == 1 or i == 2:
        continue
    if i == 0:
        lifepo4_copy = lifepo4.copy()
        calc = GPAW(txt=f'LiFePO4_relax{i}.txt',**params_GPAW)
        lifepo4_copy.calc = calc
        print(lifepo4_copy, lifepo4_copy.get_potential_energy())
        # Use the BFGS optimizer to optimize the system
        opt = QuasiNewton(lifepo4_copy, trajectory=f'LiFePO4_relax{i}.traj',
                    logfile=f'LiFePO4_relax{i}.log')
        opt.run(fmax=fmax)
    else:
        lifepo4_copy = lifepo4.copy()
        Li_indices = [atom.index for atom in lifepo4_copy if atom.symbol == 'Li']
        
        # Remove i number of Li atom
        for j in range(i):
            del lifepo4_copy[Li_indices[j]]
        calc = GPAW(txt=f'LiFePO4_relax_{i}.txt',**params_GPAW)
        lifepo4_copy.calc = calc
        print(lifepo4_copy, lifepo4_copy.get_potential_energy())
        # Use the BFGS optimizer to optimize the system
        opt = QuasiNewton(lifepo4_copy, trajectory=f'LiFePO4_relax_{i}.traj',
                    logfile=f'LiFePO4_relax_{i}.log')
        opt.run(fmax=fmax)
