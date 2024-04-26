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
from ase.io import read, write


#### Generate the NEB images ####
lifepo4 = read('LiFePO4_olivine.cif')

# Make the system larger
atoms = lifepo4*(1,2,2)
print(atoms)
concentration = 0.75 # total amounth of Li in the system
M = 'Li'

# Setup random seed
np.random.seed(11)

# Find index and position for Li
M_indices = [atom.index for atom in atoms if atom.symbol == M]
position_M = [atom.position for atom in atoms if atom.symbol == M]

# Find the atom to be moved and sort the indies according to the distance from that atom
neb_index = np.random.randint(0,len(M_indices),size=1)[0]
diff_neb = np.linalg.norm(position_M-position_M[neb_index],axis=1)
M_indices_neb = [x for _,x in sorted(zip(diff_neb,M_indices))]
# Change the symbol of the two atoms to be moved
print('Li to be moved:',M_indices_neb[0],M_indices_neb[1])
init_img,final_img = atoms.copy(), atoms.copy()
del init_img[M_indices_neb[0]]
del final_img[M_indices_neb[1]]

# find the rest of the atoms to be removed
Li_rm = int(len(M_indices_neb)*(1-concentration)) -1 # -1 to account for the atom that is moved
rm_index = np.random.choice(np.array(M_indices_neb)[2:],size=Li_rm)
print('Li to be removed:',rm_index)
# Remove the atoms
del init_img[rm_index]
del final_img[rm_index]

# Write to traj
write('NEB_init_pristine.traj',init_img)
write('NEB_final_pristine.traj',final_img)

# If you want to visualize the images
write('NEB_pristine.traj',[init_img,final_img])


### Relax the NEB images ###
params_GPAW = {}
params_GPAW['mode']        = {'name': 'pw', 'ecut': 520}                    # The used plane wave energy cutoff
params_GPAW['kpts']        = {'size': (4, 4, 4),          # The k-point mesh
                              'gamma': True}
params_GPAW['xc']          = 'PBE'                   # The used exchange-correlation functional
params_GPAW['occupations'] = FermiDirac(0.05, fixmagmom=True)        # The smearing with fixed magnetic moments
params_GPAW['convergence'] = {'eigenstates': 1,      # eV^2 / electron
                              'energy':      1.0e-5,      # eV / electron
                              'density':     1}
params_GPAW['setups']      = {'Fe': ':d,4.3'}             # U=4.3 applied to d orbitals
params_GPAW['eigensolver'] = {'name': 'dav', 'niter':1}

# We save the GPAW parameters to a toml file
toml.dump(params_GPAW, open('params_NEB_GPAW.toml', 'w'))



# Set up calulator for the initial image
calc = GPAW(txt=f'NEB_init.txt',**params_GPAW)
init_img.set_calculator(calc)

# Use the QuasiNewton optimizer to optimize the system
opt = QuasiNewton(init_img, trajectory=f'NEB_init.traj',
                    logfile=f'NEB_init.log')
opt.run(fmax=0.03)

# Set up calulator for the final image
calc = GPAW(txt=f'NEB_final.txt',**params_GPAW)
final_img.set_calculator(calc)
# Use the QuasiNewton optimizer to optimize the system
opt = QuasiNewton(final_img, trajectory=f'NEB_final.traj',
                    logfile=f'NEB_final.log')
opt.run(fmax=0.03)