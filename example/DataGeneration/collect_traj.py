from ase.io import read, write

# Define the samples to collect into one file
# Here we collect the optimization steps for LiFePO4 structures when 100%, 75% and 0% Lithiated as well as the MD trajectory of these
traj_to_collect = ['LiFePO4_relax_0.traj','LiFePO4_relax_1.traj','LiFePO4_relax_4.traj','LiFePO4_MD_0.traj','LiFePO4_MD_1.traj','LiFePO4_MD_4.traj']

# Read the structures and write them to a new file
traj = []
for traj_file in traj_to_collect:
    atoms = read(traj_file,':')
    for atom in atoms:
        traj.append(atom)
write('init_dataset.traj',traj)