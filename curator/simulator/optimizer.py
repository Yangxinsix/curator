from ase.optimize import BFGS, FIRE,  BFGS, BFGSLineSearch, LBFGS, LBFGSLineSearch
import ase
from pathlib import Path
def get_optimizer(optimizer: str, atom: ase.Atom, traj_path: Path, log_path: Path) -> ase.optimize.Optimizer:
    """Get optimizer from ASE.

    Args:
        optimizer (str): optimizer name
        atom (ase.Atom): atom object
        traj_path (Path): trajectory file path
        log_path (Path): log file path
    
    Returns:
        opt (ase.optimize.Optimizer): optimizer object
    """
    
    if optimizer == 'BFGS':
        opt = BFGS(atom,trajectory=traj_path,logfile=log_path)
    elif optimizer == 'FIRE':
        opt = FIRE(atom,trajectory=traj_path,logfile=log_path)
    elif optimizer == 'BFGSLineSearch':
        opt = BFGSLineSearch(atom,trajectory=traj_path,logfile=log_path)
    elif optimizer == 'LBFGS':
        opt = LBFGS(atom,trajectory=traj_path,logfile=log_path)
    elif optimizer == 'LBFGSLineSearch':
        opt = LBFGSLineSearch(atom,trajectory=traj_path,logfile=log_path)
    else:
        raise RuntimeError('Optimizer not found!')
    return opt