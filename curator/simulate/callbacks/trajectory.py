from __future__ import annotations

from typing import Optional, List, Any, Dict, Sequence, Union, Callable
from ase.io import Trajectory
from ase import Atoms
from ..core.callbacks import Callback
from ..core.context import SimContext
from curator.data import properties
import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)

try:  # optional dependency
    import torch_sim as ts
except Exception:  # pragma: no cover
    ts = None


class TrajectoryWriter(Callback):
    """
    Write structures to trajectory files every ``interval`` steps.

    - Works with ASE atoms or torch-sim SimState (via ctx.state['sim_state']).
    - Can write a single combined file or per-system files (using `{i}` in path or auto suffix).
    - Supports extracting model outputs (e.g., atomic_charge) and storing them in atoms.info.
    
    Parameters
    ----------
    path : str
        Path to the trajectory file. Use `{i}` for per-system files.
    interval : int
        Write every `interval` steps.
    mode : str
        File mode ('w' for write, 'a' for append).
    per_system : bool
        If True, write separate files for each system.
    store_model_outputs : Dict[str, str] or List[str], optional
        Model outputs to store in trajectory. Can be:
        - List of property names: ["atomic_charge", "energy", "maha_dist", "temperature"] 
          → stored as atoms.info[name]
        - Dict mapping property names to custom keys: {"atomic_charge": "charges"}
          → stored as atoms.info["charges"]
        
        Data sources (in priority order):
        1. atoms.calc.results
        2. ctx.engine.model.step_outputs (for model outputs like atomic_charge)
        3. ctx.state dict (for uncertainty, thermo, and custom callback data)
        4. ctx.state["sim_state"] attributes
        5. Computed thermo properties via TorchSimThermoLogger (temp, ekin, pressure, etc.)
        
    Note
    ----
    All model outputs are stored in atoms.info to ensure they are preserved,
    since ASE Trajectory only saves standard arrays (positions, numbers) but 
    preserves the complete info dict.
    """

    # Mapping from user-friendly names to TorchSimThermoLogger method names
    THERMO_PROPERTY_MAP = {
        "temperature": "temp",
        "temp": "temp",
        "kinetic_energy": "ekin",
        "ekin": "ekin",
        "potential_energy": "epot",
        "epot": "epot",
        "total_energy": "etot",
        "etot": "etot",
        "pressure": "pressure",
        "stress": "stress",
        "volume": "volume",
        "density": "density",
        "natoms": "natoms",
    }

    def __init__(
        self, 
        path: str, 
        interval: int = 1, 
        mode: str = "w", 
        per_system: bool = False,
        store_model_outputs: Optional[Union[Dict[str, str], List[str], str]] = None,
    ):
        self.path = path
        self.interval = max(1, int(interval))
        self.mode = mode
        self.per_system = per_system
        self._trajs: List[Trajectory] = []
        self._thermo_helper = None  # Lazy init for thermo calculations
        
        # Parse store_model_outputs: {property_name: store_key_name}
        self._output_map = self._parse_store_model_outputs(store_model_outputs)

    @staticmethod
    def _parse_store_model_outputs(store_model_outputs) -> Dict[str, str]:
        """Parse store_model_outputs into a dict mapping property_name -> store_key."""
        if store_model_outputs is None:
            return {}
        
        # Handle string input
        if isinstance(store_model_outputs, str):
            return {store_model_outputs: store_model_outputs}
        
        # Handle dict-like (including OmegaConf DictConfig)
        if hasattr(store_model_outputs, 'items'):
            return {str(k): str(v) for k, v in store_model_outputs.items()}
        
        # Handle list-like (including OmegaConf ListConfig)
        if hasattr(store_model_outputs, '__iter__'):
            result = {}
            for item in store_model_outputs:
                if isinstance(item, str):
                    result[item] = item
                elif hasattr(item, 'items'):
                    # Dict item in list
                    for k, v in item.items():
                        result[str(k)] = str(v)
                elif hasattr(item, '__iter__') and not isinstance(item, str):
                    # Tuple/list like (key, value)
                    item_list = list(item)
                    if len(item_list) == 2:
                        result[str(item_list[0])] = str(item_list[1])
                    else:
                        result[str(item_list[0])] = str(item_list[0])
                else:
                    result[str(item)] = str(item)
            return result
        
        # Fallback: treat as single item
        return {str(store_model_outputs): str(store_model_outputs)}

    def _to_numpy(self, value: Any) -> np.ndarray:
        """Convert tensor or array-like to numpy array."""
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        elif isinstance(value, np.ndarray):
            return value
        elif hasattr(value, '__array__'):
            return np.asarray(value)
        else:
            return np.array(value)

    def _get_thermo_helper(self):
        """Lazily initialize TorchSimThermoLogger for thermo calculations."""
        if self._thermo_helper is None:
            try:
                from .torchsim_logger import TorchSimThermoLogger
                # Create a minimal instance just for calculations (no logging)
                self._thermo_helper = TorchSimThermoLogger(
                    variables=[],
                    header=False,
                    interval=1,
                )
            except ImportError:
                pass
        return self._thermo_helper

    def _compute_thermo_property(self, prop_name: str, ctx: SimContext, sys_idx: int = 0) -> Optional[Any]:
        """Compute thermo properties using TorchSimThermoLogger methods."""
        thermo = self._get_thermo_helper()
        if thermo is None:
            return None
        
        # Map user-friendly name to method name
        method_name = self.THERMO_PROPERTY_MAP.get(prop_name)
        if method_name is None:
            return None
        
        # Get the method
        method = thermo.variable_funcs.get(method_name)
        if method is None:
            return None
        
        try:
            return method(ctx, idx=sys_idx)
        except Exception as e:
            logger.debug(f"Failed to compute thermo property {prop_name}: {e}")
            return None

    def _attach_model_outputs(self, atoms: Atoms, ctx: SimContext, sys_idx: int = 0) -> Atoms:
        """
        Attach model outputs to atoms.info for 1 frame (sys_idx).
        
        All outputs are stored in atoms.info since ASE Trajectory preserves info dict
        but not custom arrays.
        
        Retrieves values from (in order of priority):
        1. atoms.calc.results (if calculator attached)
        2. ctx.engine.model.step_outputs (primary source for atomic_charge etc.)
        3. ctx.state dict (all entries including uncertainty, thermo data, custom data)
        4. ctx.state["sim_state"] attributes (energy, forces, momenta, etc.)
        """
        if not self._output_map:
            return atoms
        
        # Collect all available outputs from multiple sources
        available_outputs: Dict[str, Any] = {}
        
        # Source 1: Calculator results
        if atoms.calc is not None and hasattr(atoms.calc, 'results') and atoms.calc.results:
            available_outputs.update(atoms.calc.results)
        
        # Source 2: Model step_outputs (from CuratorTorchSimAdapter)
        if hasattr(ctx, 'engine') and hasattr(ctx.engine, 'model'):
            model = ctx.engine.model
            if hasattr(model, 'step_outputs') and model.step_outputs is not None:
                available_outputs.update(model.step_outputs)
        
        # Source 3: All ctx.state entries (uncertainty, thermo, custom data, etc.)
        for key, value in ctx.state.items():
            if key == "sim_state":
                continue  # Handle sim_state separately below
            # Handle nested dicts (like uncertainty)
            if isinstance(value, dict):
                available_outputs.update(value)
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                # Batched dict data - get this system's entry
                if len(value) > sys_idx:
                    available_outputs.update(value[sys_idx])
            elif isinstance(value, (torch.Tensor, np.ndarray, float, int, list)):
                available_outputs[key] = value
        
        # Source 4: SimState attributes (energy, forces, momenta, etc.)
        sim_state = ctx.state.get("sim_state")
        if sim_state is not None:
            for attr in dir(sim_state):
                if not attr.startswith('_'):
                    try:
                        val = getattr(sim_state, attr)
                        if isinstance(val, (torch.Tensor, np.ndarray)):
                            available_outputs[attr] = val
                    except Exception:
                        pass
        
        # Source 5: Computed thermo properties (temp, ekin, pressure, etc.)
        for prop_name in self._output_map.keys():
            if prop_name in self.THERMO_PROPERTY_MAP and prop_name not in available_outputs:
                computed_val = self._compute_thermo_property(prop_name, ctx, sys_idx)
                if computed_val is not None:
                    available_outputs[prop_name] = computed_val
        
        # Attach requested outputs to atoms.info
        n_atoms = len(atoms)
        for prop_name, store_key in self._output_map.items():
            if prop_name not in available_outputs:
                if ctx.step == 1:  # Only warn once at first step
                    logger.warning(
                        f"TrajectoryWriter: Property '{prop_name}' not found. "
                        f"Available: {list(available_outputs.keys())}"
                    )
                continue
            
            value = available_outputs[prop_name]
            arr = self._to_numpy(value)
            
            # Handle batched data (multiple systems) - extract this system's data
            if sim_state is not None and hasattr(sim_state, 'n_atoms_per_system'):
                n_atoms_list = sim_state.n_atoms_per_system
                if isinstance(n_atoms_list, torch.Tensor):
                    n_atoms_list = n_atoms_list.tolist()
                
                total_atoms = sum(n_atoms_list)
                n_systems = len(n_atoms_list)
                
                # Per-atom data: shape[0] == total_atoms
                if arr.ndim >= 1 and arr.shape[0] == total_atoms:
                    start_idx = sum(n_atoms_list[:sys_idx])
                    end_idx = start_idx + n_atoms_list[sys_idx]
                    arr = arr[start_idx:end_idx]
                # Per-system data: shape[0] == n_systems
                elif arr.ndim >= 1 and arr.shape[0] == n_systems:
                    arr = arr[sys_idx]
            
            # Store in atoms.info (convert scalar to float for cleaner output)
            if arr.ndim == 0 or arr.size == 1:
                atoms.info[store_key] = float(arr.flat[0]) if arr.size == 1 else float(arr)
            else:
                atoms.info[store_key] = arr
            
        return atoms

    def _ensure_trajs(self, n_sys: int):
        if self._trajs:
            return
        base = self.path
        for i in range(n_sys if self.per_system else 1):
            if self.per_system:
                if "{i}" in base:
                    p = base.format(i=i)
                else:
                    stem, ext = base.rsplit(".", 1) if "." in base else (base, "traj")
                    p = f"{stem}_sys{i}.{ext}"
            else:
                p = base
            self._trajs.append(Trajectory(p, self.mode))

    def _atoms_from_ctx(self, ctx: SimContext) -> List[Any]:
        state = ctx.state.get("sim_state")
        if state is not None and ts is not None:
            try:
                return ts.io.state_to_atoms(state)
            except Exception:
                pass
        if isinstance(ctx.atoms, list):
            return ctx.atoms
        if ctx.atoms is not None:
            return [ctx.atoms]
        return []

    def on_sim_start(self, ctx: SimContext):
        atoms_list = self._atoms_from_ctx(ctx)
        n_sys = len(atoms_list) if atoms_list else 1
        self._ensure_trajs(n_sys)

    def on_step(self, ctx: SimContext):
        if ctx.step % self.interval != 0:
            return
        atoms_list = self._atoms_from_ctx(ctx)
        if not atoms_list:
            return
        self._ensure_trajs(len(atoms_list))
        if self.per_system:
            for i, atoms in enumerate(atoms_list):
                if i < len(self._trajs):
                    # Attach model outputs before writing
                    atoms = self._attach_model_outputs(atoms, ctx, sys_idx=i)
                    self._trajs[i].write(atoms)
        else:
            for i, atoms in enumerate(atoms_list):
                atoms = self._attach_model_outputs(atoms, ctx, sys_idx=i)
                self._trajs[0].write(atoms)

    def on_sim_end(self, ctx: SimContext):
        for t in self._trajs:
            try:
                t.close()
            except Exception:
                pass
        self._trajs = []
