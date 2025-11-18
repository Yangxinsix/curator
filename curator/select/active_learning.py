import torch
from torch import nn
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Union
from curator.data import collate_atomsdata
from .select import *
from .kernel import *
from curator.data import properties
try:
    from torch_scatter import scatter_add, scatter_mean, scatter_max
except ImportError:
    from curator.utils import scatter_add, scatter_mean, scatter_max
from e3nn import o3
from curator.layer.utils import find_layer_by_name_recursive
import logging

logger = logging.getLogger(__name__)

class FeatureExtractor(nn.Module):
    """Extract features from neural networks"""

    def __init__(self, model: nn.Module, target_layer: str = 'readout_mlp',) -> None:
        """Extract features from neural networks

        Args:
            model (nn.Module): pytorch model
            target_layer (str): name of target layer to extract features from
        """
        super().__init__()
        self.model = model
        self.target_layer = target_layer
        self._features = []
        self._grads = []
        self.hooks = []
        layer = find_layer_by_name_recursive(self.model, self.target_layer)
        assert layer is not None, f"Target layer {self.target_layer} is not found!"

        representation = getattr(self.model, "representation", None)
        if representation is not None and representation.__class__.__name__ == "MACE":
            layer = layer[-1]

        for child in layer.children():
            if isinstance(child, (nn.Linear, o3.Linear)):
                self.hooks.append(child.register_forward_pre_hook(self.save_feats_hook))
                self.hooks.append(child.register_backward_hook(self.save_grads_hook))

    def save_feats_hook(self, _, in_feat):
        new_feat = torch.cat((in_feat[0].detach().clone(), torch.ones_like(in_feat[0][:, 0:1])), dim=-1)
        self._features.append(new_feat)

    def save_grads_hook(self, _, __, grad_output):
        self._grads.append(grad_output[0].detach().clone())

    def unhook(self):
        for hook in self.hooks:
            hook.remove()

    def forward(self, model_inputs: Dict[str, torch.Tensor]):
        self._features = []
        self._grads = []
        _ = self.model(model_inputs)
        return self._features, self._grads[::-1]
    
class RandomProjections:
    """Store parameters of random projections"""
    def __init__(
            self, 
            model: nn.Module, 
            num_features: int,
            dtype = torch.get_default_dtype(),
            target_layer: str = 'readout_mlp',
        ):
        self.num_features = num_features
        self.in_feat_proj = []
        self.out_grad_proj = []
        device = next(model.parameters()).device
        if self.num_features > 0:
            layer = find_layer_by_name_recursive(model, target_layer)
            representation = getattr(model, "representation", None)
            if representation is not None and representation.__class__.__name__ == "MACE":
                layer = layer[-1]
            # Input feature projection matrices (in_features + 1 for bias term), output gradient projection matrices
            for l in layer.children():
                if isinstance(l, nn.Linear):
                    self.in_feat_proj.append(torch.randn(l.in_features + 1, self.num_features, dtype=dtype, device=device))
                    self.out_grad_proj.append(torch.randn(l.out_features, self.num_features, dtype=dtype, device=device))
                elif isinstance(l, o3.Linear):
                    self.in_feat_proj.append(torch.randn(l.irreps_in.dim + 1, self.num_features, dtype=dtype, device=device))
                    self.out_grad_proj.append(torch.randn(l.irreps_out.dim, self.num_features, dtype=dtype, device=device))
            
    def __repr__(self):
        return f'{self.__class__.__name__}(num_features={self.num_features})'
    
class FeatureStatistics:
    """Generate features from trained models and datasets."""

    def __init__(
        self,
        models: List[nn.Module],
        dataset: torch.utils.data.Dataset,
        n_random_features: int=500,
        random_projections: Optional[List[RandomProjections]] = None,
        batch_size: int=8,
        device: Optional[str]=None,
        debug: bool=False,
    ):
        self.models = models
        self.batch_size = batch_size
        self.dataset = dataset
        if random_projections is None:
            self.random_projections = [RandomProjections(model, n_random_features) for model in self.models]
        else:
            self.random_projections = random_projections
        self.device = device or next(models[0].parameters()).device
        self._features_cache: Dict[Tuple[str, bool], torch.Tensor] = {}
        self.ens_stats = None
        self.Fisher = None
        self.F_reg_inv = None
        self.debug = debug

        self.ensemble = None
        self._kernel_handlers = {
            'full-gradient': self._full_gradient_features,
            'local_full-g': self._local_full_gradient_features,
            'll-gradient': self._ll_gradient_features,
            'local_ll-g': self._local_ll_gradient_features,
            'gnn': self._gnn_features,
            'local_gnn': self._local_gnn_features,
        }
    
    def _compute_ens_stats(self, model_inputs: Dict[str, torch.Tensor], method: str = "ensemble") -> Dict[str, torch.Tensor]:
        """Compute energy variance, forces variance, energy absolute error, and forces absolute error"""
        ens_stats = {}
        if method == "ensemble":
            result_dict = self.ensemble(model_inputs)
            if properties.uncertainty in result_dict:
                for k, v in result_dict[properties.uncertainty].items():
                    ens_stats[k] = v
            if properties.error in result_dict:
                for k, v in result_dict[properties.error].items():
                    ens_stats[k] = v
        
        return ens_stats
                
    def _compute_features(
        self,
        feature_extractor: FeatureExtractor,
        model_inputs: Dict[str, torch.Tensor],
        random_projection: RandomProjections,
        kernel: str='ll-gradient',
        to_cpu: bool=True,
    ) -> torch.Tensor:
        """Dispatch feature computation to the registered kernel handlers."""

        if kernel not in self._kernel_handlers:
            raise RuntimeError(f"Unknown kernel '{kernel}'")
        return self._kernel_handlers[kernel](
            feature_extractor=feature_extractor,
            model_inputs=model_inputs,
            random_projection=random_projection,
            to_cpu=to_cpu,
        )

    def _project_all_layers(
        self,
        feats: List[torch.Tensor],
        grads: List[torch.Tensor],
        random_projection: RandomProjections,
        image_idx: torch.Tensor,
    ) -> torch.Tensor:
        assert random_projection.num_features != 0, "Error! Random projections must be provided!"
        atomic_g = torch.zeros((image_idx.shape[0], random_projection.num_features), device=image_idx.device)
        for feat, grad, in_proj, out_proj in zip(
            feats,
            grads,
            random_projection.in_feat_proj,
            random_projection.out_grad_proj,
        ):
            atomic_g += (feat @ in_proj) * (grad @ out_proj)
        return atomic_g

    def _aggregate_atomic_features(
        self,
        atomic_g: torch.Tensor,
        image_idx: torch.Tensor,
        to_cpu: bool,
        reduce_to_structure: bool = True,
    ) -> torch.Tensor:
        if reduce_to_structure:
            g = scatter_add(atomic_g, image_idx, dim=0)
        else:
            g = atomic_g
        return g.cpu() if to_cpu else g

    def _layer_features(
        self,
        feat: torch.Tensor,
        grad: torch.Tensor,
        random_projection: RandomProjections,
        proj_idx: int,
    ) -> torch.Tensor:
        if random_projection.num_features != 0:
            return (feat @ random_projection.in_feat_proj[proj_idx]) * (
                grad @ random_projection.out_grad_proj[proj_idx]
            )
        return feat[:, :-1]

    def _full_gradient_features(
        self,
        feature_extractor: FeatureExtractor,
        model_inputs: Dict[str, torch.Tensor],
        random_projection: RandomProjections,
        to_cpu: bool,
    ) -> torch.Tensor:
        image_idx = model_inputs[properties.image_idx]
        feats, grads = feature_extractor(model_inputs)
        atomic_g = self._project_all_layers(feats, grads, random_projection, image_idx)
        return self._aggregate_atomic_features(atomic_g, image_idx, to_cpu, reduce_to_structure=True)

    def _local_full_gradient_features(
        self,
        feature_extractor: FeatureExtractor,
        model_inputs: Dict[str, torch.Tensor],
        random_projection: RandomProjections,
        to_cpu: bool,
    ) -> torch.Tensor:
        image_idx = model_inputs[properties.image_idx]
        feats, grads = feature_extractor(model_inputs)
        atomic_g = self._project_all_layers(feats, grads, random_projection, image_idx)
        return self._aggregate_atomic_features(atomic_g, image_idx, to_cpu, reduce_to_structure=False)

    def _ll_gradient_features(
        self,
        feature_extractor: FeatureExtractor,
        model_inputs: Dict[str, torch.Tensor],
        random_projection: RandomProjections,
        to_cpu: bool,
    ) -> torch.Tensor:
        image_idx = model_inputs[properties.image_idx]
        feats, grads = feature_extractor(model_inputs)
        atomic_g = self._layer_features(feats[-1], grads[-1], random_projection, -1)
        return self._aggregate_atomic_features(atomic_g, image_idx, to_cpu, reduce_to_structure=True)

    def _local_ll_gradient_features(
        self,
        feature_extractor: FeatureExtractor,
        model_inputs: Dict[str, torch.Tensor],
        random_projection: RandomProjections,
        to_cpu: bool,
    ) -> torch.Tensor:
        image_idx = model_inputs[properties.image_idx]
        feats, grads = feature_extractor(model_inputs)
        atomic_g = self._layer_features(feats[-1], grads[-1], random_projection, -1)
        return self._aggregate_atomic_features(atomic_g, image_idx, to_cpu, reduce_to_structure=False)

    def _gnn_features(
        self,
        feature_extractor: FeatureExtractor,
        model_inputs: Dict[str, torch.Tensor],
        random_projection: RandomProjections,
        to_cpu: bool,
    ) -> torch.Tensor:
        image_idx = model_inputs[properties.image_idx]
        feats, grads = feature_extractor(model_inputs)
        atomic_g = self._layer_features(feats[0], grads[0], random_projection, 0)
        return self._aggregate_atomic_features(atomic_g, image_idx, to_cpu, reduce_to_structure=True)

    def _local_gnn_features(
        self,
        feature_extractor: FeatureExtractor,
        model_inputs: Dict[str, torch.Tensor],
        random_projection: RandomProjections,
        to_cpu: bool,
    ) -> torch.Tensor:
        image_idx = model_inputs[properties.image_idx]
        feats, grads = feature_extractor(model_inputs)
        atomic_g = self._layer_features(feats[0], grads[0], random_projection, 0)
        return self._aggregate_atomic_features(atomic_g, image_idx, to_cpu, reduce_to_structure=False)

    def _iter_batches(self, dataset):
        for batch in dataset:
            yield {k: v.to(self.device) for k, v in batch.items()}

    def _normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        mean = torch.mean(features, dim=0)
        var = torch.var(features, dim=0)
        var = torch.where(var == 0, torch.ones_like(var), var)
        return (features - mean) / var
    
    def _compute_fisher(self, g: torch.Tensor) -> torch.Tensor:
        return torch.einsum('mci, mcj -> mij', g, g)
                                                                                               
    def get_features(
        self,
        dataset: Optional[torch.utils.data.Dataset]=None,
        kernel: str='full-gradient',
        to_cpu: bool=True,
    ) -> torch.Tensor:
        """
        :return: Feature vector of ``shape=(n_models, n_structures, n_features)``.
        """
        if dataset == None:
            dataset = self.dataset
        else:
            self.dataset = dataset
            self._features_cache.clear()

        cache_key = (kernel, to_cpu)
        if cache_key not in self._features_cache:
            global_g = []
            for model, random_proj in zip(self.models, self.random_projections):
                feature_extractor = FeatureExtractor(model)
                model_batches = []
                for b, batch in enumerate(self._iter_batches(dataset)):
                    if self.debug:
                        logger.info(f"Predicting {b}th sample for model {model.__class__.__name__}.")
                    model_batches.append(self._compute_features(
                        feature_extractor=feature_extractor,
                        model_inputs=batch,
                        random_projection=random_proj,
                        kernel=kernel,
                        to_cpu=to_cpu,
                    ))
                feature_extractor.unhook()
                model_g = torch.cat(model_batches)
                global_g.append(self._normalize_features(model_g))

            self._features_cache[cache_key] = torch.stack(global_g)

        return self._features_cache[cache_key]

    def get_g(self, kernel: str='full-gradient', to_cpu: bool=True) -> torch.Tensor:
        """Compatibility helper that returns cached features for a kernel."""

        return self.get_features(kernel=kernel, to_cpu=to_cpu)

    def get_num_atoms(
        self,
        dataset: Optional[torch.utils.data.Dataset]=None,
    ):
        if dataset == None:
            dataset = self.dataset
        else:
            self.dataset = dataset
            self._features_cache.clear()
        num_atoms = []
        # dataloader = torch.utils.data.DataLoader(
        #     dataset=dataset,
        #     batch_size=self.batch_size,
        #     collate_fn=collate_atomsdata,
        # )
        for batch in self._iter_batches(dataset):
            num_atoms.append(batch[properties.n_atoms])

        return torch.cat(num_atoms)

    def get_ens_stats(self, dataset: Optional[torch.utils.data.Dataset]=None, method="ensemble") -> Dict[str, torch.Tensor]:
        """
        :return: Dict of energy statistics
        """
        if dataset == None:
            dataset = self.dataset
        else:
            self.dataset = dataset
            self.ens_stats = None
            self._features_cache.clear()
            
        if self.ens_stats is None:
            if method == "ensemble":
                from curator.model import EnsembleModel
                if self.ensemble is None:
                    self.ensemble = EnsembleModel(self.models)
            else:
                raise NotImplementedError(f"Method {method} is not implemented.")

            # dataloader = torch.utils.data.DataLoader(
            #     dataset=dataset,
            #     batch_size=self.batch_size,
            #     collate_fn=collate_atomsdata,
            # )
            # Simply using dataset is faster?
            ens_stats = []
            for i, batch in enumerate(dataset):
                if self.debug:
                    logger.info(f"Predicting {i}th sample.")
                batch = {k: v.to(self.device) for k, v in batch.items()}
                ens_stats.append(self._compute_ens_stats(batch, method))

            self.ens_stats = {k: torch.cat([ens[k] for ens in ens_stats]) for k in ens_stats[0].keys()}
            
        return self.ens_stats
    
    def get_fisher(self) -> torch.Tensor:
        if self.Fisher is None:
            self.Fisher = self._compute_fisher(self.get_features())
        return self.Fisher

    def get_F_inv(self) -> torch.Tensor:
        """
        :return: Regularized inverse of Fisher matrix of "shape=(n_models, n_features, n_features)".
        """
        if self.F_reg_inv is None:
            fisher = self.get_fisher()
            n_features = fisher.shape[-1]
            eye = torch.eye(n_features, device=fisher.device, dtype=fisher.dtype).unsqueeze(0)
            # empirical regularisation computed per-model to stabilise inversion
            lam = torch.linalg.trace(fisher, dim1=-2, dim2=-1) / max(n_features, 1)
            lam = lam[:, None, None]
            fisher_reg = fisher + lam * eye
            self.F_reg_inv = torch.linalg.inv(fisher_reg)
        return self.F_reg_inv


class DistanceMetrics:
    """Compute simple distance metrics from cached feature statistics."""

    def __init__(
        self,
        train_stats: FeatureStatistics,
        dataset_stats: Optional[FeatureStatistics] = None,
        regularization: float = 1e-6,
    ) -> None:
        self.train_stats = train_stats
        self.dataset_stats = dataset_stats
        self.regularization = regularization
        self._mean_cache: Dict[str, torch.Tensor] = {}
        self._precision_cache: Dict[str, torch.Tensor] = {}

    def get_mahalanobis_distance(
        self,
        stats: Optional[FeatureStatistics] = None,
        kernel: Optional[str] = None,
        local: bool = False,
        reduction: Optional[str] = None,
    ) -> torch.Tensor:
        kernel = kernel or self._default_kernel(local)
        stats = self._resolve_stats(stats)
        features = self._collapse_models(stats.get_features(kernel=kernel))
        mean = self.get_feature_mean(kernel)
        precision = self.get_feature_precision(kernel)
        diff = features - mean
        dist_sq = torch.einsum('bi,ij,bj->b', diff, precision, diff)
        distances = torch.sqrt(torch.clamp(dist_sq, min=0.0))
        return self._reduce(distances, stats, local, reduction)

    def get_euclidean_distance(
        self,
        stats: Optional[FeatureStatistics] = None,
        kernel: Optional[str] = None,
        local: bool = False,
        reduction: Optional[str] = None,
    ) -> torch.Tensor:
        kernel = kernel or self._default_kernel(local)
        stats = self._resolve_stats(stats)
        features = self._collapse_models(stats.get_features(kernel=kernel))
        mean = self.get_feature_mean(kernel)
        diff = features - mean
        distances = torch.sqrt(torch.clamp(torch.sum(diff * diff, dim=-1), min=0.0))
        return self._reduce(distances, stats, local, reduction)

    def get_cosine_distance(
        self,
        stats: Optional[FeatureStatistics] = None,
        kernel: Optional[str] = None,
        local: bool = False,
        reduction: Optional[str] = None,
    ) -> torch.Tensor:
        kernel = kernel or self._default_kernel(local)
        stats = self._resolve_stats(stats)
        features = self._collapse_models(stats.get_features(kernel=kernel))
        mean = self.get_feature_mean(kernel)
        norm_features = torch.linalg.norm(features, dim=-1)
        norm_mean = torch.linalg.norm(mean)
        similarity = torch.einsum('bi,i->b', features, mean) / (norm_features * norm_mean + 1e-12)
        distances = 1 - similarity
        return self._reduce(distances, stats, local, reduction)

    def set_dataset_stats(self, stats: FeatureStatistics) -> None:
        """Update dataset statistics without rebuilding the helper."""
        self.dataset_stats = stats

    def get_feature_mean(self, kernel: str = 'gnn') -> torch.Tensor:
        if kernel not in self._mean_cache:
            feats = self._collapse_models(self.train_stats.get_features(kernel=kernel))
            self._mean_cache[kernel] = torch.mean(feats, dim=0)
        return self._mean_cache[kernel]

    def get_feature_precision(self, kernel: str = 'gnn') -> torch.Tensor:
        if kernel not in self._precision_cache:
            feats = self._collapse_models(self.train_stats.get_features(kernel=kernel))
            mean = self.get_feature_mean(kernel)
            centered = feats - mean
            denom = max(centered.shape[0] - 1, 1)
            covariance = centered.T @ centered / denom
            eye = torch.eye(covariance.shape[0], device=covariance.device, dtype=covariance.dtype)
            covariance = covariance + self.regularization * eye
            self._precision_cache[kernel] = torch.linalg.inv(covariance)
        return self._precision_cache[kernel]

    def _resolve_stats(self, stats: Optional[FeatureStatistics]) -> FeatureStatistics:
        if stats is not None:
            return stats
        if self.dataset_stats is None:
            raise ValueError("Dataset statistics are not provided.")
        return self.dataset_stats

    @staticmethod
    def _collapse_models(features: torch.Tensor) -> torch.Tensor:
        if features.dim() != 3:
            raise ValueError("Expected features tensor with shape (n_models, n_items, n_features).")
        return features.mean(dim=0)

    @staticmethod
    def _default_kernel(local: bool) -> str:
        return 'local_gnn' if local else 'gnn'

    def _reduce(
        self,
        distances: torch.Tensor,
        stats: FeatureStatistics,
        local: bool,
        reduction: Optional[str],
    ) -> torch.Tensor:
        if not local or reduction is None:
            return distances
        if reduction not in {'mean', 'sum', 'max'}:
            raise ValueError(f"Unsupported reduction '{reduction}'.")
        image_idx = self._get_image_idx(stats)
        if reduction == 'mean':
            return scatter_mean(distances, image_idx, dim=0)
        if reduction == 'sum':
            return scatter_add(distances, image_idx, dim=0)
        max_result = scatter_max(distances, image_idx, dim=0)
        return max_result[0] if isinstance(max_result, tuple) else max_result

    @staticmethod
    def _get_image_idx(stats: FeatureStatistics) -> torch.Tensor:
        num_atoms = stats.get_num_atoms()
        device = num_atoms.device
        image_idx = torch.arange(num_atoms.shape[0], device=device)
        return torch.repeat_interleave(image_idx, num_atoms)

class GeneralActiveLearning:
    """Provides methods for selecting batches during active learning.

    :param kernel: Name of the kernel, e.g. "full-g", "ll-g", "full-F_inv", "ll-F_inv", "qbc-energy", "qbc-force".
                   "random" produces random selection and "ae-energy" and "ae-force" select by absolute errors
                   on the pool data, which is only possible if the pool data is already labeled.
    :param selection: Selection method, one of "max_dist_greedy", "deterministic_CUR", "lcmd_greedy", "max_det_greedy" or "max_diag".
    :param n_random_features: If "n_random_features = 0", do not use random projections.
                              Otherwise, use random projections of all linear-layer gradients.
    """
    def __init__(
        self,
        kernel = 'full-g',
        selection = 'max_diag',
        n_random_features = 0,
        save_features = False,
    ):
        self.kernel = kernel
        self.selection = selection
        self.n_random_features = n_random_features
        self.save_features = save_features
    
    def select(
        self, 
        models: List[nn.Module], 
        datasets: Dict[str, torch.utils.data.Dataset], 
        batch_size: int = 8, 
        al_batch_size: int = 100,
        debug: bool = False,
    ):
        """
        models: pytorch models,
        dataset: a dictionary containing pool, train, and validation dataset,
        batch_size: batch size for extracting features,
        al_batch_size: active learning selection batch size
        """        
        if (self.kernel == 'qbc-energy' or self.kernel == 'qbc-force' or self.kernel == 'ae-energy' or
            self.kernel == 'ae-force' or self.kernel == 'random') and self.selection != 'max_diag':
            raise RuntimeError(f'{self.kernel} kernel can only be used with max_diag selection method,'
                               f' not with {self.selection}!')
        
        stats = {
            key: FeatureStatistics(models, ds, self.n_random_features, batch_size=batch_size, debug=debug)
            for key, ds in datasets.items()
        }
        
        # pool-based selection or pool + train based selection
        if datasets.get('train'):
            matrix = self._get_kernel_matrix(stats['pool'], stats['train'])
            n_train = len(datasets['train'])
        else:
            matrix = self._get_kernel_matrix(stats['pool'])
            n_train = 0
        
        if self.selection == 'max_dist_greedy':
            idxs = max_dist_greedy(matrix=matrix, batch_size=al_batch_size, n_train=n_train)
        elif self.selection == 'max_diag':
            idxs = max_diag(matrix=matrix, batch_size=al_batch_size)
        elif self.selection == 'max_det_greedy':
            idxs = max_det_greedy(matrix=matrix, batch_size=al_batch_size)
        elif self.selection == 'lcmd_greedy':
            idxs = lcmd_greedy(matrix=matrix, batch_size=al_batch_size, n_train=n_train)
        elif self.selection == 'max_det_greedy_local':
            idxs = max_det_greedy_local(matrix=matrix, batch_size=al_batch_size, num_atoms=num_atoms)
        elif self.selection == False:
            idxs = torch.tensor([0])
        else:
            raise NotImplementedError(f"Unknown selection method '{self.selection}' for active learning!")
        
        if self.save_features:
            features = { key: s.get_features() for key, s in stats.items()}
            torch.save(features, 'features.pt')

        return idxs.cpu().tolist()

    def _get_kernel_matrix(self, pool_stats: FeatureStatistics, train_stats: Optional[FeatureStatistics]=None) -> KernelMatrix:
        stats_list = [pool_stats] if train_stats == None else [pool_stats, train_stats]
        
        if self.kernel == 'full-g':
            return FeatureKernelMatrix(torch.cat([s.get_features(kernel='full-gradient') for s in stats_list], dim=1))
        elif self.kernel == 'll-g':
            return FeatureKernelMatrix(torch.cat([s.get_features(kernel='ll-gradient') for s in stats_list], dim=1))
        elif self.kernel == 'gnn':
            return FeatureKernelMatrix(torch.cat([s.get_features(kernel='gnn') for s in stats_list], dim=1))
        elif self.kernel == 'local_full-g':
            matrix = FeatureKernelMatrix(torch.cat([s.get_features(kernel='local_full-g') for s in stats_list], dim=1))
            num_atoms = torch.cat([s.get_num_atoms() for s in stats_list])
            return matrix, num_atoms
        elif self.kernel == 'local_ll-g':
            matrix = FeatureKernelMatrix(torch.cat([s.get_features(kernel='local_ll-g') for s in stats_list], dim=1))
            num_atoms = torch.cat([s.get_num_atoms() for s in stats_list])
            return matrix, num_atoms 
        elif self.kernel == 'local_gnn':
            matrix = FeatureKernelMatrix(torch.cat([s.get_features(kernel='local_gnn') for s in stats_list], dim=1))
            num_atoms = torch.cat([s.get_num_atoms() for s in stats_list])
            return matrix, num_atoms 
        elif self.kernel == 'full-F_inv':
            return FeatureCovKernelMatrix(torch.cat([s.get_features(kernel='full-gradient') for s in stats_list], dim=1),
                                          train_stats.get_F_reg_inv())
        elif self.kernel == 'll-F_inv':
            return FeatureCovKernelMatrix(torch.cat([s.get_features(kernel='ll-gradient') for s in stats_list], dim=1),
                                          train_stats.get_F_reg_inv())
        elif self.kernel == 'qbc-energy':
            return DiagonalKernelMatrix(pool_stats.get_ens_stats()['Energy-Var'])
        elif self.kernel == 'qbc-force':
            return DiagonalKernelMatrix(pool_stats.get_ens_stats()['Forces-Var'])
        elif self.kernel == 'ae-energy':
            return DiagonalKernelMatrix(pool_stats.get_ens_stats()['Energy-AE'])
        elif self.kernel == 'ae-force':
            return DiagonalKernelMatrix(pool_stats.get_ens_stats()['Forces-AE'])
        elif self.kernel == 'random':
            return DiagonalKernelMatrix(torch.rand([sum([len(s.dataset) for s in stats_list])]))
        else:
            raise RuntimeError(f"Unknown active learning kernel {self.kernel}!")