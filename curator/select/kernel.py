import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Try to import faiss, set flag if available
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    # logger.warning("Faiss not installed. FaissKernelMatrix will not be available. "
    #                "Install with: conda install -c pytorch faiss-gpu (or faiss-cpu)")

class KernelMatrix:
    """Abstract kernel class used to calculate kernel matrix by giving a feature matrix"""
    def __init__(self, num_col: int) -> None:
        self.num_columns = num_col
    
    def get_number_of_columns(self) -> int:
        return self.num_columns
    
    def get_column(self, i: int) -> torch.Tensor:
        raise RuntimeError("Not implemented")
        
    def get_diag(self) -> torch.Tensor:
        raise RuntimeError("Not implemented")
        
    def get_sq_dists(self, i: int) -> torch.Tensor:
        diag = self.get_diag()
        return diag[i] + diag - 2 * self.get_column(i)
    
class DiagonalKernelMatrix(KernelMatrix):
    """
    Represents a diagonal kernel matrix, where get_column() and get_sq_dists() is not implemented.

    :param g: Diagonal of the kernel matrix.
    """
    def __init__(self, g: torch.Tensor) -> None:
        super().__init__(g.shape[0])
        self.diag = g

    def get_diag(self) -> torch.Tensor:
        return self.diag
        
class FeatureKernelMatrix(KernelMatrix):
    """
    input: m x n x p matrix
    m: number of models
    n: number of entries
    p: dimensionality of features
    """
    def __init__(self, mat: torch.Tensor) -> None:
        super().__init__(mat.shape[1])
        self.mat = mat
        self.diag = torch.einsum('mbi, mbi -> mb', mat, mat)
    
    def get_column(self, i: int) -> torch.Tensor:
        return torch.mean(torch.einsum("mnp, mp -> mn", self.mat, self.mat[:, i, :]), dim=0)
    
    def get_diag(self) -> torch.Tensor:
        return torch.mean(self.diag, dim=0)
    
class FeatureCovKernelMatrix(KernelMatrix):
    """
    input: m x n x p matrix mat, m x p x p covariance matrix
    m: number of models
    n: number of entries
    p: dimensionality of features
    """
    def __init__(self, g: torch.Tensor, cov_mat: torch.Tensor) -> None:
        super().__init__(g.shape[1])
        self.g = g
        self.cov_mat = cov_mat
        self.cov_g = torch.einsum('mij, mbi -> mbj', self.cov_mat, g)
        self.diag = torch.einsum('mbi, mbi -> mb', self.cov_g, g)
        
    def get_diag(self) -> torch.Tensor:
        return torch.mean(self.diag, dim=0)

    def get_column(self, i: int) -> torch.Tensor:
        return torch.mean(torch.einsum('mbi, mi -> mb', self.g, self.cov_g[:, i, :]), dim=0)


class FaissKernelMatrix(KernelMatrix):
    """
    Faiss-accelerated kernel matrix for fast similarity search.
    
    Uses Faiss IndexFlatL2 for GPU-accelerated distance computation.
    Input: m x n x p matrix (will be collapsed to n x p by averaging over models)
    
    m: number of models (ensemble)
    n: number of entries (samples)
    p: dimensionality of features
    """
    def __init__(self, mat: torch.Tensor, use_gpu: bool = True) -> None:
        
        # Collapse ensemble dimension by averaging
        # mat: (M, N, P) -> features: (N, P)
        if mat.dim() == 3:
            features = mat.mean(dim=0)
        else:
            features = mat
        
        n_samples, n_features = features.shape
        super().__init__(n_samples)
        
        # Store original device for output conversion
        self._device = features.device
        self._dtype = features.dtype
        
        # Convert to numpy float32 (Faiss requirement)
        self.features_np = features.cpu().contiguous().numpy().astype(np.float32)
        
        # Precompute norms squared (||g_i||^2)
        self.norms_sq = np.sum(self.features_np ** 2, axis=1)
        self._diag = torch.from_numpy(self.norms_sq).to(self._device, self._dtype)
        
        # Create Faiss index
        self.use_gpu = use_gpu and faiss.get_num_gpus() > 0
        
        if self.use_gpu:
            logger.info(f"FaissKernelMatrix: Using GPU acceleration with {faiss.get_num_gpus()} GPU(s)")
            self.res = faiss.StandardGpuResources()
            self.index = faiss.GpuIndexFlatL2(self.res, n_features)
        else:
            logger.info("FaissKernelMatrix: Using CPU (GPU not available or disabled)")
            self.index = faiss.IndexFlatL2(n_features)
        
        # Add all vectors to the index
        self.index.add(self.features_np)
        logger.info(f"FaissKernelMatrix initialized: {n_samples} samples, {n_features} features")
    
    def get_diag(self) -> torch.Tensor:
        """Return precomputed norms squared."""
        return self._diag
    
    def get_sq_dists(self, i: int) -> torch.Tensor:
        """
        Compute squared L2 distances from point i to all points.
        
        ||a - b||² = ||a||² + ||b||² - 2<a, b>
        
        """
        query = self.features_np[i:i+1]  # (1, P)
        
        # 使用公式: ||q - x||² = ||q||² + ||x||² - 2 * q·x
        # 这种方式没有 k 限制，且利用了预计算的 norms_sq
        query_norm = self.norms_sq[i]  # scalar: ||q||²
        inner_products = self.features_np @ query.T  # (N, 1): q·x_i
        distances = query_norm + self.norms_sq - 2 * inner_products.flatten()
        
        return torch.from_numpy(distances.astype(np.float32)).to(self._device, self._dtype)
    
    def get_column(self, i: int) -> torch.Tensor:
        """
        Compute inner products <g_i, g_j> for all j.
        
        Uses the identity: <a,b> = (||a||^2 + ||b||^2 - ||a-b||^2) / 2
        """
        sq_dists = self.get_sq_dists(i)
        # <g_i, g_j> = (||g_i||^2 + ||g_j||^2 - ||g_i - g_j||^2) / 2
        inner_products = (self._diag[i] + self._diag - sq_dists) / 2
        return inner_products
    
    def get_features(self) -> np.ndarray:
        """Return the feature matrix as numpy array (for direct Faiss operations)."""
        return self.features_np
    
    def create_index_from_indices(self, indices: list) -> 'faiss.Index':
        """
        Create a new Faiss index containing only the specified points.
        
        This is useful for building an index of selected points incrementally.
        
        Args:
            indices: List of point indices to include in the new index
            
        Returns:
            A new Faiss index containing only the specified points
        """
        subset = self.features_np[indices]
        n_features = self.features_np.shape[1]
        
        if self.use_gpu:
            new_index = faiss.GpuIndexFlatL2(self.res, n_features)
        else:
            new_index = faiss.IndexFlatL2(n_features)
        
        new_index.add(subset)
        return new_index
    
    def batch_nearest_in_selected(self, query_indices: np.ndarray, selected_indices: list) -> tuple:
        """
        For each query point, find its nearest neighbor among the selected points.
        
        Args:
            query_indices: Array of query point indices
            selected_indices: List of selected point indices
            
        Returns:
            (distances, local_indices): 
                - distances[i] = query_indices[i] 到最近已选点的距离²
                - local_indices[i] = 最近已选点在 selected_indices 中的位置
        """
        # Build index from selected points
        selected_features = self.features_np[selected_indices]
        n_features = self.features_np.shape[1]
        
        if self.use_gpu:
            selected_index = faiss.GpuIndexFlatL2(self.res, n_features)
        else:
            selected_index = faiss.IndexFlatL2(n_features)
        
        selected_index.add(selected_features)
        
        # Query all pool points
        query_features = self.features_np[query_indices]
        distances, indices = selected_index.search(query_features, k=1)
        
        return distances.flatten(), indices.flatten()

