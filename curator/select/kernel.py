import torch

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
        super().__init__(mat.shape[1])
        self.g = g
        self.cov_mat = cov_mat
        self.cov_g = torch.einsum('mij, mbi -> mbj', self.cov_mat, g)
        self.diag = torch.einsum('mbi, mbi -> mb', self.cov_g, g)
        
    def get_diag(self) -> torch.Tensor:
        return torch.mean(self.diag, dim=0)

    def get_column(self, i: int) -> torch.Tensor:
        return torch.mean(torch.einsum('mbi, mi -> mb', self.g, self.cov_g[:, i, :]), dim=0)
