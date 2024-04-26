import torch
from .kernel import KernelMatrix

def max_diag(matrix: KernelMatrix, batch_size: int) -> torch.Tensor:
    """
    maximize uncertainty selection method
    """
    return torch.argsort(matrix.get_diag())[-batch_size:]

def max_det_greedy(matrix: KernelMatrix, batch_size: int) -> torch.Tensor:
    vec_c = matrix.get_diag()
    batch_idxs = [torch.argmax(vec_c)]

    l_n = None

    for n in range(1, batch_size):
        opt_idx = batch_idxs[-1]
        l_n_T_l_n = 0.0 if l_n is None else torch.einsum('w,wc->c', l_n[:, opt_idx], l_n)
        mat_col = matrix.get_column(opt_idx)
        update = (1 / torch.sqrt(vec_c[opt_idx])) * (mat_col - l_n_T_l_n)
        vec_c = vec_c - update ** 2
        l_n = update.unsqueeze(0) if l_n is None else torch.concat((l_n, update.unsqueeze(0)))
        new_idx = torch.argmax(vec_c)
        if vec_c[new_idx] <= 1e-12 or new_idx in batch_idxs:
            break
        else:
            batch_idxs.append(new_idx)

    batch_idxs = torch.hstack(batch_idxs)    
    return batch_idxs

def max_det_greedy_local(matrix: KernelMatrix, batch_size: int, num_atoms: torch.Tensor) -> torch.Tensor:
    vec_c = matrix.get_diag()
    batch_idxs = [torch.argmax(vec_c)]

    l_n = None
    image_idx = torch.arange(
        num_atoms.shape[0],
        device=num_atoms.device,                                   
    )
    image_idx = torch.repeat_interleave(image_idx, num_atoms)
    
    selected_idx = []
    n = 0
    while len(selected_idx) < batch_size:
        opt_idx = batch_idxs[-1]
        l_n_T_l_n = 0.0 if l_n is None else torch.einsum('w,wc->c', l_n[:, opt_idx], l_n)
        mat_col = matrix.get_column(opt_idx)
        update = (1 / torch.sqrt(vec_c[opt_idx])) * (mat_col - l_n_T_l_n)
        vec_c = vec_c - update ** 2
        l_n = update.unsqueeze(0) if l_n is None else torch.concat((l_n, update.unsqueeze(0)))
        new_idx = torch.argmax(vec_c)
        if vec_c[new_idx] <= 1e-12 or new_idx in batch_idxs:
            break
        else:
            batch_idxs.append(new_idx)
        if image_idx[new_idx] not in selected_idx:
            selected_idx.append(image_idx[new_idx])
 
    return torch.stack(selected_idx)

def lcmd_greedy(matrix: KernelMatrix, batch_size: int, n_train: int) -> torch.Tensor:
    """
    Only accept matrix with double dtype!!!
    Selects batch elements by greedily picking those with the maximum distance in the largest cluster,
    including training points. Assumes that the last ``n_train`` columns of ``matrix`` correspond to training points.

    :param matrix: Kernel matrix.
    :param batch_size: Size of the AL batch.
    :param n_train: Number of training structures.
    :return: Indices of the selected structures.
    """
    # assumes that the matrix contains pool samples, optionally followed by train samples
    n_pool = matrix.get_number_of_columns() - n_train
    sq_dists = matrix.get_diag()
    batch_idxs = [n_pool if n_train > 0 else torch.argmax(sq_dists)]
    closest_idxs = torch.zeros((n_pool,), dtype=int, device=sq_dists.device)
    min_sq_dists = matrix.get_sq_dists(batch_idxs[-1])[:n_pool]

    for i in range(1, batch_size + n_train):
        if i < n_train:
            batch_idxs.append(n_pool+i)
        else:
            bincount = torch.bincount(closest_idxs, weights=min_sq_dists, minlength=i)
            max_bincount = torch.max(bincount)
            new_idx = torch.argmax(torch.where(
                torch.gather(bincount, 0, closest_idxs) == max_bincount, 
                min_sq_dists, 
                torch.zeros_like(min_sq_dists)-float("Inf")))
            batch_idxs.append(new_idx)
        sq_dists = matrix.get_sq_dists(batch_idxs[-1])[:n_pool]
        new_min = sq_dists < min_sq_dists
        closest_idxs = torch.where(new_min, i, closest_idxs)
        min_sq_dists = torch.where(new_min, sq_dists, min_sq_dists)

    return torch.hstack(batch_idxs[n_train:])

def deterministic_CUR(matrix: KernelMatrix, batch_size: int, lambd: float=0.1, eposilon: float=1E-3) -> torch.Tensor:
    """
    CUR matrix decomposition, the matrix must be normalized.
    """
    n = matrix.num_columns
    W = torch.zeros(n, n)
    I = torch.eye(n, n)
    while True:
        W_t = W
        for i in range(matrix.num_columns):
            z = matrix.get_column(i) @ (I - W) + matrix.get_diag()[i] * W[i]
            coeff = 1 - lambd / torch.linalg.norm(z)
            W[i] = coeff * z if coeff > 0 else 0 * z
        if torch.linalg.norm(W - W_t) < eposilon:
            break
    
    return torch.argsort(torch.linalg.norm(W, dim=1))[-batch_size:]
