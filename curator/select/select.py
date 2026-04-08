import torch
from .kernel import KernelMatrix

def max_diag(matrix: KernelMatrix, batch_size: int) -> torch.Tensor:
    """
    maximize uncertainty selection method
    """
    return torch.argsort(matrix.get_diag())[-batch_size:]

def max_dist_greedy(matrix: KernelMatrix, batch_size: int, n_train: int = 0) -> torch.Tensor:
    """Greedily select points that maximise the distance to the current set.

    When ``n_train`` is provided, the last ``n_train`` columns are considered
    already-selected (training) points and are used to initialise the distance
    landscape before picking new pool elements.
    """

    n_pool = matrix.get_number_of_columns() - n_train
    if n_pool <= 0 or batch_size <= 0:
        return torch.empty(0, dtype=torch.long)

    diag = matrix.get_diag()[:n_pool]
    device = diag.device
    dtype = diag.dtype

    if n_train > 0:
        min_sq_dists = torch.full((n_pool,), float('inf'), device=device, dtype=dtype)
        for j in range(n_pool, n_pool + n_train):
            train_dists = matrix.get_sq_dists(j)[:n_pool]
            min_sq_dists = torch.minimum(min_sq_dists, train_dists)
        start_idx = torch.argmax(min_sq_dists)
    else:
        min_sq_dists = diag.clone()
        start_idx = torch.argmax(diag)

    selected = [start_idx]
    selected_mask = torch.zeros(n_pool, dtype=torch.bool, device=device)
    selected_mask[start_idx] = True

    while len(selected) < min(batch_size, n_pool):
        sq_dists = matrix.get_sq_dists(selected[-1])[:n_pool]
        min_sq_dists = torch.minimum(min_sq_dists, sq_dists)
        min_sq_dists = min_sq_dists.masked_fill(selected_mask, float('-inf'))
        next_idx = torch.argmax(min_sq_dists)
        if selected_mask[next_idx]:
            break
        selected.append(next_idx)
        selected_mask[next_idx] = True

    return torch.tensor(selected, device=device, dtype=torch.long)

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
    import time
    import logging
    logger = logging.getLogger(__name__)
    
    start_time = time.time()
    
    # assumes that the matrix contains pool samples, optionally followed by train samples
    n_pool = matrix.get_number_of_columns() - n_train
    sq_dists = matrix.get_diag()
    batch_idxs = [n_pool if n_train > 0 else torch.argmax(sq_dists)]
    closest_idxs = torch.zeros((n_pool,), dtype=int, device=sq_dists.device)
    min_sq_dists = matrix.get_sq_dists(batch_idxs[-1])[:n_pool]
    
    init_time = time.time()
    logger.info(f"[lcmd_greedy] Initialization: {init_time - start_time:.3f}s, n_pool={n_pool}, batch_size={batch_size}")
    
    loop_start = time.time()
    get_sq_dists_total = 0.0
    
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
        
        t0 = time.time()
        sq_dists = matrix.get_sq_dists(batch_idxs[-1])[:n_pool]
        get_sq_dists_total += time.time() - t0
        
        new_min = sq_dists < min_sq_dists
        closest_idxs = torch.where(new_min, i, closest_idxs)
        min_sq_dists = torch.where(new_min, sq_dists, min_sq_dists)

    loop_time = time.time() - loop_start
    total_time = time.time() - start_time
    
    logger.info(f"[lcmd_greedy] Loop time: {loop_time:.3f}s")
    logger.info(f"[lcmd_greedy] get_sq_dists total time: {get_sq_dists_total:.3f}s ({get_sq_dists_total/loop_time*100:.1f}%)")
    logger.info(f"[lcmd_greedy] Total time: {total_time:.3f}s")
    
    return torch.hstack(batch_idxs[n_train:])


def lcmd_greedy_faiss(matrix: 'FaissKernelMatrix', batch_size: int, n_train: int) -> torch.Tensor:
    """
    Faiss accelerated lcmd_greedy
    
    Args:
        matrix: FaissKernelMatrix instance
        batch_size: Number of points to select
        n_train: Number of training points (last n_train columns)
        
    Returns:
        Selected indices from the pool
    """
    import numpy as np
    import time
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        import faiss
    except ImportError:
        raise RuntimeError("Faiss is required for lcmd_greedy_faiss")
    
    start_time = time.time()
    
    n_total = matrix.get_number_of_columns()
    n_pool = n_total - n_train
    features = matrix.get_features()
    n_features = features.shape[1]
    device = matrix._device
    
    if n_train > 0:
        selected_global_idxs = list(range(n_pool, n_pool + n_train))
    else:
        sq_dists = matrix.get_diag()
        first_idx = torch.argmax(sq_dists).item()
        selected_global_idxs = [first_idx]
    
    if matrix.use_gpu:
        selected_index = faiss.GpuIndexFlatL2(matrix.res, n_features)
    else:
        selected_index = faiss.IndexFlatL2(n_features)
    
    selected_index.add(features[selected_global_idxs])
    
    pool_features = features[:n_pool].astype(np.float32)
    
    min_sq_dists, closest_idxs = selected_index.search(pool_features, k=1)
    min_sq_dists = min_sq_dists.flatten()
    closest_idxs = closest_idxs.flatten()
    
    init_time = time.time()
    logger.info(f"[lcmd_greedy_faiss] Initialization: {init_time - start_time:.3f}s, n_pool={n_pool}, batch_size={batch_size}")
    
    selected_from_pool = []
    
    loop_start = time.time()
    dist_calc_total = 0.0
    
    for i in range(batch_size):
        n_selected = len(selected_global_idxs)
        bincount = np.bincount(closest_idxs, weights=min_sq_dists, minlength=n_selected)
        
        max_weight = np.max(bincount)
        
        cluster_weights = bincount[closest_idxs]
        is_in_max_cluster = (cluster_weights == max_weight)
        
        masked_dists = np.where(is_in_max_cluster, min_sq_dists, -np.inf)
        new_idx = int(np.argmax(masked_dists))
        
        if new_idx in selected_from_pool:
            break
            
        selected_from_pool.append(new_idx)
        selected_global_idxs.append(new_idx)
        
        new_feature = features[new_idx:new_idx+1].astype(np.float32)
        selected_index.add(new_feature)
        
        new_idx_in_selected = len(selected_global_idxs) - 1
        
        t0 = time.time()
        new_dists = np.sum((pool_features - new_feature) ** 2, axis=1)
        dist_calc_total += time.time() - t0
        
        is_closer = new_dists < min_sq_dists
        min_sq_dists = np.where(is_closer, new_dists, min_sq_dists)
        closest_idxs = np.where(is_closer, new_idx_in_selected, closest_idxs)
        
        min_sq_dists[new_idx] = -np.inf
    
    loop_time = time.time() - loop_start
    total_time = time.time() - start_time
    
    logger.info(f"[lcmd_greedy_faiss] Loop time: {loop_time:.3f}s")
    logger.info(f"[lcmd_greedy_faiss] Distance calculation time: {dist_calc_total:.3f}s ({dist_calc_total/loop_time*100:.1f}%)")
    logger.info(f"[lcmd_greedy_faiss] Total time: {total_time:.3f}s")
    
    return torch.tensor(selected_from_pool, device=device, dtype=torch.long)

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
