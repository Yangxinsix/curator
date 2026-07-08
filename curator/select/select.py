import inspect
import math
import warnings
from typing import Any, Callable, Optional

import numpy as np
import torch
from .kernel import KernelMatrix


def _call_selection(
    selection_fn: Callable[..., torch.Tensor],
    *,
    selection_kwargs: Optional[dict[str, Any]] = None,
    **base_kwargs: Any,
) -> torch.Tensor:
    """Call a selection function with shared framework kwargs plus user options."""
    signature = inspect.signature(selection_fn)
    params = signature.parameters
    accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values())
    user_kwargs = dict(selection_kwargs or {})
    if not accepts_kwargs:
        unknown = sorted(key for key in user_kwargs if key not in params)
        if unknown:
            name = getattr(selection_fn, "__name__", str(selection_fn))
            raise TypeError(f"{name} got unsupported selection_kwargs: {unknown}")

    call_kwargs = {
        key: value
        for key, value in base_kwargs.items()
        if accepts_kwargs or key in params
    }
    call_kwargs.update(user_kwargs)
    return selection_fn(**call_kwargs)

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


def direct_birch(
    matrix: KernelMatrix,
    batch_size: int,
    n_train: int = 0,
    n_clusters: Optional[int] = None,
    k: int = 1,
    threshold: float = 0.5,
    weighting_pcs: bool = True,
    selection_criteria: str = "center",
    random_state: Optional[int] = None,
    max_threshold_adjustments: int = 20,
) -> torch.Tensor:
    """DIRECT-style PCA + BIRCH stratified sampling over stored features."""
    if not hasattr(matrix, "mat"):
        raise ValueError("direct_birch requires a FeatureKernelMatrix with stored features.")
    if batch_size <= 0:
        return torch.empty(0, dtype=torch.long)
    if k <= 0:
        raise ValueError("k must be positive.")
    if threshold <= 0:
        raise ValueError("threshold must be positive.")
    if selection_criteria not in {"center", "random"}:
        raise ValueError("selection_criteria must be 'center' or 'random'.")

    try:
        from sklearn.cluster import Birch
        from sklearn.decomposition import PCA
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise ImportError("direct_birch requires scikit-learn to be installed.") from exc

    mat = matrix.mat
    n_pool = matrix.get_number_of_columns() - n_train
    if n_pool <= 0:
        return torch.empty(0, dtype=torch.long, device=mat.device)
    if batch_size >= n_pool:
        return torch.arange(n_pool, dtype=torch.long, device=mat.device)

    target_clusters = int(n_clusters) if n_clusters is not None else math.ceil(batch_size / k)
    target_clusters = max(1, min(target_clusters, n_pool))
    features = mat[:, :n_pool, :].detach().permute(1, 0, 2).reshape(n_pool, -1).cpu().numpy()
    features = StandardScaler().fit_transform(features)

    pca = PCA()
    transformed = pca.fit_transform(features)
    n_pcs = int(np.sum(pca.explained_variance_ > 1.0))
    n_pcs = max(1, min(n_pcs, transformed.shape[1]))
    pca_features = transformed[:, :n_pcs]
    if weighting_pcs:
        pca_features = pca_features * pca.explained_variance_ratio_[:n_pcs]

    current_threshold = float(threshold)
    model = None
    for _ in range(max(int(max_threshold_adjustments), 1)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model = Birch(n_clusters=target_clusters, threshold=current_threshold).fit(pca_features)
        n_subclusters = len(set(model.subcluster_labels_))
        if n_subclusters >= target_clusters:
            break
        current_threshold *= max(n_subclusters, 1) / target_clusters
    if model is None:
        raise RuntimeError("BIRCH clustering failed to initialize.")

    labels = model.predict(pca_features)
    label_centers = {
        int(label): center
        for label, center in zip(model.subcluster_labels_, model.subcluster_centers_)
    }
    rng = np.random.default_rng(random_state)
    selected: list[int] = []
    for label in sorted(np.unique(labels)):
        cluster_indices = np.where(labels == label)[0]
        if cluster_indices.size == 0:
            continue
        if selection_criteria == "random":
            size = min(k, cluster_indices.size)
            chosen = rng.choice(cluster_indices, size=size, replace=False)
        elif k >= cluster_indices.size:
            center = label_centers.get(int(label), pca_features[cluster_indices].mean(axis=0))
            distances = np.linalg.norm(pca_features[cluster_indices] - center, axis=1)
            chosen = cluster_indices[np.argsort(distances)]
        else:
            center = label_centers.get(int(label), pca_features[cluster_indices].mean(axis=0))
            distances = np.linalg.norm(pca_features[cluster_indices] - center, axis=1)
            ranked = cluster_indices[np.argsort(distances)]
            positions = np.linspace(0, cluster_indices.size - 1, k).astype(int)
            chosen = ranked[positions]
        selected.extend(int(idx) for idx in chosen)
        if len(selected) >= batch_size:
            break

    selected = selected[:batch_size]
    return torch.tensor(selected, dtype=torch.long, device=mat.device)

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
