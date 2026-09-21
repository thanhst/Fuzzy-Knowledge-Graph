import numpy as np
from sklearn.preprocessing import StandardScaler


def _normalize_rows(values):
    norms = np.linalg.norm(values, axis=1, keepdims=True) + 1e-8
    return values / norms


def _fit_cross_modal_svd(Fimg, Ftab, requested_rank):
    max_rank = min(Fimg.shape[1], Ftab.shape[1])
    rank = min(int(requested_rank), max_rank)
    if rank < 1:
        raise ValueError("common_dim must be at least 1 and both modalities must have features.")

    cross_cov = Fimg.T @ Ftab / max(1, Fimg.shape[0] - 1)
    U, singular_values, Vt = np.linalg.svd(cross_cov, full_matrices=False)
    return U[:, :rank], Vt[:rank, :].T, singular_values[:rank]


def _project_cross_modal(Fimg, Ftab, Wimg, Wtab, singular_values):
    weights = np.sqrt(np.maximum(singular_values, 1e-12))
    Zimg = _normalize_rows((Fimg @ Wimg) * weights)
    Ztab = _normalize_rows((Ftab @ Wtab) * weights)
    return Zimg, Ztab


def hadamard_fusion(Fimg, Ftab, common_dim=64, return_projection=False):
    """Fuse image and table features with train-fitted cross-SVD Hadamard factors.

    W_img and W_tab are learned from C = F_img.T @ F_tab / (n - 1) via
    C = U S V.T. This is the orthogonal objective max trace(W_img.T C W_tab),
    so the learned projections have compatible dimensions before the
    element-wise Hadamard product Z_img * Z_tab.
    """
    Fimg = StandardScaler().fit_transform(np.asarray(Fimg, dtype=float))
    Ftab = StandardScaler().fit_transform(np.asarray(Ftab, dtype=float))

    Wimg, Wtab, singular_values = _fit_cross_modal_svd(Fimg, Ftab, common_dim)
    Zimg, Ztab = _project_cross_modal(Fimg, Ftab, Wimg, Wtab, singular_values)
    Ffused = Zimg * Ztab

    if return_projection:
        return Ffused, Wimg, Wtab, singular_values
    return Ffused
