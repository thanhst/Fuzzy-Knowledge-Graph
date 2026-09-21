import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import MinMaxScaler


def compute_feature_importance(X, y, seed=42):
    X = MinMaxScaler().fit_transform(np.asarray(X, dtype=float))
    mi_scores = mutual_info_classif(X, y, random_state=seed)
    rf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    rf.fit(X, y)
    return (mi_scores + rf.feature_importances_) / 2


def _absolute_correlation(left, right):
    corr = np.corrcoef(left, right)[0, 1]
    return abs(corr) if np.isfinite(corr) else 0.0


def _select_multimodal_indices(Fimg, Ftab, img_scores, tab_scores, k_img, k_tab, corr_threshold):
    if corr_threshold <= 0 or corr_threshold > 1:
        raise ValueError("corr_threshold must be in the interval (0, 1].")

    selected_img = []
    selected_tab = []
    selected_vectors = []
    candidates = []
    candidates.extend(("image", int(i), float(img_scores[i])) for i in np.argsort(img_scores)[::-1])
    candidates.extend(("table", int(i), float(tab_scores[i])) for i in np.argsort(tab_scores)[::-1])
    candidates.sort(key=lambda item: item[2], reverse=True)

    for modality, index, _score in candidates:
        if modality == "image":
            if len(selected_img) >= k_img:
                continue
            vector = Fimg[:, index]
        else:
            if len(selected_tab) >= k_tab:
                continue
            vector = Ftab[:, index]

        if any(_absolute_correlation(vector, selected_vector) > corr_threshold for selected_vector in selected_vectors):
            continue

        if modality == "image":
            selected_img.append(index)
        else:
            selected_tab.append(index)
        selected_vectors.append(vector)

        if len(selected_img) >= k_img and len(selected_tab) >= k_tab:
            break

    return selected_img, selected_tab


def filter_multimodal_selection(
    Fimg,
    Ftab,
    target,
    k_img=10,
    k_tab=10,
    corr_threshold=0.9,
    return_indices=False,
):
    """Filter multimodal features by importance and cross-modal correlation.

    Features from both modalities are ranked together by MI/RF score. A
    candidate is kept only when its absolute Pearson correlation with every
    already selected feature, either image or table, is <= corr_threshold.
    """
    Fimg = np.asarray(Fimg, dtype=float)
    Ftab = np.asarray(Ftab, dtype=float)
    target = np.asarray(target).ravel()

    img_scores = compute_feature_importance(Fimg, target, seed=42)
    tab_scores = compute_feature_importance(Ftab, target, seed=43)
    selected_img, selected_tab = _select_multimodal_indices(
        Fimg,
        Ftab,
        img_scores,
        tab_scores,
        min(int(k_img), Fimg.shape[1]),
        min(int(k_tab), Ftab.shape[1]),
        float(corr_threshold),
    )

    Ffused = np.concatenate([Fimg[:, selected_img], Ftab[:, selected_tab]], axis=1)
    if return_indices:
        return Ffused, selected_img, selected_tab
    return Ffused
