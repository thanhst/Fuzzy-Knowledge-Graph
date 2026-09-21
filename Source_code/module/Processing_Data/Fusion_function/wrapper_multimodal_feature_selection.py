import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


def evaluate_feature_set(X, y, model=None, cv=5):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).ravel()
    if X.shape[1] == 0:
        return -np.inf

    label_counts = pd.Series(y).value_counts()
    actual_cv = min(int(cv), int(label_counts.min()))
    if model is None:
        model = RandomForestClassifier(random_state=42, n_jobs=-1)
    if actual_cv < 2:
        model.fit(X, y)
        return float(model.score(X, y))
    scores = cross_val_score(model, X, y, cv=actual_cv, scoring="accuracy")
    return float(scores.mean())


def add_best_feature(X, selected_indices, y, remaining_indices, cache, cache_prefix):
    best_score = None
    best_feature = None
    for index in sorted(remaining_indices):
        trial = selected_indices + [index]
        cache_key = (cache_prefix, tuple(trial))
        if cache_key not in cache:
            cache[cache_key] = evaluate_feature_set(X[:, trial], y)
        score = cache[cache_key]
        if best_score is None or score > best_score:
            best_score = score
            best_feature = index
    return best_feature, best_score


def wrapper_multimodal_selection(
    Fimg,
    Ftab,
    target,
    max_img=5,
    max_tab=5,
    min_img=1,
    min_tab=1,
    return_indices=False,
):
    """Sequential wrapper selection with a real baseline after min features.

    The initialized feature set is evaluated before the forward loop. A new
    feature is accepted only if it improves that score, and each modality is
    considered only while it remains below its configured maximum.
    """
    Fimg = np.asarray(Fimg, dtype=float)
    Ftab = np.asarray(Ftab, dtype=float)
    target = np.asarray(target).ravel()

    max_img = min(max(0, int(max_img)), Fimg.shape[1])
    max_tab = min(max(0, int(max_tab)), Ftab.shape[1])
    min_img = min(max(0, int(min_img)), max_img)
    min_tab = min(max(0, int(min_tab)), max_tab)

    selected_img = []
    selected_tab = []
    img_indices = set(range(Fimg.shape[1]))
    tab_indices = set(range(Ftab.shape[1]))
    score_cache = {}

    for _ in range(min_img):
        best_feature, _ = add_best_feature(
            Fimg,
            selected_img,
            target,
            img_indices - set(selected_img),
            score_cache,
            "min_image",
        )
        if best_feature is not None:
            selected_img.append(best_feature)

    for _ in range(min_tab):
        best_feature, _ = add_best_feature(
            Ftab,
            selected_tab,
            target,
            tab_indices - set(selected_tab),
            score_cache,
            "min_table",
        )
        if best_feature is not None:
            selected_tab.append(best_feature)

    fused = np.concatenate([Fimg[:, selected_img], Ftab[:, selected_tab]], axis=1)
    best_score = evaluate_feature_set(fused, target)

    while True:
        best_new_score = None
        best_new_feature = None
        best_modality = None

        if len(selected_img) < max_img:
            for index in sorted(img_indices - set(selected_img)):
                fused_trial = np.concatenate(
                    [Fimg[:, selected_img + [index]], Ftab[:, selected_tab]],
                    axis=1,
                )
                cache_key = ("image", tuple(selected_img + [index]), tuple(selected_tab))
                if cache_key not in score_cache:
                    score_cache[cache_key] = evaluate_feature_set(fused_trial, target)
                score = score_cache[cache_key]
                if best_new_score is None or score > best_new_score:
                    best_new_score = score
                    best_new_feature = index
                    best_modality = "image"

        if len(selected_tab) < max_tab:
            for index in sorted(tab_indices - set(selected_tab)):
                fused_trial = np.concatenate(
                    [Fimg[:, selected_img], Ftab[:, selected_tab + [index]]],
                    axis=1,
                )
                cache_key = ("table", tuple(selected_img), tuple(selected_tab + [index]))
                if cache_key not in score_cache:
                    score_cache[cache_key] = evaluate_feature_set(fused_trial, target)
                score = score_cache[cache_key]
                if best_new_score is None or score > best_new_score:
                    best_new_score = score
                    best_new_feature = index
                    best_modality = "table"

        if best_new_feature is None or best_new_score <= best_score:
            break

        best_score = best_new_score
        if best_modality == "image":
            selected_img.append(best_new_feature)
        else:
            selected_tab.append(best_new_feature)

    Ffused = np.concatenate([Fimg[:, selected_img], Ftab[:, selected_tab]], axis=1)
    if return_indices:
        return Ffused, selected_img, selected_tab
    return Ffused
