import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


def evaluate_feature_set(X, y, model=None, cv=5):
    if model is None:
        model = RandomForestClassifier(random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    return scores.mean()


def add_best_feature(X, selected_indices, y, remaining_indices):
    best_score = -np.inf
    best_feature = None
    for i in remaining_indices:
        temp_indices = selected_indices + [i]
        score = evaluate_feature_set(X[:, temp_indices], y)
        if score > best_score:
            best_score = score
            best_feature = i
    return best_feature, best_score


def wrapper_multimodal_selection(Fimg, Ftab, target, max_img=5, max_tab=5, min_img=1, min_tab=1):
    selected_img = []
    selected_tab = []
    best_score = -np.inf

    img_indices = list(range(Fimg.shape[1]))
    tab_indices = list(range(Ftab.shape[1]))

    # Step 1: Ensure minimum features from each modality
    for _ in range(min_img):
        best_feature, _ = add_best_feature(Fimg, selected_img, target, list(set(img_indices) - set(selected_img)))
        if best_feature is not None:
            selected_img.append(best_feature)

    for _ in range(min_tab):
        best_feature, _ = add_best_feature(Ftab, selected_tab, target, list(set(tab_indices) - set(selected_tab)))
        if best_feature is not None:
            selected_tab.append(best_feature)

    # Step 2: Sequential Forward Selection
    while len(selected_img) < max_img or len(selected_tab) < max_tab:
        best_new_score = -np.inf
        best_new_feature = None
        best_modality = None

        # Try image features
        for i in set(img_indices) - set(selected_img):
            fused = np.concatenate([
                Fimg[:, selected_img + [i]],
                Ftab[:, selected_tab]
            ], axis=1)
            score = evaluate_feature_set(fused, target)
            if score > best_new_score:
                best_new_score = score
                best_new_feature = i
                best_modality = 'img'

        # Try tabular features
        for j in set(tab_indices) - set(selected_tab):
            fused = np.concatenate([
                Fimg[:, selected_img],
                Ftab[:, selected_tab + [j]]
            ], axis=1)
            score = evaluate_feature_set(fused, target)
            if score > best_new_score:
                best_new_score = score
                best_new_feature = j
                best_modality = 'tab'

        # Update sets
        if best_new_score > best_score:
            best_score = best_new_score
            if best_modality == 'img':
                selected_img.append(best_new_feature)
            else:
                selected_tab.append(best_new_feature)
        else:
            break  # No improvement

    Ffused = np.concatenate([
        Fimg[:, selected_img],
        Ftab[:, selected_tab]
    ], axis=1)

    return Ffused