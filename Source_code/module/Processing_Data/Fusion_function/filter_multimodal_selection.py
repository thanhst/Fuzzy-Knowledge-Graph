import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler


def compute_feature_importance(X, y):
    # Normalize features
    X = MinMaxScaler().fit_transform(X)

    # Mutual Information
    mi_scores = mutual_info_classif(X, y)

    # Random Forest Importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    rf_scores = rf.feature_importances_

    # Combine scores (average)
    combined_scores = (mi_scores + rf_scores) / 2
    return combined_scores


def remove_correlated_features(X, indices, threshold=0.9):
    selected = []
    for i in indices:
        too_correlated = False
        for j in selected:
            corr = np.corrcoef(X[:, i], X[:, j])[0, 1]
            if abs(corr) > threshold:
                too_correlated = True
                break
        if not too_correlated:
            selected.append(i)
    return selected


def filter_multimodal_selection(Fimg, Ftab, target, k_img=10, k_tab=10, corr_threshold=0.9):
    # Step 1: Feature importance
    img_scores = compute_feature_importance(Fimg, target)
    tab_scores = compute_feature_importance(Ftab, target)

    # Step 2: Sort features
    sorted_img_indices = np.argsort(img_scores)[::-1]
    sorted_tab_indices = np.argsort(tab_scores)[::-1]

    # Step 3: Initial candidate selection (2x)
    candidate_img_indices = sorted_img_indices[:2 * k_img]
    candidate_tab_indices = sorted_tab_indices[:2 * k_tab]

    # Step 4: Remove intra-modal correlation
    final_img_indices = remove_correlated_features(Fimg, candidate_img_indices, corr_threshold)
    final_tab_indices = remove_correlated_features(Ftab, candidate_tab_indices, corr_threshold)

    # Step 5: Truncate
    final_img_indices = final_img_indices[:k_img]
    final_tab_indices = final_tab_indices[:k_tab]

    # Step 6: Concatenate features
    Ffused = np.concatenate((Fimg[:, final_img_indices], Ftab[:, final_tab_indices]), axis=1)

    return Ffused
