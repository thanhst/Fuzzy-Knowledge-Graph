import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import Ridge

def hadamard_fusion(Fimg, Ftab, common_dim=64, alpha=0.01):
    scaler_img = StandardScaler()
    scaler_tab = StandardScaler()
    Fimg = scaler_img.fit_transform(Fimg)
    Ftab = scaler_tab.fit_transform(Ftab)

    proj_img = Ridge(alpha=alpha, fit_intercept=False)
    proj_tab = Ridge(alpha=alpha, fit_intercept=False)

    X_random = np.random.randn(Fimg.shape[0], common_dim)
    Y_random = np.random.randn(Ftab.shape[0], common_dim)

    proj_img.fit(Fimg, X_random)
    proj_tab.fit(Ftab, Y_random)

    Fimg_proj = proj_img.predict(Fimg)
    Ftab_proj = proj_tab.predict(Ftab)

    Fimg_proj = Fimg_proj / (np.linalg.norm(Fimg_proj, axis=1, keepdims=True) + 1e-8)
    Ftab_proj = Ftab_proj / (np.linalg.norm(Ftab_proj, axis=1, keepdims=True) + 1e-8)

    Ffused = Fimg_proj * Ftab_proj
    Fconcat = np.concatenate([Ffused, np.tanh(Fimg_proj), np.tanh(Ftab_proj)], axis=1)

    return Fconcat
