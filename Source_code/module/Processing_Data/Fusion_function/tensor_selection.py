from sklearn.decomposition import TruncatedSVD
import numpy as np
def tensor_fusion(Fimg, Ftab, rank=10):
    flattened_tensors = []
    for f_img, f_tab in zip(Fimg, Ftab):
        tensor = np.outer(f_img, f_tab)
        flat_tensor = tensor.flatten()
        flattened_tensors.append(flat_tensor)

    X_tensor = np.stack(flattened_tensors)  # shape: (n_samples, f_img_dim * f_tab_dim)

    # Bước 2: Giảm chiều với SVD
    svd = TruncatedSVD(n_components=rank)
    reduced = svd.fit_transform(X_tensor)  # shape: (n_samples, rank)

    # Bước 3: Lấy các thành phần chính nếu cần
    # components = svd.components_.flatten()[:rank]  # flatten (rank, n_features) -> lấy top rank thành phần

    # Bước 4: Kết hợp lại
    # fused_features = np.concatenate([reduced, np.tile(components, (reduced.shape[0], 1))], axis=1)

    return reduced
