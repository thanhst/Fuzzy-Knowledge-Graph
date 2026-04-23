"""
Convert .npz files to CSV format
"""
import numpy as np
import pandas as pd
import os
from pathlib import Path

# Thư mục chứa file .npz
npz_folder = Path(__file__).parent
output_folder = npz_folder / "csv_features"
output_folder.mkdir(exist_ok=True)

splits = ['train', 'val', 'test', 'val_retrieval']

for split in splits:
    npz_file = npz_folder / f"processed_features_{split}.npz"
    
    if not npz_file.exists():
        print(f"⚠ File not found: {npz_file}")
        continue
    
    # Load .npz
    data = np.load(npz_file)
    print(f"\n[{split}] Loaded keys: {list(data.keys())}")
    
    # Lấy fused_features (đặc trưng kết hợp) + labels
    fused_features = data['fused_features']
    labels = data['labels'] if 'labels' in data else None
    hadm_id = data['hadm_id'] if 'hadm_id' in data else None
    
    print(f"  - Fused features shape: {fused_features.shape}")
    if labels is not None:
        print(f"  - Labels shape: {labels.shape}")
    
    # Tạo DataFrame
    n_features = fused_features.shape[1]
    feature_names = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(fused_features, columns=feature_names)
    
    # Thêm columns khác
    if hadm_id is not None:
        df.insert(0, 'hadm_id', hadm_id)
    
    if labels is not None:
        df['label'] = labels
    
    # Lưu CSV
    csv_file = output_folder / f"processed_features_{split}.csv"
    df.to_csv(csv_file, index=False)
    print(f"  ✓ Saved: {csv_file}")

print("\n" + "="*60)
print("✓ Conversion completed!")
print(f"Output folder: {output_folder}")
print("="*60)
