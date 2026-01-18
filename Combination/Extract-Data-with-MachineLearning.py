"""
Data Processing & Feature Extraction for Symile-MIMIC Dataset
Xử lý 3 phương thức: CXR, ECG, Labs và kết hợp đặc trưng
"""

import numpy as np
import pandas as pd
import torch
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import cv2
from scipy import signal
from typing import Dict, Tuple, List
import argparse
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class CXRFeatureExtractor:

    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        
    def preprocess_image(self, img: np.ndarray) -> np.ndarray:
        img_resized = cv2.resize(img, self.target_size)
        
        if len(img_resized.shape) == 2:
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)

        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # Histogram equalization để tăng contrast
        for i in range(3):
            img_normalized[:, :, i] = cv2.equalizeHist(
                (img_normalized[:, :, i] * 255).astype(np.uint8)
            ).astype(np.float32) / 255.0
        
        return img_normalized
    
    def extract_statistical_features(self, img: np.ndarray) -> np.ndarray:

        features = []
        
        if len(img.shape) == 3:
            gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (img * 255).astype(np.uint8)
        

        features.append(np.mean(gray))          
        features.append(np.std(gray))           
        features.append(np.median(gray))         
        features.append(np.min(gray))            
        features.append(np.max(gray))            
        features.append(np.percentile(gray, 25)) 
        features.append(np.percentile(gray, 75))

        hist, _ = np.histogram(gray.flatten(), bins=256, range=[0, 256])
        hist = hist.astype(np.float32) / hist.sum()
        
        features.append(np.sum(hist * np.log2(hist + 1e-10)))  # Entropy
        features.append(np.sum((np.arange(256) - np.mean(gray))**3 * hist) / (np.std(gray)**3 + 1e-10))  # Skewness
        features.append(np.sum((np.arange(256) - np.mean(gray))**4 * hist) / (np.std(gray)**4 + 1e-10))  # Kurtosis
        
        edges = cv2.Canny(gray, 50, 150)
        features.append(np.sum(edges) / edges.size) 
        
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        features.append(np.mean(magnitude))
        features.append(np.std(magnitude))
        features.append(np.max(magnitude))
        
        dct = cv2.dct(gray.astype(np.float32))
        features.append(np.mean(np.abs(dct[:10, :10]))) 
        features.append(np.mean(np.abs(dct[10:, 10:])))  
        
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            areas = [cv2.contourArea(c) for c in contours]
            features.append(np.max(areas))
            features.append(np.mean(areas))
            features.append(len(contours))
        else:
            features.extend([0, 0, 0])
        
        return np.array(features)
    
    def process_batch(self, images: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        batch_size = images.shape[0]
        
        preprocessed_imgs = np.zeros((batch_size, *self.target_size, 3), dtype=np.float32)
        stat_features = None
        
        for i in range(batch_size):
            img = images[i]

            if img.ndim == 3 and (img.shape[0] == 3 or img.shape[0] == 1):
                img = np.transpose(img, (1, 2, 0))

            preprocessed_imgs[i] = self.preprocess_image(img)
            
            feats = self.extract_statistical_features(preprocessed_imgs[i])
            if stat_features is None:
                stat_features = np.zeros((batch_size, feats.shape[0]), dtype=np.float32)
            stat_features[i] = feats
        
        return preprocessed_imgs, stat_features

class ECGFeatureExtractor:

    def __init__(self, sampling_rate=500):
        self.fs = sampling_rate
        
    def denoise_signal(self, ecg_signal: np.ndarray) -> np.ndarray:

        sos = signal.butter(4, [0.5, 40], btype='bandpass', fs=self.fs, output='sos')
        filtered_signal = signal.sosfiltfilt(sos, ecg_signal, axis=1)
        
        return filtered_signal
    
    def extract_temporal_features(self, ecg_signal: np.ndarray) -> np.ndarray:

        if ecg_signal.ndim != 2:
            raise ValueError(f"ECG signal must be 2D (num_leads, signal_length), got shape {ecg_signal.shape}")
        num_leads = ecg_signal.shape[0]
        features_per_lead = []
        
        for lead in range(num_leads):
            signal_lead = np.asarray(ecg_signal[lead]).astype(np.float32).reshape(-1)
            lead_features = []

            lead_features.append(np.mean(signal_lead))
            lead_features.append(np.std(signal_lead))
            lead_features.append(np.min(signal_lead))
            lead_features.append(np.max(signal_lead))
            lead_features.append(np.median(signal_lead))
            lead_features.append(np.percentile(signal_lead, 25))
            lead_features.append(np.percentile(signal_lead, 75))
            
            # Peak detection
            peaks, _ = signal.find_peaks(signal_lead, distance=int(0.6 * self.fs))
            lead_features.append(len(peaks))  # Number of peaks (R-peaks)
            
            if len(peaks) > 1:
                rr_intervals = np.diff(peaks) / self.fs  # RR intervals in seconds
                lead_features.append(np.mean(rr_intervals))  # Mean RR
                lead_features.append(np.std(rr_intervals))   # SDNN (HRV metric)
                lead_features.append(60 / np.mean(rr_intervals))  # Heart rate
            else:
                lead_features.extend([0, 0, 0])
            
            # Zero crossing rate
            zcr = np.sum(np.diff(np.sign(signal_lead)) != 0) / len(signal_lead)
            lead_features.append(zcr)
            
            features_per_lead.append(lead_features)
        
        return np.array(features_per_lead).flatten()
    
    def extract_frequency_features(self, ecg_signal: np.ndarray) -> np.ndarray:
        if ecg_signal.ndim != 2:
            raise ValueError(f"ECG signal must be 2D (num_leads, signal_length), got shape {ecg_signal.shape}")
        num_leads = ecg_signal.shape[0]
        freq_features = []
        
        for lead in range(num_leads):
            signal_lead = np.asarray(ecg_signal[lead]).astype(np.float32).reshape(-1)
            fft_vals = np.fft.fft(signal_lead)
            fft_freq = np.fft.fftfreq(len(signal_lead), 1/self.fs)
            
            psd = np.abs(fft_vals)**2

            vlf_band = (fft_freq >= 0.003) & (fft_freq < 0.04)
            lf_band = (fft_freq >= 0.04) & (fft_freq < 0.15)
            hf_band = (fft_freq >= 0.15) & (fft_freq < 0.4)
            
            freq_features.append(np.sum(psd[vlf_band])) 
            freq_features.append(np.sum(psd[lf_band]))   
            freq_features.append(np.sum(psd[hf_band]))   
            

            dominant_freq_idx = np.argmax(psd[:len(psd)//2])
            freq_features.append(fft_freq[dominant_freq_idx])
            
            psd_norm = psd / np.sum(psd)
            spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
            freq_features.append(spectral_entropy)
        
        return np.array(freq_features)
    
    def process_batch(self, ecg_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        if ecg_batch.ndim == 4 and ecg_batch.shape[1] == 1:
            ecg_batch = np.squeeze(ecg_batch, axis=1)  
        if ecg_batch.ndim != 3:
            raise ValueError(f"ECG batch must be 3D (batch, num_leads, signal_length), got shape {ecg_batch.shape}")

        if ecg_batch.shape[1] == 12:
            
            formatted = ecg_batch
        elif ecg_batch.shape[2] == 12:
            formatted = np.transpose(ecg_batch, (0, 2, 1))
        else:
            raise ValueError(f"Cannot infer ECG layout, expected one dimension to be 12 leads, got shape {ecg_batch.shape}")

        batch_size = formatted.shape[0]
        
        denoised = np.zeros_like(formatted)
        all_features = []
        
        for i in range(batch_size):
            # Denoise
            denoised[i] = self.denoise_signal(formatted[i])
            
            # Extract features
            temporal_feats = self.extract_temporal_features(denoised[i])
            freq_feats = self.extract_frequency_features(denoised[i])
            
            # Combine
            combined = np.concatenate([temporal_feats, freq_feats])
            all_features.append(combined)
        
        return denoised, np.array(all_features)


class LabsFeatureSelector:
    def __init__(self, method='random_forest', n_features=20):
        self.method = method
        self.n_features = n_features
        self.selector = None
        self.scaler = StandardScaler()
        self.selected_indices = None
        
    def handle_missing_values(self, labs_data: np.ndarray, 
                              missingness: np.ndarray) -> np.ndarray:

        imputed = labs_data.copy()
        
        for i in range(labs_data.shape[1]):
            mask = missingness[:, i] == 0  
            if np.sum(~mask) > 0:  
                mean_val = np.mean(labs_data[~mask, i])
                imputed[mask, i] = mean_val
        
        return imputed
    
    def fit(self, labs_data: np.ndarray, missingness: np.ndarray,
            labels: np.ndarray | None):

        if self.method != 'pca' and labels is None:
            raise ValueError(
                "LabsFeatureSelector.fit() requires labels when method != 'pca'. "
                "Provide a labeled split (e.g., test/val_retrieval) or use --labs-selection-method pca."
            )
       
        imputed = self.handle_missing_values(labs_data, missingness)
        
        combined = np.concatenate([imputed, missingness], axis=1)  # (n, 100)
        
        combined_scaled = self.scaler.fit_transform(combined)
        
        print(f"Original features: {combined.shape[1]}")

        if self.method == 'random_forest':
            rf = RandomForestClassifier(n_estimators=100, random_state=42, 
                                       max_depth=10, n_jobs=-1)
            rf.fit(combined_scaled, labels)

            importances = rf.feature_importances_
            indices = np.argsort(importances)[::-1][:self.n_features]
            
            self.selected_indices = indices
            self.selector = rf
            
            print(f"Random Forest - Selected {self.n_features} features")
            print(f"Top 5 feature importances: {importances[indices[:5]]}")
            
        elif self.method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_classif, 
                                  k=self.n_features)
            selector.fit(combined_scaled, labels)
            
            self.selected_indices = selector.get_support(indices=True)
            self.selector = selector
            
            print(f"Mutual Information - Selected {self.n_features} features")
            print(f"Top 5 scores: {selector.scores_[self.selected_indices[:5]]}")
            
        elif self.method == 'f_test':
            selector = SelectKBest(score_func=f_classif, k=self.n_features)
            selector.fit(combined_scaled, labels)
            
            self.selected_indices = selector.get_support(indices=True)
            self.selector = selector
            
            print(f"F-test - Selected {self.n_features} features")
            print(f"Top 5 F-scores: {selector.scores_[self.selected_indices[:5]]}")
            
        elif self.method == 'pca':
            pca = PCA(n_components=self.n_features)
            pca.fit(combined_scaled)
            
            self.selector = pca
            self.selected_indices = None  # PCA transforms, không select indices
            
            print(f"PCA - Reduced to {self.n_features} components")
            print(f"Explained variance ratio: {np.sum(pca.explained_variance_ratio_):.4f}")
        
        return self
    
    def transform(self, labs_data: np.ndarray, 
                  missingness: np.ndarray) -> np.ndarray:

        imputed = self.handle_missing_values(labs_data, missingness)
        
        combined = np.concatenate([imputed, missingness], axis=1)
        
        combined_scaled = self.scaler.transform(combined)

        if self.method == 'pca':
            selected = self.selector.transform(combined_scaled)
        else:
            selected = combined_scaled[:, self.selected_indices]
        
        return selected
    
    def get_feature_names(self, original_names: List[str]) -> List[str]:
        if self.method == 'pca':
            return [f'PC{i+1}' for i in range(self.n_features)]
        else:
            all_names = original_names + [f'{name}_missing' for name in original_names]
            return [all_names[i] for i in self.selected_indices]

class MultimodalDataProcessor:

    def __init__(self, labs_selection_method='random_forest', 
                 n_labs_features=20):
        self.cxr_extractor = CXRFeatureExtractor()
        self.ecg_extractor = ECGFeatureExtractor()
        self.labs_selector = LabsFeatureSelector(
            method=labs_selection_method,
            n_features=n_labs_features
        )
        
        self.is_fitted = False
        
    def fit(self, cxr_data: np.ndarray, ecg_data: np.ndarray,
            labs_data: np.ndarray, labs_missingness: np.ndarray,
            labels: np.ndarray | None):
        print("\n" + "="*60)
        print("FITTING MULTIMODAL DATA PROCESSOR")
        print("="*60)
        
        # Fit Labs selector
        print("\n[Labs Feature Selection]")
        self.labs_selector.fit(labs_data, labs_missingness, labels)
        
        self.is_fitted = True
        print("\n✓ Processor fitted successfully!")
        
    def process(self, cxr_data: np.ndarray, ecg_data: np.ndarray,
                labs_data: np.ndarray, labs_missingness: np.ndarray) -> Dict:
      
        print("\n" + "="*60)
        print("PROCESSING MULTIMODAL DATA")
        print("="*60)
        
        batch_size = cxr_data.shape[0]
        
        # 1. Process CXR
        print(f"\n[1/3] Processing CXR data ({batch_size} samples)...")
        cxr_images, cxr_features = self.cxr_extractor.process_batch(cxr_data)
        print(f"  ✓ CXR images shape: {cxr_images.shape}")
        print(f"  ✓ CXR features shape: {cxr_features.shape}")
        
        # 2. Process ECG
        print(f"\n[2/3] Processing ECG data ({batch_size} samples)...")
        ecg_signals, ecg_features = self.ecg_extractor.process_batch(ecg_data)
        print(f"  ✓ ECG signals shape: {ecg_signals.shape}")
        print(f"  ✓ ECG features shape: {ecg_features.shape}")
        
        # 3. Process Labs
        print(f"\n[3/3] Processing Labs data ({batch_size} samples)...")
        if not self.is_fitted:
            raise ValueError("Labs selector chưa được fit! Gọi fit() trước.")
        
        labs_features = self.labs_selector.transform(labs_data, labs_missingness)
        print(f"  ✓ Labs features shape: {labs_features.shape}")
        
        # 4. Fusion
        print("\n[Fusion] Combining all features...")
        fused_features = np.concatenate([
            cxr_features,    # (batch, 21)
            ecg_features,    # (batch, ecg_dim)
            labs_features    # (batch, n_labs_features)
        ], axis=1)
        
        print(f"  ✓ Fused features shape: {fused_features.shape}")
        
        print("\n" + "="*60)
        print("FEATURE DIMENSIONS SUMMARY")
        print("="*60)
        print(f"CXR features:    {cxr_features.shape[1]}")
        print(f"ECG features:    {ecg_features.shape[1]}")
        print(f"Labs features:   {labs_features.shape[1]}")
        print(f"─" * 60)
        print(f"TOTAL (Fused):   {fused_features.shape[1]}")
        print("="*60)
        
        return {
            'cxr_images': cxr_images,
            'cxr_features': cxr_features,
            'ecg_signals': ecg_signals,
            'ecg_features': ecg_features,
            'labs_features': labs_features,
            'fused_features': fused_features
        }


def _load_optional_labels_from_csv(csv_path: Path, label_column: str) -> np.ndarray | None:
    if not csv_path.exists():
        return None
    # đọc header trước
    with csv_path.open("r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
    if label_column not in header:
        return None
    df = pd.read_csv(csv_path, usecols=[label_column])
    return df[label_column].to_numpy()


def load_symile_split(
    data_npy_root: Path,
    split: str,
    csv_root: Path | None = None,
    label_column: str | None = None,
) -> Dict[str, np.ndarray]:

    split_dir = data_npy_root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    def _npy(name: str) -> Path:
        return split_dir / f"{name}_{split}.npy"

    cxr = np.load(_npy("cxr"))
    ecg = np.load(_npy("ecg"))
    labs = np.load(_npy("labs_percentiles"))
    missingness = np.load(_npy("labs_missingness"))
    hadm_id = np.load(_npy("hadm_id"))

    labels = None
    label_path = split_dir / f"label_{split}.npy"
    if label_path.exists():
        labels = np.load(label_path)
    elif csv_root is not None and label_column is not None:
        csv_path = csv_root / f"{split}.csv"
        labels = _load_optional_labels_from_csv(csv_path, label_column)

    if labels is not None:
        labels = np.asarray(labels).astype(np.float32)
        # Map -1 -> 0
        labels = np.where(labels == -1, 0, labels)
        # Remove NaN labels
        valid_mask = ~np.isnan(labels)
        if not np.all(valid_mask):
            n_removed = int((~valid_mask).sum())
            print(f"[{split}] Removing {n_removed} samples with NaN labels")
            cxr = cxr[valid_mask]
            ecg = ecg[valid_mask]
            labs = labs[valid_mask]
            missingness = missingness[valid_mask]
            hadm_id = hadm_id[valid_mask]
            labels = labels[valid_mask]
        labels = labels.astype(np.int64)

    cxr = cxr.astype(np.float32, copy=False)
    ecg = ecg.astype(np.float32, copy=False)
    labs = labs.astype(np.float32, copy=False)
    missingness = missingness.astype(np.float32, copy=False)
    return {
        "cxr": cxr,
        "ecg": ecg,
        "labs": labs,
        "missingness": missingness,
        "hadm_id": hadm_id,
        "labels": labels,
    }


def main():
    
    parser = argparse.ArgumentParser(description="Symile-MIMIC Multimodal Feature Extraction (Machine Learning)")
    parser.add_argument(
        "--data-npy-root",
        type=str,
        default=str(Path("Symile") / "symile-mimic-a-multimodal-clinical-dataset-of-chest-x-rays-electrocardiograms-and-blood-labs-from-mimic-iv-1.0.0" / "data_npy"),
        help="Directory containing data_npy/{train,val,test,val_retrieval}",
    )
    parser.add_argument(
        "--csv-root",
        type=str,
        default=str(Path("Symile") / "symile-mimic-a-multimodal-clinical-dataset-of-chest-x-rays-electrocardiograms-and-blood-labs-from-mimic-iv-1.0.0"),
        help="Directory containing train.csv/val.csv/test.csv/val_retrieval.csv (optional label source).",
    )
    parser.add_argument(
        "--fit-split",
        type=str,
        default="train",
        help="Split used to fit the Labs selector (needs labels unless method=pca).",
    )
    parser.add_argument(
        "--process-splits",
        type=str,
        nargs="+",
        default=["train", "val", "test", "val_retrieval"],
        help="Splits to process and save as .npz.",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="Pleural Effusion",
        help="Column name in CSV to use as label when no label_*.npy exists (e.g., 'Pleural Effusion').",
    )
    parser.add_argument("--labs-selection-method", type=str, default="random_forest", choices=["random_forest", "mutual_info", "f_test", "pca"])
    parser.add_argument("--n-labs-features", type=int, default=20)
    parser.add_argument("--output-dir", type=str, default="processed_outputs", help="Output directory.")
    args = parser.parse_args()

    data_npy_root = Path(args.data_npy_root)
    csv_root = Path(args.csv_root) if args.csv_root else None
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load fit split (labels from label_*.npy or CSV label_column)
    fit_pack = load_symile_split(
        data_npy_root,
        args.fit_split,
        csv_root=csv_root,
        label_column=args.label_column,
    )
    fit_labels = fit_pack["labels"]

    labs_method = args.labs_selection_method
    if fit_labels is None and labs_method != "pca":
        print(
            f"\n[WARN] Split '{args.fit_split}' has no labels. "
            f"Auto-switching --labs-selection-method from '{labs_method}' to 'pca' (no labels required)."
        )
        labs_method = "pca"

    processor = MultimodalDataProcessor(
        labs_selection_method=labs_method,
        n_labs_features=args.n_labs_features,
    )

    processor.fit(
        fit_pack["cxr"],
        fit_pack["ecg"],
        fit_pack["labs"],
        fit_pack["missingness"],
        fit_labels,
    )

    # Process each split
    for split in args.process_splits:
        pack = load_symile_split(
            data_npy_root,
            split,
            csv_root=csv_root,
            label_column=args.label_column,
        )
        results = processor.process(pack["cxr"], pack["ecg"], pack["labs"], pack["missingness"])
        # kèm metadata
        results["hadm_id"] = pack["hadm_id"]
        if pack["labels"] is not None:
            results["labels"] = pack["labels"]

        out_path = out_dir / f"processed_features_{split}.npz"
        np.savez_compressed(out_path, **results)
        print(f"\n✓ Saved: {out_path}")


if __name__ == '__main__':
    main()