# Diabetic Retinopathy Diagnosis Using FKGS Method

This project applies image processing and machine learning techniques to analyze fundus images and patient metadata for the purpose of diagnosing diabetic retinopathy. It combines features extracted from images with fuzzy systems and knowledge graphs to improve prediction accuracy and speed.

## 📌 Overview

The system includes the following components:

- **Image preprocessing**
- **Feature extraction** using **GLCM (Gray Level Co-occurrence Matrix)**
- **Integration of metadata** with extracted image features
- **Fuzzy Inference System (FIS)** and **Fuzzy Knowledge Graph (FKG)**
- Advanced reasoning using **Fuzzy Knowledge Graph Sampling (FKGS)**

Libraries used include OpenCV and `scikit-image` for image processing, and various Python modules for building fuzzy models and graphs.

## ⚙️ System Requirements

- Python 3.x  
- Operating System: **Windows**
- Required Libraries:
  - OpenCV
  - scikit-image
  - NumPy
  - Pandas

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/thanhst/Fuzzy-Knowledge-Graph.git
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment

Ensure Python 3.x is installed and all libraries listed in `requirements.txt` are installed correctly.

## 📁 Project Structure

```
📦 Project
├── 📁 Source_code
│   ├── base/                         # Core theories and base models
│   ├── data/
│   │   ├── BaseData/                # Preliminary experimental data
│   │   ├── Dataset/                 # Initial dataset
│   │   ├── Dataset_diabetic/       # Diabetic retinopathy scenarios
│   │   ├── FIS/input/              # FIS model inputs
│   │   ├── FIS/output/             # Outputs including FRB, rule lists, etc.
│   │   ├── FKG/                    # FKG algorithm outputs
│   │   └── Metadata/Metadata.csv   # Patient metadata
│   ├── main/                        # Main execution scenarios
│   ├── models/                      # Model outputs by scenario
│   └── module/                      # Supporting modules
│       ├── Convert/                 # Converts numeric rules to linguistic ones
│       ├── FCM/                     # Fuzzy clustering
│       ├── FIS/                     # FIS model
│       ├── FKG/                     # All FKG algorithms (multi-label, sampling, etc.)
│       ├── Helper/                  # Utility functions
│       ├── Membership_Function/     # Membership functions: Gaussian, Sigmoid, etc.
│       ├── Module_CPP/              # C++ modules for high-performance calculations
│       ├── Processing_Data/         # Preprocessing for each scenario
│       ├── Rules_Function/          # Rule generation, reduction, weighting
│       └── Setup_module/            # CMake & C++ for Python C++ modules
├── *.bat                            # Batch scripts for scenario execution
└── README.md
```

## ▶️ How to Run

- Navigate to the root folder.
- Execute any `.bat` script depending on the feature type:
  - `Scenario_diabetic_retinopathy_GLCM_feature.bat` – GLCM image features
  - `Scenario_diabetic_retinopathy_statistical_feature.bat` – Statistical features
  - `Scenario_diabetic_retinopathy_table_feature.bat` – Metadata features
  - `Scenario_diabetic_retinopathy_fusion_feature.bat` – Combined features (GLCM + statistical + metadata)

### 🧪 Fusion Cases

- **Two-modal fusion** (`fusion-case` folder):  
  Scenarios for combining image and metadata using:
  - Wrapper selection
  - Feature selection
  - Filter-based selection
  - Hadamard
  - Tensor selection

- **Three-modal fusion** (`Multimodality` folder):  
  Combinations:
  - Fundus + Table
  - Fundus + OCT
  - OCT + Table
  - Fundus + OCT + Table

  Metadata attributes include:  
  `race`, `male`, `hispanic`, `maritalstatus`, `language`, `dr_class`

  OCT and fundus features:
  - Contrast, Dissimilarity, Homogeneity, Energy, Correlation, ASM
  - Mean, Variance, Standard Deviation, RMS

  **Best-performing method**: Feature Selection, which yields highest accuracy across all fusion approaches.

## 🔁 Workflow Summary

1. Preprocess image and metadata
2. Select fusion method
3. Extract top features
4. Generate FRB rules using FCM
5. Train FKGS model with different hyperparameters (`ran`, `e`):
   - 15, 0.2
   - 15, 0.3
   - 20, 0.2
   - 20, 0.3
6. Evaluate model on test data

## ⚠️ Notes

- This project uses C++ modules (FISA) for high-performance fuzzy calculations.
- **Currently only supports Windows OS**. Linux systems are not yet supported for C++ execution modules.