# Course project for *Machine Learning* at the IT University of Copenhagen(2025/2026).  

The project focuses on predicting automobile insurance claim probability using real-world insurance data.  
It includes data cleaning, exploratory data analysis (PCA and clustering), and model development.

---

## Repository Structure
```bash
.
├── data/
│   ├── claims_train.csv
│   ├── claims_test.csv
│   ├── claims_train_clean.csv
│   └── claims_test_clean.csv
│
├── figures/
│   ├── claims_risk_vs_numerical_features.png
│   ├── correlation_matrix_of_features_with_claimnb.png
│   ├── dataset_size_feature_deduplication.png
│   ├── distribution_of_claims_risk.png
│   ├── k_means_clusters_visualized_in_pca_space.png
│   ├── pca_of_claims_data_colored_by_log_risk.png
│   └── pca_of_claims_data_colored_by_risk.png
│
├── models/
│   ├── Decision_tree.py
│   ├── Feed_forward_neural_netrworl.py
│   └── xgb_tree.py
│
├── data_cleaning.ipynb          # Data preprocessing and deduplication
├── EDA-PCA-Clustering.ipynb     # Exploratory analysis, PCA, and clustering
├── requirements.txt             # Project dependencies
└── README.md
```

---

## Setup

Make sure you have **Python 3.9+** installed.

```bash
pip install -r requirements.txt
```
## How to Run
### 1. Data Cleaning
Run the preprocessing notebook:
```bash
jupyter notebook data_cleaning.ipynb
```
This will create the cleaned datasets:
- `data/claims_train_clean.csv`
- `data/claims_test_clean.csv`

---

### 2. Exploratory Analysis and PCA
Run the exploratory notebook:
```bash
jupyter notebook EDA-PCA-Clustering.ipynb
```
Generated figures will be saved automatically in the `figures/` directory.

---

### 3. Train and Evaluate Models
Run the model scripts:
```bash
python models/Decision_tree.py
python models/Feed_forward_neural_netrworl.py
python models/xgb_tree.py
```

---

##  Proejct group:
- **Kateryna Tkachuk (ktka@itu.dk)**
- **Gabriel Catalin Ionescu (gaio@itu.dk)**
