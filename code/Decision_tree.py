import pandas as pd
import numpy as np

df_train = pd.read_csv("data/claims_train.csv")
df_train = df_train.drop_duplicates(subset=[col for col in df_train.columns if col != "IDpol"])

df_test = pd.read_csv("data/claims_test.csv")
df_test = df_test.drop_duplicates(subset=[col for col in df_test.columns if col != "IDpol"])

bm_min, bm_max = 50, 230

for df in (df_train, df_test):
    df["BM_Risk"] = (df["BonusMalus"] - bm_min) / (bm_max - bm_min)
    df["LogDensity"] = np.log1p(df["Density"])
    df["BadScore_BM_Dens"] = df["BM_Risk"] * df["LogDensity"]

def minmax(series):
    v = series.values.astype(float)
    lo, hi = np.percentile(v, 1), np.percentile(v, 99)
    v = np.clip(v, lo, hi)
    return (v - v.min()) / (v.max() - v.min() + 1e-9)

for df in (df_train, df_test):
    df["Norm_ClaimNb"]   = minmax(df["ClaimNb"])
    df["Norm_Exposure"]  = minmax(df["Exposure"])
    df["Norm_BonusMalus"] = minmax(df["BonusMalus"])
    
    sev = ((1 * df["Norm_ClaimNb"]) + (0.5 * df["Norm_Exposure"]))
    
    sev = (sev - sev.min()) / (sev.max() - sev.min() + 1e-9)
    df["Severity"] = sev

features = [
    "Exposure", "VehPower", "VehAge", "DrivAge",
    "BonusMalus", "Density", "BadScore_BM_Dens", "BM_Risk"
]
cat_cols = ["Area", "VehBrand", "VehGas", "Region"]
target = "ClaimNb"

def one_hot_encode(df, cat_cols):
    """One-hot encode categorical columns."""
    return pd.get_dummies(df, columns=cat_cols, drop_first=True)

class SimpleDecisionTreeRegressor:
    def __init__(self, max_depth=3, min_samples_leaf=20):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.tree = None
    
    def fit(self, X, y, sample_weight=None):
        def build_tree(X, y, w, depth):
            if depth == self.max_depth or len(y) < self.min_samples_leaf:
                val = np.average(y, weights=w)
                return {'val': val}
            best_feat, best_thresh, best_score = None, None, np.inf
            for i in range(X.shape[1]):
                vals = np.unique(X[:, i])
                for thresh in vals:
                    left = X[:, i] <= thresh
                    right = X[:, i] > thresh
                    if np.sum(left) < self.min_samples_leaf or np.sum(right) < self.min_samples_leaf:
                        continue
                    left_loss = np.average((y[left] - np.average(y[left], weights=w[left])) ** 2, weights=w[left])
                    right_loss = np.average((y[right] - np.average(y[right], weights=w[right])) ** 2, weights=w[right])
                    score = left_loss * np.sum(w[left]) + right_loss * np.sum(w[right])
                    if score < best_score:
                        best_feat, best_thresh, best_score = i, thresh, score
            if best_feat is None:
                return {'val': np.average(y, weights=w)}
            left_mask = X[:, best_feat] <= best_thresh
            right_mask = X[:, best_feat] > best_thresh
            return {
                'feat': best_feat,
                'thresh': best_thresh,
                'left': build_tree(X[left_mask], y[left_mask], w[left_mask], depth + 1),
                'right': build_tree(X[right_mask], y[right_mask], w[right_mask], depth + 1)
            }
        self.tree = build_tree(X, y, sample_weight, 0)

    def predict_one(self, x, node):
        if 'val' in node:
            return node['val']
        if x[node['feat']] <= node['thresh']:
            return self.predict_one(x, node['left'])
        else:
            return self.predict_one(x, node['right'])

    def predict(self, X):
        return np.array([self.predict_one(row, self.tree) for row in X])

oversample_factor = 10  
weight_nonzero = 1.2     
undersample_factor = 0.7 

df_zeros = df_train[df_train[target] == 0]
df_nonzero = df_train[df_train[target] > 0]

if undersample_factor < 1.0:
    n_keep = int(len(df_zeros) * undersample_factor)
    df_zeros = df_zeros.sample(n=n_keep, random_state=42)

df_nonzero_oversampled = pd.concat([df_nonzero] * oversample_factor, ignore_index=True)

df_balanced = pd.concat([df_zeros, df_nonzero_oversampled], ignore_index=True)

X = df_balanced[features + cat_cols]
y = df_balanced[target]
X_enc = one_hot_encode(X, cat_cols)

train_cols = X_enc.columns

sample_weights = np.where(y > 0, weight_nonzero, 1)
X_enc_np = X_enc.to_numpy()

idx = np.arange(len(X_enc_np))
np.random.shuffle(idx)
split = int(0.8 * len(idx))
train_idx, val_idx = idx[:split], idx[split:]

X_train, y_train = X_enc_np[train_idx], y.iloc[train_idx].to_numpy()
w_train = sample_weights[train_idx]
X_val, y_val = X_enc_np[val_idx], y.iloc[val_idx].to_numpy()

tree = SimpleDecisionTreeRegressor(max_depth=3, min_samples_leaf=20)
tree.fit(X_train, y_train, sample_weight=w_train)

y_pred = tree.predict(X_val)
mse = np.mean((y_val - y_pred) ** 2)
mae = np.mean(np.abs(y_val - y_pred))
r2 = 1 - np.sum((y_val - y_pred) ** 2) / np.sum((y_val - np.mean(y_val)) ** 2)
y_val_bin = (y_val >= 1).astype(int)
y_pred_bin = (y_pred >= 0.5).astype(int)
tp = np.sum((y_val_bin == 1) & (y_pred_bin == 1))
fp = np.sum((y_val_bin == 0) & (y_pred_bin == 1))
fn = np.sum((y_val_bin == 1) & (y_pred_bin == 0))
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"Global validation -- Samples: {len(X_val)}, R2: {r2:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

tree_full = SimpleDecisionTreeRegressor(max_depth=3, min_samples_leaf=20)
tree_full.fit(X_enc_np, y.to_numpy(), sample_weight=sample_weights)

X_test = df_test[features + cat_cols]
X_test_enc = one_hot_encode(X_test, cat_cols)

for col in train_cols:
    if col not in X_test_enc.columns:
        X_test_enc[col] = 0
X_test_enc = X_test_enc[train_cols]
X_test_np = X_test_enc.to_numpy()

y_test_pred = tree_full.predict(X_test_np)

if "ClaimNb" in df_test.columns:
    y_true = df_test["ClaimNb"].to_numpy()
    mse_test = np.mean((y_true - y_test_pred) ** 2)
    mae_test = np.mean(np.abs(y_true - y_test_pred))
    r2_test = 1 - np.sum((y_true - y_test_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    y_true_bin = (y_true >= 1).astype(int)
    y_pred_bin = (y_test_pred >= 0.5).astype(int)
    tp = np.sum((y_true_bin == 1) & (y_pred_bin == 1))
    fp = np.sum((y_true_bin == 0) & (y_pred_bin == 1))
    fn = np.sum((y_true_bin == 1) & (y_pred_bin == 0))
    precision_test = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_test = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_test = (2 * precision_test * recall_test) / (precision_test + recall_test) if (precision_test + recall_test) > 0 else 0

    print(f"Global test -- Samples: {len(y_true)}, R2: {r2_test:.4f}, F1: {f1_test:.4f}, Precision: {precision_test:.4f}, Recall: {recall_test:.4f}")
else:
    print(f"Test predictions complete. Generated {len(y_test_pred)} predictions.")
