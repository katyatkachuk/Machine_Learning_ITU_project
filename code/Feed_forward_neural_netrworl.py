import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from tqdm import tqdm

df_train = pd.read_csv("data/claims_train_clean.csv")
df_train = df_train.drop_duplicates(subset=[col for col in df_train.columns if col != "IDpol"])

df_test = pd.read_csv("data/claims_test_clean.csv")
df_test = df_test.drop_duplicates(subset=[col for col in df_test.columns if col != "IDpol"])

bm_min = df_train['BonusMalus'].min()
bm_max = df_train['BonusMalus'].max()

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
    return pd.get_dummies(df, columns=cat_cols, drop_first=True)

def preprocess(df, scaler=None, fit_scaler=False):
    df_cat = pd.get_dummies(df[cat_cols], drop_first=True)
    df_num = df[features]
    if scaler is None and fit_scaler:
        scaler = StandardScaler()
        df_num_scaled = pd.DataFrame(scaler.fit_transform(df_num), columns=features, index=df_num.index)
    elif scaler is not None:
        df_num_scaled = pd.DataFrame(scaler.transform(df_num), columns=features, index=df_num.index)
    else:
        df_num_scaled = df_num
    df_proc = pd.concat([df_num_scaled, df_cat], axis=1)
    return df_proc, scaler

# Balancing
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

print(f"Balanced dataset: {len(df_balanced)} samples")

X_raw = df_balanced[features + cat_cols]
y = df_balanced[target].to_numpy().reshape(-1, 1)

X_proc, scaler = preprocess(X_raw, fit_scaler=True)
X_np = X_proc.to_numpy()

sample_weights = np.where(y.flatten() > 0, weight_nonzero, 1.0)

n_samples = X_np.shape[0]
idx = np.arange(n_samples)
np.random.shuffle(idx)
split_idx = int(0.8 * n_samples)
train_idx, val_idx = idx[:split_idx], idx[split_idx:]

X_train = X_np[train_idx]
y_train = y[train_idx]
w_train = sample_weights[train_idx]

X_val = X_np[val_idx]
y_val = y[val_idx]

class SimpleNeuralNetwork:
    def __init__(self, input_dim, hidden_dims, output_dim=1, learning_rate=0.001):
        self.learning_rate = learning_rate
        layer_dims = [input_dim] + hidden_dims + [output_dim]
        self.weights = []
        self.biases = []
        for i in range(len(layer_dims) - 1):
            W = np.array(np.random.randn(layer_dims[i], layer_dims[i+1]) * np.sqrt(2. / layer_dims[i]), dtype=np.float64)
            b = np.zeros((1, layer_dims[i+1]), dtype=np.float64)
            self.weights.append(W)
            self.biases.append(b)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(np.float64)

    def forward(self, X):
        self.z = []
        self.a = [X]
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            z = self.a[-1].dot(W) + b
            a = self.relu(z)
            self.z.append(z)
            self.a.append(a)
        z = self.a[-1].dot(self.weights[-1]) + self.biases[-1]
        self.z.append(z)
        self.a.append(z)  
        return z

    def compute_loss(self, y_pred, y_true, sample_weight=None):
        if sample_weight is not None:
            diff = y_pred - y_true
            weighted_diff = diff * np.sqrt(sample_weight.reshape(-1, 1))
            return np.mean(weighted_diff ** 2)
        else:
            return np.mean((y_pred - y_true) ** 2)

    def backward(self, y_pred, y_true, sample_weight=None):
        m = y_true.shape[0]
        if sample_weight is not None:
            dz = 2 * (y_pred - y_true) * sample_weight.reshape(-1, 1) / m
        else:
            dz = 2 * (y_pred - y_true) / m

        grads_w = []
        grads_b = []

        dw = self.a[-2].T.dot(dz).astype(np.float64)
        db = np.sum(dz, axis=0, keepdims=True).astype(np.float64)
        grads_w.insert(0, dw)
        grads_b.insert(0, db)

        da = dz.dot(self.weights[-1].T)

        for i in reversed(range(len(self.weights) - 1)):
            dz = da * self.relu_derivative(self.z[i])
            dw = self.a[i].T.dot(dz).astype(np.float64)
            db = np.sum(dz, axis=0, keepdims=True).astype(np.float64)
            grads_w.insert(0, dw)
            grads_b.insert(0, db)
            if i > 0:
                da = dz.dot(self.weights[i].T)

        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] - self.learning_rate * grads_w[i]
            self.biases[i] = self.biases[i] - self.learning_rate * grads_b[i]

    def train(self, X, y, epochs=100, batch_size=64, sample_weight=None, verbose=True):
        n_samples = X.shape[0]
        for epoch in range(epochs):
            perm = np.random.permutation(n_samples)
            X_shuf = X[perm]
            y_shuf = y[perm]
            if sample_weight is not None:
                w_shuf = sample_weight[perm]
            else:
                w_shuf = None

            loss_epoch = 0
            for i in tqdm(range(0, n_samples, batch_size), desc=f"Epoch {epoch+1}/{epochs}", leave=False):
                X_batch = X_shuf[i:i+batch_size]
                y_batch = y_shuf[i:i+batch_size]
                w_batch = w_shuf[i:i+batch_size] if w_shuf is not None else None
                y_pred = self.forward(X_batch)
                loss = self.compute_loss(y_pred, y_batch, sample_weight=w_batch)
                loss_epoch += loss * X_batch.shape[0]
                self.backward(y_pred, y_batch, sample_weight=w_batch)
            loss_epoch /= n_samples
            if verbose:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss_epoch:.5f}")

    def predict(self, X):
        return self.forward(X)

# Train NN
input_dim = X_train.shape[1]
hidden_layers = [32, 16]

print("Training Neural Network...")
nn = SimpleNeuralNetwork(input_dim=input_dim, hidden_dims=hidden_layers, learning_rate=0.001)
nn.train(X_train, y_train, epochs=100, batch_size=64, sample_weight=w_train)

# VALIDATION metrics
print("\n--- Validation Metrics ---")
y_val_pred = nn.predict(X_val).flatten()
mse_val = np.mean((y_val.flatten() - y_val_pred) ** 2)
mae_val = np.mean(np.abs(y_val.flatten() - y_val_pred))
r2_val = 1 - np.sum((y_val.flatten() - y_val_pred) ** 2) / np.sum((y_val.flatten() - np.mean(y_val.flatten())) ** 2)

print(f"Validation MSE={mse_val:.5f}, MAE={mae_val:.5f}, R2={r2_val:.5f}")

# TEST evaluation (FIXED!)
print("\n--- Test Evaluation ---")
X_test_raw = df_test[features + cat_cols].copy()
X_test_proc, _ = preprocess(X_test_raw, scaler=scaler)

# Align test columns with train columns (CRITICAL FIX)
train_cols = X_proc.columns
for col in train_cols:
    if col not in X_test_proc.columns:
        X_test_proc[col] = 0

# Reorder columns to match train exactly, preserve all rows
X_test_proc = X_test_proc.reindex(columns=train_cols, fill_value=0)
X_test_np = X_test_proc.to_numpy()

print(f"Test shape: {X_test_np.shape}, df_test shape: {df_test.shape}")

y_test_pred = nn.predict(X_test_np).flatten()
y_test_true = df_test["ClaimNb"].values

test_mse = np.mean((y_test_true - y_test_pred) ** 2)
test_r2 = r2_score(y_test_true, y_test_pred)
test_mae = np.mean(np.abs(y_test_true - y_test_pred))

print(f"Test MSE: {test_mse:.6f}")
print(f"Test R2 Score: {test_r2:.4f}")
print(f"Test MAE: {test_mae:.5f}")

# FULL XGBoost-style ranking metrics
print("\n=== Ranking Quality & Lift Analysis ===")
df_res = pd.DataFrame({
    "Actual_Claims": y_test_true,
    "Predicted_Score": y_test_pred
})
df_res = df_res.sort_values("Predicted_Score", ascending=False).reset_index(drop=True)
df_res["Rank"] = df_res.index + 1
df_res["Percentile"] = df_res["Rank"] / len(df_res)

avg_pct = df_res[df_res["Actual_Claims"] >= 1]["Percentile"].mean() * 100
avg_pct_sev = df_res[df_res["Actual_Claims"] >= 2]["Percentile"].mean() * 100

print(f"\n[A] Average Rank Position (Lower is Better)")
print(f" - Any Claim (>0): Top {avg_pct:.1f}%")
print(f" - Severe Claim (>=2): Top {avg_pct_sev:.1f}%")

k_values = [100, 500, 1000, 2000, 5000]
for k in k_values:
    top_k = df_res.iloc[:k]
    n_claims = (top_k["Actual_Claims"] > 0).sum()
    precision = n_claims / k
    lift = precision / (len(df_res[df_res['Actual_Claims']>0])/len(df_res))
    print(f" - Top {k}: {n_claims} Claims (Prec: {precision:.1%}, Lift: {lift:.1f}x)")

print(f"\n[C] Bucket Analysis (Recall & Lift)")
top_fracs = [0.05, 0.10, 0.20, 0.50]
total_claims = (df_res["Actual_Claims"] > 0).sum()
total_severe = (df_res["Actual_Claims"] >= 2).sum()
base_rate = total_claims / len(df_res)

for frac in top_fracs:
    k = int(len(df_res) * frac)
    top_k = df_res.iloc[:k]
    
    n_claims = (top_k["Actual_Claims"] > 0).sum()
    n_severe = (top_k["Actual_Claims"] >= 2).sum()
    
    recall = n_claims / total_claims
    recall_sev = n_severe / total_severe if total_severe > 0 else 0
    precision = n_claims / k
    lift = precision / base_rate
    
    print(f" - Top {int(frac*100)}% ({k} rows):")
    print(f"    Claims Found: {n_claims}/{total_claims} (Recall: {recall:.1%})")
    print(f"    Severe Found: {n_severe}/{total_severe} (Recall: {recall_sev:.1%})")
    print(f"    Precision: {precision:.1%} | Lift: {lift:.1f}x")

top_5_idx = int(len(df_res) * 0.05)
top_5_df = df_res.iloc[:top_5_idx]
corr = top_5_df["Predicted_Score"].corr(top_5_df["Actual_Claims"], method="spearman")

print(f"\n[D] Rank Correlation inside Top 5% Bucket: {corr:.4f}")
print("\nDone.")
