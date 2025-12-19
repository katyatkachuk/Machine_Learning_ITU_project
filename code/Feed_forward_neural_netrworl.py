import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

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
    return pd.get_dummies(df, columns=cat_cols, drop_first=True)

def preprocess(df, scaler=None, fit_scaler=False):
    df_cat = pd.get_dummies(df[cat_cols], drop_first=True)
    df_num = df[features]
    if scaler is None and fit_scaler:
        scaler = StandardScaler()
        df_num_scaled = pd.DataFrame(scaler.fit_transform(df_num), columns=features)
    elif scaler is not None:
        df_num_scaled = pd.DataFrame(scaler.transform(df_num), columns=features)
    else:
        df_num_scaled = df_num
    df_proc = pd.concat([df_num_scaled, df_cat], axis=1)
    return df_proc, scaler

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

    def train(self, X, y, epochs=50, batch_size=64, sample_weight=None, verbose=True):
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

input_dim = X_train.shape[1]
hidden_layers = [32, 16]

nn = SimpleNeuralNetwork(input_dim=input_dim, hidden_dims=hidden_layers, learning_rate=0.001)
nn.train(X_train, y_train, epochs=100, batch_size=64, sample_weight=w_train)

y_val_pred = nn.predict(X_val).flatten()
mse = np.mean((y_val.flatten() - y_val_pred) ** 2)
mae = np.mean(np.abs(y_val.flatten() - y_val_pred))
r2 = 1 - np.sum((y_val.flatten() - y_val_pred) ** 2) / np.sum((y_val.flatten() - np.mean(y_val.flatten())) ** 2)
y_val_bin = (y_val.flatten() >= 1).astype(int)
y_pred_bin = (y_val_pred >= 0.5).astype(int)
tp = np.sum((y_val_bin == 1) & (y_pred_bin == 1))
fp = np.sum((y_val_bin == 0) & (y_pred_bin == 1))
fn = np.sum((y_val_bin == 1) & (y_pred_bin == 0))
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"Validation metrics: MSE={mse:.5f}, MAE={mae:.5f}, R2={r2:.5f}")
print(f"Classification metrics: Precision={precision:.5f}, Recall={recall:.5f}, F1={f1:.5f}")
