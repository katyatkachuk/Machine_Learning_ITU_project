import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor

print("Loading and preprocessing data...")

df_train = pd.read_csv("data/claims_train.csv")
df_test = pd.read_csv("data/claims_test.csv")

for df in (df_train, df_test):
    df.drop_duplicates(subset=[c for c in df.columns if c != "IDpol"], inplace=True)
    df["ClaimFlag"] = (df["ClaimNb"] > 0).astype(int)

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

num_features = [
    "Exposure", "VehPower", "VehAge", "DrivAge",
    "BonusMalus", "Density", "BadScore_BM_Dens", "BM_Risk"
]
cat_cols = ["Area", "VehBrand", "VehGas", "Region"]

def build_cat(df, ref=None):
    d = pd.get_dummies(df[cat_cols].fillna("Unknown"), drop_first=True)
    if ref is not None:
        d = d.reindex(columns=ref, fill_value=0)
    return d

df_cat_train = build_cat(df_train)
cat_all = df_cat_train.columns
df_cat_test = build_cat(df_test, ref=cat_all)

scaler = StandardScaler()
X_num_train = pd.DataFrame(
    scaler.fit_transform(df_train[num_features]),
    columns=num_features,
    index=df_train.index,
)
X_num_test = pd.DataFrame(
    scaler.transform(df_test[num_features]),
    columns=num_features,
    index=df_test.index,
)

X_train = pd.concat([X_num_train, df_cat_train], axis=1).astype(float)
X_test = pd.concat([X_num_test, df_cat_test], axis=1).astype(float)

y_train = df_train["Severity"].astype(float)
y_test = df_test["Severity"].astype(float)

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

print("\nTraining XGBoost Regressor (M3)...")
reg = XGBRegressor(
    n_estimators=600,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    tree_method="hist", 
    n_jobs=-1
)

reg.fit(X_tr, y_tr)

y_test_pred = reg.predict(X_test)

print("\n--- Basic Regression Metrics ---")
print(f"Test R2 Score: {r2_score(y_test, y_test_pred):.4f}")
print(f"Test MSE: {mean_squared_error(y_test, y_test_pred):.6f}")

print("\n=== Ranking Quality & Lift Analysis ===")

df_res = pd.DataFrame({
    "Actual_Claims": df_test["ClaimNb"].values,
    "Predicted_Score": y_test_pred
})

df_res = df_res.sort_values("Predicted_Score", ascending=False).reset_index(drop=True)
df_res["Rank"] = df_res.index + 1
df_res["Percentile"] = df_res["Rank"] / len(df_res)

avg_pct = df_res[df_res["Actual_Claims"] >= 1]["Percentile"].mean() * 100
avg_pct_sev = df_res[df_res["Actual_Claims"] >= 2]["Percentile"].mean() * 100

print(f"\n[A] Average Rank Position (Lower is Better)")
print(f"  - Any Claim (>0): Top {avg_pct:.1f}%")
print(f"  - Severe Claim (>=2): Top {avg_pct_sev:.1f}%")

print(f"\n[B] Concentration at the Very Top (Granular Precision)")
k_values = [100, 500, 1000, 2000, 5000]
for k in k_values:
    top_k = df_res.iloc[:k]
    n_claims = (top_k["Actual_Claims"] > 0).sum()
    precision = n_claims / k
    lift = precision / (len(df_res[df_res['Actual_Claims']>0])/len(df_res))
    
    print(f"  - Top {k}: {n_claims} Claims (Prec: {precision:.1%}, Lift: {lift:.1f}x)")

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
    recall_sev = n_severe / total_severe
    precision = n_claims / k
    lift = precision / base_rate
    
    print(f"  - Top {int(frac*100)}% ({k} rows):")
    print(f"      Claims Found: {n_claims}/{total_claims} (Recall: {recall:.1%})")
    print(f"      Severe Found: {n_severe}/{total_severe} (Recall: {recall_sev:.1%})")
    print(f"      Precision: {precision:.1%} | Lift: {lift:.1f}x")

top_5_idx = int(len(df_res) * 0.05)
top_5_df = df_res.iloc[:top_5_idx]
corr = top_5_df["Predicted_Score"].corr(top_5_df["Actual_Claims"], method="spearman")

print(f"\n[D] Rank Correlation inside Top 5% Bucket: {corr:.4f}")
print("\nDone.")
