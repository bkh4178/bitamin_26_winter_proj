#%%
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.model_selection import TimeSeriesSplit


def load_raw(path):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df.ffill().bfill()
    return df


def build_features(df):
    df = df.copy()

    df["neg_z_inv"] = -df["neg_z"]
    df["sent_std_inv"] = -df["sent_std"]
    df["sent_energy"] = df["sent_strength_w"] * df["sent_norm_w"]

    df["sent_norm_diff"] = df["sent_norm_w"].diff()
    df["neg_z_diff"] = df["neg_z"].diff()
    df["sent_norm_ma5"] = df["sent_norm_w"].rolling(5).mean()
    df["neg_z_ma5"] = df["neg_z"].rolling(5).mean()

    for i in range(1, 8):
        df[f"sub_index{i}_lag1"] = df[f"sub_index{i}"].shift(1)

    df["target_reg"] = df["log_return_t+1"]
    df["target_5d"] = df["log_return_t+1"].shift(-4).rolling(5).sum()

    df["dayofweek"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month

    return df.dropna().reset_index(drop=True)


# -----------------------------
# 너의 KFGI (Ridge 기반 + 뒤집기)
# -----------------------------
def create_kfgi_ridge(df, train_end="2024-12-31"):
    df = df.copy()
    train_mask = df["date"] <= train_end

    core_feats = [f"sub_index{i}" for i in range(1, 8)] + [
        "sent_norm_w", "sent_energy", "sent_std_inv", "neg_z_inv"
    ]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(df.loc[train_mask, core_feats])
    X_all = scaler.transform(df[core_feats])

    ridge = RidgeCV(alphas=[0.1, 1.0, 10.0])
    ridge.fit(X_train, df.loc[train_mask, "target_reg"])

    w = ridge.coef_
    w = w / (np.sum(np.abs(w)) + 1e-12)

    raw = X_all @ w
    p1, p99 = np.percentile(raw[train_mask], [1, 99])
    k_fgi = 100 * (np.clip(raw, p1, p99) - p1) / (p99 - p1 + 1e-12)

    df["K_FGI_ridge"] = k_fgi
    df["KFGI_ridge"] = 100 - df["K_FGI_ridge"]   # ✅ 뒤집기

    return df


# -----------------------------
# 친구 Logical KFGI (+ 뒤집기)
# -----------------------------
def create_kfgi_logical(df, train_end="2024-12-31"):
    df = df.copy()
    train_mask = df["date"] <= train_end

    core_feats = [f"sub_index{i}" for i in range(1, 8)] + [
        "sent_norm_w", "sent_energy", "sent_std_inv", "neg_z_inv"
    ]

    directions = {col: 1 for col in core_feats}
    dir_vector = np.array([directions[col] for col in core_feats])

    scaler = StandardScaler()
    X_train = scaler.fit_transform(df.loc[train_mask, core_feats])
    X_all = scaler.transform(df[core_feats])

    ridge = RidgeCV(alphas=np.logspace(-2, 2, 10))
    ridge.fit(X_train, df.loc[train_mask, "target_reg"])

    logical_w = np.abs(ridge.coef_) * dir_vector
    logical_w = logical_w / (np.sum(np.abs(logical_w)) + 1e-12)

    raw = X_all @ logical_w
    p1, p99 = np.percentile(raw[train_mask], [1, 99])
    k_fgi = 100 * (np.clip(raw, p1, p99) - p1) / (p99 - p1 + 1e-12)

    df["K_FGI_logical"] = k_fgi
    df["KFGI_logical"] = 100 - df["K_FGI_logical"]  # ✅ 뒤집기

    return df


def evaluate(df, kfgi_col, target_col):
    base_feats = [f"sub_index{i}_lag1" for i in range(1, 8)] + ["dayofweek", "month"]
    feats = base_feats + [kfgi_col]

    X = df[feats]
    y = df[target_col]

    tscv = TimeSeriesSplit(n_splits=5)

    preds_all, actual_all = [], []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = Ridge(alpha=1.0)
        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)

        preds_all.extend(preds)
        actual_all.extend(y_test.values)

    preds_all = np.array(preds_all)
    actual_all = np.array(actual_all)

    signal = (preds_all > 0).astype(int)
    strat = signal * actual_all

    ann_ret = strat.mean() * 252
    ann_vol = strat.std() * np.sqrt(252)
    sharpe = ann_ret / (ann_vol + 1e-12)
    corr = np.corrcoef(preds_all, actual_all)[0, 1]

    return {
        "Sharpe": round(float(sharpe), 3),
        "AnnualReturn": round(float(ann_ret), 3),
        "Corr": round(float(corr), 3),
        "SignalFreq": round(float(signal.mean()), 3),
    }


if __name__ == "__main__":

    df = load_raw('/Users/user/Desktop/bitamin/26_winter_proj/data/KFG/KFG_final_2.csv')
    df = build_features(df)

    df = create_kfgi_ridge(df)
    df = create_kfgi_logical(df)

    print("\n===== 지수 상관 (뒤집기 통일 후) =====")
    print("corr(KFGI_ridge, KFGI_logical) =", df["KFGI_ridge"].corr(df["KFGI_logical"]))

    print("\n===== 1D 비교 =====")
    print("Ridge KFGI:", evaluate(df, "KFGI_ridge", "target_reg"))
    print("Logical KFGI:", evaluate(df, "KFGI_logical", "target_reg"))

    print("\n===== 5D 비교 =====")
    print("Ridge KFGI:", evaluate(df, "KFGI_ridge", "target_5d"))
    print("Logical KFGI:", evaluate(df, "KFGI_logical", "target_5d"))

#%%
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def plot_kospi_kfgi(df, kospi_col="close"):

    df_plot = df.copy()

    # 누적 로그수익률 기반 지수 생성 (혹시 종가 없으면 이걸로)
    if kospi_col not in df_plot.columns:
        df_plot["kospi_cum"] = np.exp(df_plot["log_return_t+1"].cumsum())
        kospi_col = "kospi_cum"

    scaler = MinMaxScaler()

    df_plot["kospi_scaled"] = scaler.fit_transform(df_plot[[kospi_col]])
    df_plot["ridge_scaled"] = scaler.fit_transform(df_plot[["KFGI_ridge"]])
    df_plot["logical_scaled"] = scaler.fit_transform(df_plot[["KFGI_logical"]])

    plt.figure(figsize=(14,6))

    plt.plot(df_plot["date"], df_plot["kospi_scaled"], label="KOSPI (scaled)", linewidth=2)
    plt.plot(df_plot["date"], df_plot["ridge_scaled"], label="Ridge KFGI", alpha=0.8)
    plt.plot(df_plot["date"], df_plot["logical_scaled"], label="Logical KFGI", alpha=0.8)

    plt.legend()
    plt.title("KOSPI vs Ridge KFGI vs Logical KFGI")
    plt.xlabel("Date")
    plt.ylabel("Scaled Value")
    plt.grid(True)
    plt.show()

def plot_kfgi_only(df):

    plt.figure(figsize=(14,6))

    plt.plot(df["date"], df["KFGI_ridge"], label="Ridge KFGI", alpha=0.9)
    plt.plot(df["date"], df["KFGI_logical"], label="Logical KFGI", alpha=0.9)

    plt.legend()
    plt.title("Ridge vs Logical KFGI (Raw Scale)")
    plt.xlabel("Date")
    plt.ylabel("KFGI")
    plt.grid(True)
    plt.show()

def scatter_kfgi(df):

    plt.figure(figsize=(6,6))

    plt.scatter(df["KFGI_ridge"], df["KFGI_logical"], alpha=0.3)
    plt.xlabel("Ridge KFGI")
    plt.ylabel("Logical KFGI")
    plt.title("Scatter: Ridge vs Logical KFGI")
    plt.grid(True)
    plt.show()
#%%
plot_kospi_kfgi(df)
#%%
scatter_kfgi(df)

#%%
df[["sub_index5", "log_return_t+1"]].corr().iloc[0,1]