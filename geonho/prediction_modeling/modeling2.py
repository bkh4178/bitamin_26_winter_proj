#%%
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 데이터 로드 및 가이드 기반 피처 엔지니어링 (동일)
input_file = '/Users/user/Desktop/bitamin/26_winter_proj/data/KFG/KFG_final_2.csv'
df = pd.read_csv(input_file)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True).fillna(method='ffill').fillna(method='bfill')

def build_advanced_features(df):
    df = df.copy()
    '''
    추후 예측에 사용할 변수 미리 생성
    '''
    df['neg_z_inv'] = -df['neg_z']
    df['sent_std_inv'] = -df['sent_std']
    df['sent_energy'] = df['sent_strength_w'] * df['sent_norm_w']
    df['sent_norm_diff'] = df['sent_norm_w'].diff()
    df['neg_z_diff'] = df['neg_z'].diff()
    df['sent_norm_ma5'] = df['sent_norm_w'].rolling(5).mean()
    df['neg_z_ma5'] = df['neg_z'].rolling(5).mean()
    # 20일 누적수익률 (모멘텀)
    r = df['log_return_t+1'].shift(1)

    df['ret_20d'] = r.rolling(20).sum()
    df['vol_20d'] = r.rolling(20).std()

    # 모멘텀 더미 (상승장=1, 하락장=0)
    df['regime_mom'] = (df['ret_20d'] > 0).astype(int)

    # 고변동성 더미
    vol_threshold = df['vol_20d'].rolling(252).median()
    df['regime_vol'] = (df['vol_20d'] > vol_threshold).astype(int)

    # -----------------------------
    # Controlled Lag Expansion (과적합 최소화 설계)
    # -----------------------------
    sub_cols = [f'sub_index{i}' for i in range(1, 8)]

    # sub_index는 lag1~2만
    for col in sub_cols:
        for lag in range(1, 3):
            df[f'{col}_lag{lag}'] = df[col].shift(lag)

    # KFGI는 lag1~3
    # (KFGI는 create_kfgi 이후 생성되므로 여기선 원본 sub 기반만 생성)
    # sent_norm_w는 lag1~3
    for lag in range(1, 4):
        df[f'sent_norm_w_lag{lag}'] = df['sent_norm_w'].shift(lag)

    # 5일 누적 수익률 (단기 모멘텀)
    df['ret_5d'] = df['log_return_t+1'].shift(1).rolling(5).sum()

    # 5일 변동성 (단기 변동성)
    df['vol_5d'] = df['log_return_t+1'].shift(1).rolling(5).std()

    df['dayofweek'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['target_reg'] = df['log_return_t+1']
    df['target_cls'] = (df['log_return_t+1'] > 0).astype(int)
    df['sample_weight'] = np.log1p(df['effective_n'])

    return df.dropna().reset_index(drop=True)

# 앞으로 사용하게 될 dataset
df_eng = build_advanced_features(df)

# 2. K-FGI 지수 생성 (Ridge) , 100 - kfgi
def create_kfgi(df, train_end='2024-12-31'):
    train_mask = df['date'] <= train_end
    core_feats = [f'sub_index{i}' for i in range(1, 8)] + \
                 ['sent_norm_w', 'sent_energy', 'sent_std_inv', 'neg_z_inv']
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(df.loc[train_mask, core_feats])
    X_all_scaled = scaler.transform(df[core_feats])
    ridge = RidgeCV(alphas=[0.1, 1.0, 10.0]).fit(X_train_scaled, df.loc[train_mask, 'target_reg'])
    w = ridge.coef_ / np.sum(np.abs(ridge.coef_))
    raw = X_all_scaled @ w
    p1, p99 = np.percentile(raw[train_mask], [1, 99])
    df['K_FGI'] = 100 * (np.clip(raw, p1, p99) - p1) / (p99 - p1)
    return df, core_feats, w

df_fgi, core_feats, w = create_kfgi(df_eng)

df_fgi['KFGI'] = 100 - df_fgi['K_FGI']
df_fgi['KFGI_x_regime'] = df_fgi['KFGI'] * df_fgi['regime_mom']
df_fgi['sent_x_regime'] = df_fgi['sent_norm_w'] * df_fgi['regime_mom']


'''
예측 모델 심화
'''
# 타겟 변경
df_fgi['target_reg_3d'] = df_fgi['log_return_t+1'].shift(-2).rolling(3).sum()
df_fgi['target_reg_5d'] = df_fgi['log_return_t+1'].shift(-4).rolling(5).sum()

df_fgi = df_fgi.dropna().reset_index(drop=True)


from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

 # -----------------------------
 # Controlled Feature Set (약 30개 내외 유지)
 # -----------------------------
features = []

# sub_index lag1~2
for i in range(1, 8):
    for lag in range(1, 3):
        features.append(f'sub_index{i}_lag{lag}')

# sentiment lag
for lag in range(1, 4):
    features.append(f'sent_norm_w_lag{lag}')

# 기존 engineered 변수
features += [
    'sent_energy',
    'sent_std_inv',
    'neg_z_inv',
    'sent_norm_diff',
    'neg_z_diff',
    'sent_norm_ma5',
    'neg_z_ma5',
    'ret_5d',
    'vol_5d',
    'KFGI',
    'regime_mom',
    'regime_vol',
    'KFGI_x_regime',
    'sent_x_regime',
    'dayofweek',
    'month'
]

X = df_fgi[features].values
y = df_fgi['target_reg'].values

tscv = TimeSeriesSplit(n_splits=5)
window_size = 750

preds_all = []
actual_all = []
dates_all = []

# -----------------------------
# 2️⃣ Walk-forward CV
# -----------------------------
print("\n===== Rolling Window Performance =====")
for i in range(window_size, len(X)):
    train_idx = range(i - window_size, i)
    test_idx = i    

    X_train = X[list(train_idx)]
    y_train = y[list(train_idx)]
    X_test = X[test_idx:test_idx+1]
    y_test = y[test_idx]

    # 🔴 scaler는 반드시 fold 안에서 fit
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = Ridge(alpha=1.0)
    model.fit(X_train_scaled, y_train)
    pred = model.predict(X_test_scaled)[0]

    preds_all.append(pred)
    actual_all.append(y_test)
    dates_all.append(df_fgi.iloc[test_idx]['date'])

preds_all = np.array(preds_all)
actual_all = np.array(actual_all)

# -----------------------------
# 3️⃣ 전체 성능 평가
# -----------------------------
rmse = np.sqrt(mean_squared_error(actual_all, preds_all))
dir_acc = (np.sign(preds_all) == np.sign(actual_all)).mean()
corr = np.corrcoef(preds_all, actual_all)[0, 1]

print("===== Ridge Baseline (Scaled, Rolling Window) =====")
print(f"RMSE: {rmse:.6f}")
print(f"Directional Accuracy: {dir_acc:.4f}")
print(f"Correlation: {corr:.4f}")


#%%
#%%
eval_df = pd.DataFrame({
    "pred": preds_all,
    "actual": actual_all,
    "date": dates_all
})

eval_df = eval_df.sort_values("date").reset_index(drop=True)

# 모델 전략 (long-only, pred>0)
eval_df["signal"] = (eval_df["pred"] > 0).astype(int)
eval_df["strategy_ret"] = eval_df["signal"] * eval_df["actual"]

# Buy & Hold
eval_df["bh_ret"] = eval_df["actual"]

#%%
print(eval_df['date'].min())
print(eval_df['date'].max())
#%%
def performance_report(df, name):

    ann_ret = df["strategy_ret"].mean() * 252
    ann_vol = df["strategy_ret"].std() * np.sqrt(252)
    sharpe = ann_ret / (ann_vol + 1e-9)

    bh_ann_ret = df["bh_ret"].mean() * 252
    bh_ann_vol = df["bh_ret"].std() * np.sqrt(252)
    bh_sharpe = bh_ann_ret / (bh_ann_vol + 1e-9)

    dir_acc = (np.sign(df["pred"]) == np.sign(df["actual"])).mean()
    corr = np.corrcoef(df["pred"], df["actual"])[0, 1]

    print(f"\n===== {name} =====")
    print(f"Samples: {len(df)}")
    print(f"Model Sharpe: {sharpe:.3f}")
    print(f"Buy&Hold Sharpe: {bh_sharpe:.3f}")
    print(f"Directional Acc: {dir_acc:.3f}")
    print(f"Correlation: {corr:.3f}")

#%%
# 2022~2024
df_early = eval_df[eval_df["date"] < "2025-01-01"]

# 2025
df_2025 = eval_df[eval_df["date"] >= "2025-01-01"]

performance_report(df_early, "2022~2024")
performance_report(df_2025, "2025 Only")


#%%
def run_lgbm_tscv(df, features, target):

    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_squared_error
    import lightgbm as lgb
    import numpy as np

    X = df[features]
    y = df[target]

    tscv = TimeSeriesSplit(n_splits=5)

    preds_all = []
    actual_all = []
    dates_all = []

    for train_idx, test_idx in tscv.split(X):

        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        model = lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=3,
            num_leaves=15,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        preds_all.extend(preds)
        actual_all.extend(y_test)
        dates_all.extend(df.iloc[test_idx]['date'])

    preds_all = np.array(preds_all)
    actual_all = np.array(actual_all)

    rmse = np.sqrt(mean_squared_error(actual_all, preds_all))
    dir_acc = (np.sign(preds_all) == np.sign(actual_all)).mean()
    corr = np.corrcoef(preds_all, actual_all)[0,1]

    print("\n===== LGBM TSCV Result =====")
    print(f"Target: {target}")
    print(f"RMSE: {rmse:.6f}")
    print(f"Dir Acc: {dir_acc:.4f}")
    print(f"Corr: {corr:.4f}")

    return preds_all, actual_all, dates_all

#%%
TARGET = 'target_reg'
preds, actual, dates = run_lgbm_tscv(df_fgi, features, TARGET)






#%%
#%%
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

# 반드시 최신 df 기준
df_fgi = df_fgi.dropna().reset_index(drop=True)

X = df_fgi[features]        # DataFrame 유지
y = df_fgi['target_reg_5d']    # 1일 기준 (3d/5d면 여기만 바꾸면 됨)

tscv = TimeSeriesSplit(n_splits=5)

preds_all = []
actual_all = []
dates_all = []

print("\n===== LGBM TSCV Performance =====")

for train_idx, test_idx in tscv.split(X):

    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]

    model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=3,
        num_leaves=15,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    preds_all.extend(preds)
    actual_all.extend(y_test)
    dates_all.extend(df_fgi.iloc[test_idx]['date'])

preds_all = np.array(preds_all)
actual_all = np.array(actual_all)

rmse = np.sqrt(mean_squared_error(actual_all, preds_all))
dir_acc = (np.sign(preds_all) == np.sign(actual_all)).mean()
corr = np.corrcoef(preds_all, actual_all)[0,1]

print(f"RMSE: {rmse:.6f}")
print(f"Directional Accuracy: {dir_acc:.4f}")
print(f"Correlation: {corr:.4f}")


#%%
eval_df = pd.DataFrame({
    "date": dates_all,
    "pred": preds_all,
    "actual": actual_all
}).sort_values("date").reset_index(drop=True)

# long-only 전략
eval_df["signal"] = (eval_df["pred"] > 0).astype(int)
eval_df["strategy_ret"] = eval_df["signal"] * eval_df["actual"]

# buy & hold
eval_df["bh_ret"] = eval_df["actual"]

#%%
def performance_report(df, name):

    if len(df) == 0:
        print(f"\n===== {name} =====")
        print("No samples in this period.")
        return

    ann_ret = df["strategy_ret"].mean() * 252
    ann_vol = df["strategy_ret"].std() * np.sqrt(252)
    sharpe = ann_ret / (ann_vol + 1e-9)

    bh_ann_ret = df["bh_ret"].mean() * 252
    bh_ann_vol = df["bh_ret"].std() * np.sqrt(252)
    bh_sharpe = bh_ann_ret / (bh_ann_vol + 1e-9)

    dir_acc = (np.sign(df["pred"]) == np.sign(df["actual"])).mean()
    corr = np.corrcoef(df["pred"], df["actual"])[0, 1]

    print(f"\n===== {name} =====")
    print(f"Samples: {len(df)}")
    print(f"Model Sharpe: {sharpe:.3f}")
    print(f"Buy&Hold Sharpe: {bh_sharpe:.3f}")
    print(f"Directional Acc: {dir_acc:.3f}")
    print(f"Correlation: {corr:.3f}")


#%%
df_early = eval_df[eval_df["date"] < "2025-01-01"]
df_2025 = eval_df[eval_df["date"] >= "2025-01-01"]

performance_report(df_early, "2022~2024")
performance_report(df_2025, "2025 Only")

#%%
eval_df['signal'].mean()

#%%
#%%
eval_df = pd.DataFrame({
    "date": dates_all,
    "pred": preds_all,
    "actual": actual_all
}).sort_values("date").reset_index(drop=True)

# 상위 20% 임계값
threshold = eval_df["pred"].quantile(0.8)

eval_df["signal_top20"] = (eval_df["pred"] >= threshold).astype(int)
eval_df["strategy_top20"] = eval_df["signal_top20"] * eval_df["actual"]

print("Signal Frequency:", eval_df["signal_top20"].mean())

#%%
#%%
def performance_detail(df, ret_col, name):

    if len(df) == 0:
        print(f"\n{name} - No samples")
        return

    ret = df[ret_col]

    ann_ret = ret.mean() * 252
    ann_vol = ret.std() * np.sqrt(252)
    sharpe = ann_ret / (ann_vol + 1e-9)

    # 누적수익
    cum = np.exp(ret.cumsum())

    # Max Drawdown
    rolling_max = cum.cummax()
    drawdown = cum / rolling_max - 1
    mdd = drawdown.min()

    print(f"\n===== {name} =====")
    print(f"Samples: {len(df)}")
    print(f"Annual Return: {ann_ret:.3f}")
    print(f"Sharpe: {sharpe:.3f}")
    print(f"Max Drawdown: {mdd:.3f}")

    return cum

cum_top20 = performance_detail(eval_df, "strategy_top20", "Top 20% Strategy")

#%%
#%%
df_early = eval_df[eval_df["date"] < "2025-01-01"]
df_2025 = eval_df[eval_df["date"] >= "2025-01-01"]

performance_detail(df_early, "strategy_top20", "Top20 2022~2024")
performance_detail(df_2025, "strategy_top20", "Top20 2025")

#%%
eval_df.groupby(pd.qcut(eval_df["pred"], 5))["actual"].mean()