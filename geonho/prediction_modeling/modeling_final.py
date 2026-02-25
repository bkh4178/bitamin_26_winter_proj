'''
K_FGI 뒤집은 버전
'''
#%%%
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

    # 20일 변동성
    df['vol_20d'] = df['log_return_t+1'].rolling(20).std()

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


#%%
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
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import numpy as np

tscv = TimeSeriesSplit(n_splits=5)

X_lgb = df_fgi[features].values
y_lgb = df_fgi['target_reg'].values

preds_all = []
actual_all = []

print("\n===== LGBM TimeSeriesSplit Performance =====")

for train_idx, test_idx in tscv.split(X_lgb):

    X_train = X_lgb[train_idx]
    y_train = y_lgb[train_idx]
    X_test = X_lgb[test_idx]
    y_test = y_lgb[test_idx]

    model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=3,          # 과적합 방지
        num_leaves=15,        # depth=3이면 2^4-1 수준
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    preds_all.extend(preds)
    actual_all.extend(y_test)

preds_all = np.array(preds_all)
actual_all = np.array(actual_all)

rmse = np.sqrt(mean_squared_error(actual_all, preds_all))
dir_acc = (np.sign(preds_all) == np.sign(actual_all)).mean()
corr = np.corrcoef(preds_all, actual_all)[0, 1]

print(f"RMSE: {rmse:.6f}")
print(f"Directional Accuracy: {dir_acc:.4f}")
print(f"Correlation: {corr:.4f}")

#%%

from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import numpy as np

tscv = TimeSeriesSplit(n_splits=5)

print("\n===== LGBM Fold-wise Performance =====")

for i, (train_idx, test_idx) in enumerate(tscv.split(X_lgb)):

    X_train = X_lgb[train_idx]
    y_train = y_lgb[train_idx]
    X_test = X_lgb[test_idx]
    y_test = y_lgb[test_idx]

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
        verbosity = -1
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    dir_acc = (np.sign(preds) == np.sign(y_test)).mean()
    corr = np.corrcoef(preds, y_test)[0, 1]

    start_date = df_fgi.iloc[test_idx]['date'].min()
    end_date = df_fgi.iloc[test_idx]['date'].max()

    print(f"\nFold {i+1} ({start_date.date()} ~ {end_date.date()})")
    print(f"RMSE: {rmse:.6f}")
    print(f"Dir Acc: {dir_acc:.4f}")
    print(f"Corr: {corr:.4f}")






#%%

eval_df = pd.DataFrame({
    "pred": preds_all,
    "actual": actual_all
})

# test 구간에 해당하는 regime만 가져와야 함
# 따라서 test_idx를 따로 저장했어야 정확하지만,
# 간단하게는 다시 split 돌려서 모은다.

mask = df_fgi['regime_mom'] == 1

preds_cond = preds_all[mask[test_idx]]
actual_cond = actual_all[mask[test_idx]]

#%%
#%%
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

# 반드시 최신 df 기준
df_fgi = df_fgi.dropna().reset_index(drop=True)

X = df_fgi[features]          # DataFrame 유지
y = df_fgi['target_reg']      # 1일 수익률 기준

tscv = TimeSeriesSplit(n_splits=5)

preds_all = []
actual_all = []
regime_all = []
dates_all = []

print("\n===== LGBM Fold-wise Performance =====")

for i, (train_idx, test_idx) in enumerate(tscv.split(X)):

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

    # Fold별 성능
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    dir_acc = (np.sign(preds) == np.sign(y_test)).mean()
    corr = np.corrcoef(preds, y_test)[0, 1]

    start_date = df_fgi.iloc[test_idx]['date'].min()
    end_date = df_fgi.iloc[test_idx]['date'].max()

    print(f"\nFold {i+1} ({start_date.date()} ~ {end_date.date()})")
    print(f"RMSE: {rmse:.6f}")
    print(f"Dir Acc: {dir_acc:.4f}")
    print(f"Corr: {corr:.4f}")

    # 전체 평가용 저장
    preds_all.extend(preds)
    actual_all.extend(y_test)
    regime_all.extend(df_fgi.iloc[test_idx]['regime_mom'])
    dates_all.extend(df_fgi.iloc[test_idx]['date'])

# 전체 성능
preds_all = np.array(preds_all)
actual_all = np.array(actual_all)

print("\n===== Overall Performance =====")
rmse = np.sqrt(mean_squared_error(actual_all, preds_all))
dir_acc = (np.sign(preds_all) == np.sign(actual_all)).mean()
corr = np.corrcoef(preds_all, actual_all)[0, 1]

print(f"RMSE: {rmse:.6f}")
print(f"Dir Acc: {dir_acc:.4f}")
print(f"Corr: {corr:.4f}")

#%%
#%%
eval_df = pd.DataFrame({
    "pred": preds_all,
    "actual": actual_all,
    "regime": regime_all,
    "date": dates_all
})

def evaluate_subset(df_subset, name):
    rmse = np.sqrt(mean_squared_error(df_subset['actual'], df_subset['pred']))
    dir_acc = (np.sign(df_subset['pred']) == np.sign(df_subset['actual'])).mean()
    corr = np.corrcoef(df_subset['pred'], df_subset['actual'])[0, 1]

    print(f"\n===== {name} =====")
    print(f"Samples: {len(df_subset)}")
    print(f"RMSE: {rmse:.6f}")
    print(f"Dir Acc: {dir_acc:.4f}")
    print(f"Corr: {corr:.4f}")

# regime=1
evaluate_subset(eval_df[eval_df['regime'] == 1], "Regime = 1")

# regime=0
evaluate_subset(eval_df[eval_df['regime'] == 0], "Regime = 0")



#%%
