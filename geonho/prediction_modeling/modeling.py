#%%
'''
K_FGI 뒤집은 버전
'''

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
    df['ret_20d'] = df['log_return_t+1'].rolling(20).sum()

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
# 3. 통합 비교 실험 및 피처 중요도 분석
'''
Sub_index만 있는 데이터 vs Sub_index + sentiment 데이터 비교 in Regression, Classification
'''

def run_full_analysis(df, train_end='2024-12-31'):
    train_mask = df['date'] <= train_end
    test_mask = df['date'] > train_end

    sub_only = [f'sub_index{i}_lag1' for i in range(1, 8)] + ['dayofweek', 'month']
    with_sent = sub_only + ['sent_norm_w', 'sent_energy', 'sent_std_inv', 'neg_z_inv',
                            'sent_norm_diff', 'neg_z_diff', 'sent_norm_ma5', 'neg_z_ma5', 'KFGI']

    experiments = [
        ('Reg', 'Sub Only', sub_only),
        ('Reg', 'Sub+Sent', with_sent),
        ('Cls', 'Sub Only', sub_only),
        ('Cls', 'Sub+Sent', with_sent)
    ]

    summary = []
    fig_cum, ax_cum = plt.subplots(figsize=(10, 5))
    fig_imp, axes_imp = plt.subplots(2, 2, figsize=(15, 12))
    axes_imp = axes_imp.flatten()

    for i, (m_type, f_type, f_list) in enumerate(experiments):
        target = 'target_reg' if m_type == 'Reg' else 'target_cls'

        # 모델 학습
        dtrain = lgb.Dataset(df.loc[train_mask, f_list], label=df.loc[train_mask, target],
                             weight=df.loc[train_mask, 'sample_weight'])
        params = {'objective': 'regression' if m_type == 'Reg' else 'binary', 'verbosity': -1, 'learning_rate': 0.02}
        model = lgb.train(params, dtrain, num_boost_round=300)

        # 예측 및 성과
        preds = model.predict(df.loc[test_mask, f_list])
        actual_ret = df.loc[test_mask, 'log_return_t+1']
        signal = (preds < 0) if m_type == 'Reg' else (preds < 0.5)
        strat_ret = signal * actual_ret

        # 지표 산출
        ann_ret = strat_ret.mean() * 252
        ann_vol = (strat_ret.std() * np.sqrt(252)) + 1e-9
        sharpe = ann_ret / ann_vol

        summary.append({'Model': f"{m_type}_{f_type}", 'Sharpe': round(sharpe, 3), 'Return': f"{ann_ret*100:.2f}%"})

        # 수익 곡선 플롯
        ax_cum.plot(df.loc[test_mask, 'date'], np.exp(strat_ret.cumsum()), label=f"{m_type}_{f_type} (S:{round(sharpe,2)})")

        # 피처 중요도 플롯
        importances = pd.DataFrame({'Feature': f_list, 'Importance': model.feature_importance(importance_type='gain')})
        importances = importances.sort_values(by='Importance', ascending=False).head(10)
        sns.barplot(x='Importance', y='Feature', data=importances, ax=axes_imp[i], palette='viridis')
        axes_imp[i].set_title(f"Top 10 Features: {m_type}_{f_type}")

    # 최종 결과 정리
    ax_cum.plot(df.loc[test_mask, 'date'], np.exp(actual_ret.cumsum()), 'k--', label='Market', alpha=0.3)
    ax_cum.set_title("Cumulative Returns Comparison")
    ax_cum.legend(); ax_cum.grid(True)

    plt.tight_layout()
    plt.show()

    return pd.DataFrame(summary)

# 실행 및 결과 출력
results = run_full_analysis(df_fgi)
print("\n===== 벤치마크 결과 요약 =====")
print(results.to_string(index=False))







#%%
'''
예측 모델 심화
'''
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
coef_list = []

for train_idx, test_idx in tscv.split(X):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X[train_idx])
    model = Ridge(alpha=1.0)
    model.fit(X_train_scaled, y[train_idx])
    coef_list.append(model.coef_)

coef_mean = np.mean(coef_list, axis=0)

pd.Series(coef_mean, index=features).sort_values()


#%%
print("\n===== Fold-wise Performance =====")

for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = Ridge(alpha=1.0)
    model.fit(X_train_scaled, y_train)

    preds = model.predict(X_test_scaled)

    rmse_fold = np.sqrt(mean_squared_error(y_test, preds))
    dir_fold = (np.sign(preds) == np.sign(y_test)).mean()
    corr_fold = np.corrcoef(preds, y_test)[0,1]

    start_date = df_fgi.iloc[test_idx]['date'].min()
    end_date = df_fgi.iloc[test_idx]['date'].max()

    print(f"\nFold {i+1} ({start_date.date()} ~ {end_date.date()})")
    print(f"RMSE: {rmse_fold:.6f}")
    print(f"Dir Acc: {dir_fold:.4f}")
    print(f"Corr: {corr_fold:.4f}")

#%%
'''
XGBoost Rolling Window Test (Lag 확장 전 기본 피처 기준)
'''

from xgboost import XGBRegressor

# -----------------------------
# 1️⃣ XGB 피처 정의 (현재 features 그대로 사용)
# -----------------------------
X_xgb = df_fgi[features].values
y_xgb = df_fgi['target_reg'].values

window_size = 750

preds_xgb = []
actual_xgb = []
dates_xgb = []

print("\n===== XGB Rolling Window Performance =====")

for i in range(window_size, len(X_xgb)):

    train_idx = range(i - window_size, i)
    test_idx = i

    X_train = X_xgb[list(train_idx)]
    y_train = y_xgb[list(train_idx)]
    X_test = X_xgb[test_idx:test_idx+1]
    y_test = y_xgb[test_idx]

    model = XGBRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        objective='reg:squarederror'
    )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)[0]

    preds_xgb.append(pred)
    actual_xgb.append(y_test)
    dates_xgb.append(df_fgi.iloc[test_idx]['date'])

preds_xgb = np.array(preds_xgb)
actual_xgb = np.array(actual_xgb)

rmse_xgb = np.sqrt(mean_squared_error(actual_xgb, preds_xgb))
dir_acc_xgb = (np.sign(preds_xgb) == np.sign(actual_xgb)).mean()
corr_xgb = np.corrcoef(preds_xgb, actual_xgb)[0, 1]

print("===== XGB (Rolling Window) =====")
print(f"RMSE: {rmse_xgb:.6f}")
print(f"Directional Accuracy: {dir_acc_xgb:.4f}")
print(f"Correlation: {corr_xgb:.4f}")





#%%
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

tscv = TimeSeriesSplit(n_splits=5)

X_xgb = df_fgi[features].values
y_xgb = df_fgi['target_reg'].values

preds_all = []
actual_all = []

print("\n===== XGB TimeSeriesSplit Performance (Optimized) =====")

for train_idx, test_idx in tscv.split(X_xgb):

    X_train = X_xgb[train_idx]
    y_train = y_xgb[train_idx]
    X_test = X_xgb[test_idx]
    y_test = y_xgb[test_idx]

    model = XGBRegressor(
        n_estimators=150,          # 줄임
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        objective='reg:squarederror',

        tree_method="hist",        # 🔥 핵심 (속도 5~10배 개선)
        n_jobs=-1                  # 🔥 모든 코어 사용
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