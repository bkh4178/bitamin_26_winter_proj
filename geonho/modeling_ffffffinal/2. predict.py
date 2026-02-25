#%%
# ============================================
# Next-Day Return Prediction Module
# Input  : KFG_with_KFGI.csv
# Output : prediction_output.csv
# ============================================

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error


# -------------------------------------------------
# 1️⃣ 데이터 로드
# -------------------------------------------------
def load_data(path):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


# -------------------------------------------------
# 2️⃣ Feature 선택 (고정 버전)
# -------------------------------------------------
def select_features(df):

    features = []

    # sub_index lag
    for i in range(1, 8):
        features.append(f"sub_index{i}_lag1")
        features.append(f"sub_index{i}_lag2")

    # sentiment 관련
    features += [
        "sent_norm_w",
        "sent_energy",
        "sent_std_inv",
        "neg_z_inv",
        "sent_norm_ma5",
        "neg_z_ma5",
    ]

    # KFGI
    features += ["KFGI"]

    # calendar
    features += ["dayofweek", "month"]

    return features


# -------------------------------------------------
# 3️⃣ TimeSeries CV 예측
# -------------------------------------------------
def run_tscv_prediction(df, features):

    df = df.dropna().reset_index(drop=True)

    X = df[features]
    y_reg = df["log_return_t+1"]
    y_cls = (df["log_return_t+1"] > 0).astype(int)

    tscv = TimeSeriesSplit(n_splits=5)

    pred_all = []

    for train_idx, test_idx in tscv.split(X):

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train_reg, y_test_reg = y_reg.iloc[train_idx], y_reg.iloc[test_idx]
        y_train_cls = y_cls.iloc[train_idx]

        # ----- 회귀 모델 -----
        reg = lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=3,
            num_leaves=15,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )
        reg.fit(X_train, y_train_reg)
        pred_ret = reg.predict(X_test)

        # ----- 분류 모델 -----
        clf = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=3,
            num_leaves=15,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )
        clf.fit(X_train, y_train_cls)
        pred_prob = clf.predict_proba(X_test)[:, 1]

        tmp = pd.DataFrame({
            "date": df.loc[test_idx, "date"].values,
            "actual_ret": y_test_reg.values,
            "pred_ret": pred_ret,
            "pred_prob_up": pred_prob,
        })

        pred_all.append(tmp)

    pred_df = pd.concat(pred_all).sort_values("date").reset_index(drop=True)
    return pred_df


# -------------------------------------------------
# 4️⃣ 간단 검증 출력
# -------------------------------------------------
def quick_validation(pred_df):

    rmse = np.sqrt(mean_squared_error(pred_df["actual_ret"], pred_df["pred_ret"]))
    dir_acc = (np.sign(pred_df["actual_ret"]) == np.sign(pred_df["pred_ret"])).mean()
    corr = np.corrcoef(pred_df["actual_ret"], pred_df["pred_ret"])[0, 1]

    print("\n===== Validation Summary =====")
    print(f"RMSE: {rmse:.6f}")
    print(f"Directional Accuracy: {dir_acc:.3f}")
    print(f"Correlation: {corr:.3f}")


# -------------------------------------------------
# 5️⃣ 실행부
# -------------------------------------------------
if __name__ == "__main__":

    input_path = "KFG_with_KFGI.csv"  # 경로 맞게 수정
    df = load_data(input_path)

    features = select_features(df)
    pred_df = run_tscv_prediction(df, features)

    pred_df.to_csv("prediction_output.csv", index=False)

    quick_validation(pred_df)

    print("\nPrediction file saved: prediction_output.csv")





#%%
