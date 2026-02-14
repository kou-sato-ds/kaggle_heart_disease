import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# ==========================================
# 1. DATA LOADING
# ==========================================
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# --- 今後のTODO: ここでオリジナルデータをマージ予定 ---
# original = pd.read_csv('data/heart_uci.csv')
# train = pd.concat([train, original], axis=0).reset_index(drop=True)

# ==========================================
# 2. FEATURES
# ==========================================
def create_features(df):
    df = df.copy()
    # 独自の特徴量: 血圧と最大心拍数の掛け合わせ
    df['BP_HR_Risk'] = df['Systolic BP'] * df['Max HR']
    
    # カテゴリ型への変換
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if col != 'Heart Disease':
            df[col] = df[col].astype('category')
    return df

train = create_features(train)
test = create_features(test)

X = train.drop(['id', 'Heart Disease'], axis=1)
y = train['Heart Disease'].map({'Presence': 1, 'Absence': 0})
X_test = test.drop(['id'], axis=1)

# ==========================================
# 3. K-FOLD & ENSEMBLE TRAINING
# ==========================================
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
lgb_test_preds = np.zeros(len(X_test))
xgb_test_preds = np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(folds.split(X, y)):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    # --- Engine 1: LightGBM ---
    m_lgb = lgb.LGBMClassifier(
        n_estimators=1000, 
        learning_rate=0.05, 
        random_state=2026, 
        verbose=-1
    )
    m_lgb.fit(
        X_tr, y_tr, 
        eval_set=[(X_val, y_val)], 
        callbacks=[lgb.early_stopping(50)]
    )
    lgb_test_preds += m_lgb.predict_proba(X_test)[:, 1] / folds.n_splits

    # --- Engine 2: XGBoost ---
    m_xgb = xgb.XGBClassifier(
        n_estimators=1000, 
        learning_rate=0.05, 
        max_depth=4, 
        enable_categorical=True, 
        random_state=2026
    )
    m_xgb.fit(
        X_tr, y_tr, 
        eval_set=[(X_val, y_val)], 
        early_stopping_rounds=50, 
        verbose=False
    )
    xgb_test_preds += m_xgb.predict_proba(X_test)[:, 1] / folds.n_splits

# ==========================================
# 4. BLENDING (50:50)
# ==========================================
# 2つのモデルの予測値を平均化して安定性を高める
final_preds = (lgb_test_preds * 0.5) + (xgb_test_preds * 0.5)

# ==========================================
# 5. SUBMIT
# ==========================================
submission = pd.DataFrame({'id': test['id'], 'Heart Disease': final_preds})
submission.to_csv('sub_v20.csv', index=False)

print("V20: Ensemble model execution successful.")