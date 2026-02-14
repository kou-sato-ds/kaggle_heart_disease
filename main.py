import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
# ==========================================
# 1. DATA LOADING
# ==========================================
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# ==========================================
# 2. FEATURES (Iteration 18: Feature Engineering)
# ==========================================
def create_features(df):
    df = df.copy()
    df['BP_HR_Risk'] = df['Systolic BP'] * df['Max HR']

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
# 3. K-FOLD STRATIFIED SPLIT & TRAINING
# ==========================================
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
test_preds = np.zeros(len(X_test))
oof_preds = np.zeros(len(X))

# LightGBMのパラメータ
lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'random_state': 2026,
    'learning_rate': 0.05,
    'num_leaves': 31,
}

for fold, (tr_idx, val_idx) in enumerate(folds.split(X, y)):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks =[lgb.early_stopping(stopping_rounds=50)]
    )

    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    test_preds += model.predict_proba(X_test)[:, 1] / folds.n_splits

print(f"OOF AUC Score: {roc_auc_score(y, oof_preds):.4f}")

# ==========================================
# 4. SUBMIT
# ==========================================
submission = pd.DataFrame({'id': test['id'], 'Heart Disease': test_preds})
submission.to_csv('sub_v18.csv', index=False)
print("V18: LightGMB submission file created.")