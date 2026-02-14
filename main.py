import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# ==========================================
# 1. DATA LOADING & PREP
# ==========================================
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

def create_features(df):
    df = df.copy()
    # 'Systolic BP'を'BP'にリネームしてリスク計算
    if 'Systolic BP' in df.columns:
        df = df.rename(columns={'Systolic BP': 'BP'})
    
    if 'BP' in df.columns and 'Max HR' in df.columns:
        df['BP_HR_Risk'] = df['BP'] * df['Max HR']
        
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if col != 'Heart Disease':
            df[col] = df[col].astype('category')
    return df

train = create_features(train)
test = create_features(test)

X = train.drop(['id', 'Heart Disease'], axis=1)
y = train['Heart Disease'].map({'Presence': 1, 'Absence': 0}) # 大文字に修正
X_test = test.drop(['id'], axis=1)

# ==========================================
# 2. STACKING: LEVEL 1 (Base Models)
# ==========================================
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)

oof_lgb = np.zeros(len(X))
oof_xgb = np.zeros(len(X))
test_lgb = np.zeros(len(X_test))
test_xgb = np.zeros(len(X_test))

xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'random_state': 2026,
    'learning_rate': 0.05,
    'max_depth': 4,
    'enable_categorical': True, # タイポ修正
    'tree_method': 'hist',
    'n_estimators': 1000
}

print("Start Level 1 Training...")
for fold, (tr_idx, val_idx) in enumerate(folds.split(X, y)):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    # Model 1: LightGBM
    m_lgb = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.05, random_state=2026, verbose=-1)
    m_lgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50)])
    oof_lgb[val_idx] = m_lgb.predict_proba(X_val)[:, 1]
    test_lgb += m_lgb.predict_proba(X_test)[:, 1] / folds.n_splits # 除算を追加

    # Model 2: XGBoost (Native API)
    dtrain = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True) # スペル修正
    dval = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)
    dtest = xgb.DMatrix(X_test, enable_categorical=True)

    m_xgb = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=xgb_params['n_estimators'],
        evals=[(dval, 'validation')],
        early_stopping_rounds=50,
        verbose_eval=False
    )
    oof_xgb[val_idx] = m_xgb.predict(dval, iteration_range=(0, m_xgb.best_iteration))
    test_xgb += m_xgb.predict(dtest, iteration_range=(0, m_xgb.best_iteration)) / folds.n_splits # 除算を追加

# ==========================================
# 3. STACKING: LEVEL 2 (Meta Model)
# ==========================================
stack_X = pd.DataFrame({'lgb_preds': oof_lgb, 'xgb_preds': oof_xgb})
stack_test = pd.DataFrame({'lgb_preds': test_lgb, 'xgb_preds': test_xgb})

print("Meta-Model Learning...")
meta_model = LogisticRegression()
meta_model.fit(stack_X, y)
final_preds = meta_model.predict_proba(stack_test)[:, 1]

# ==========================================
# 4. SUBMIT
# ==========================================
submission = pd.DataFrame({'id': test['id'], 'Heart Disease': final_preds})
submission.to_csv('sub_v21_stacking.csv', index=False)
print("V21: Stacking implementation complete!")