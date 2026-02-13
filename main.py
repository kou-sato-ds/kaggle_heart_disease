import pandas as pd
import numpy as np
import zipfile
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# ==========================================
# 1. DATA LOADING
# ==========================================
# ※パスはご自身の環境に合わせて調整してください
try:
    with zipfile.ZipFile('playground-series-s6e2.zip', 'r') as z:
        z.extractall('data')
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
except FileNotFoundError:
    print("Zipファイルが見つかりません。パスを確認してください。")

# ==========================================
# 2. PREPROCESSING (堅牢化Ver.)
# ==========================================
def preprocess(df):
    df = df.copy()
    # 特徴量生成
    df['RPP'] = df['Max HR'] * df['Systolic BP']
    
    # 数値列の欠損値処理（中央値埋め）
    num_cols = df.select_dtypes(exclude='object').columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    return df

train = preprocess(train)
test = preprocess(test)

# ==========================================
# 3. ENCODING (ラベル一致の儀式)
# ==========================================
cat_cols = train.select_dtypes(include=['object']).columns
for col in cat_cols:
    if col == 'Heart Disease': continue
    le = LabelEncoder()
    # 訓練とテストを合体させてからFit
    full_labels = pd.concat([train[col].astype(str), test[col].astype(str)])
    le.fit(full_labels)
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

# 特徴量とターゲットの分離
X = train.drop(['id', 'Heart Disease'], axis=1, errors='ignore')
y = train['Heart Disease'].map({'Presence': 1, 'Absence': 0})
X_test = test.drop(['id'], axis=1, errors='ignore')

# 列の並びを合わせる
X, X_test = X.align(X_test, join='left', axis=1, fill_value=0)

# ==========================================
# 4. CROSS VALIDATION & SUBMIT
# ==========================================
seeds = [42, 2026]
test_preds = np.zeros(len(X_test))

for seed in seeds:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr_idx, val_idx in skf.split(X, y):
        model = RandomForestClassifier(n_estimators=500, max_depth=10, random_state=seed, n_jobs=-1)
        model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        test_preds += model.predict_proba(X_test)[:, 1] / (5 * len(seeds))

# 保存
submission = pd.DataFrame({'id': test['id'], 'Heart Disease': test_preds})
submission.to_csv('sub_v17.csv', index=False)
print("Finished: sub_v17.csv generated.")