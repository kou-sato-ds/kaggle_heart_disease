!pip install autogluon  # Colabで実行する場合は最初にこれを行ってください

import pandas as pd
import zipfile
import os
from autogluon.tabular import TabularPredictor

# ==========================================
# 1. ZIP解凍 & データ読み込み
# ==========================================
def extract_and_read(zip_path, csv_name):
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall('data/extracted')
    return pd.read_csv(f'data/extracted/{csv_name}')

# ファイル名はアップロードしたものに合わせて適宜変更してください
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# 2つのZipからオリジナルデータを取得
# 例: archive (1).zip 内の heart_disease_uci.csv
orig_1 = extract_and_read('/content/archive (1).zip', 'heart_disease_uci.csv') # パスを修正
# 例: archive (2).zip 内の heart_cleveland_upload.csv
orig_2 = extract_and_read('/content/archive (2).zip', 'heart_cleveland_upload.csv') # パスを修正

# ========================================== 
# 2. 名寄せ (Data Alignment) 
# ========================================== 
# コンペ側のカラム名に合わせるための変換辞書
# ※UCIデータセットの列名をコンペデータセットの列名に統一
rename_dict_1 = {
    'age': 'Age',
    'sex': 'Sex',
    'cp': 'Chest pain type',
    'trestbps': 'BP',
    'chol': 'Cholesterol',
    'fbs': 'FBS over 120',
    'restecg': 'EKG results',
    'thalch': 'Max HR',
    'exang': 'Exercise angina',
    'oldpeak': 'ST depression',
    'slope': 'Slope of ST',
    'ca': 'Number of vessels fluro',
    'thal': 'Thallium',
    'num': 'Heart Disease'
}

rename_dict_2 = {
    'age': 'Age',
    'sex': 'Sex',
    'cp': 'Chest pain type',
    'trestbps': 'BP',
    'chol': 'Cholesterol',
    'fbs': 'FBS over 120',
    'restecg': 'EKG results',
    'thalach': 'Max HR',
    'exang': 'Exercise angina',
    'oldpeak': 'ST depression',
    'slope': 'Slope of ST',
    'ca': 'Number of vessels fluro',
    'thal': 'Thallium',
    'condition': 'Heart Disease' # 'condition'を'Heart Disease'に修正
}

orig_1 = orig_1.rename(columns=rename_dict_1)
orig_2 = orig_2.rename(columns=rename_dict_2)

# UCIデータセットにのみ存在する列を削除（またはコンペデータに合わせて変換）
# 'dataset'列はコンペデータに存在しないため削除
if 'dataset' in orig_1.columns:
    orig_1 = orig_1.drop(columns=['dataset'])
# 'id'列はコンペのtrainデータにはあるが、orig_1, orig_2には不要なので削除
if 'id' in orig_1.columns:
    orig_1 = orig_1.drop(columns=['id'])
# orig_2にはid列はないので確認不要

# ターゲット値を Presence(1) / Absence(0) に統一
# オリジナルが数値(0, 1, 2...)の場合は、0をAbsence、1以上をPresenceにするのが一般的です
def unify_target(val):
    if val == 0 or val == 'Absence': return 'Absence'
    return 'Presence'

orig_1['Heart Disease'] = orig_1['Heart Disease'].apply(unify_target)
orig_2['Heart Disease'] = orig_2['Heart Disease'].apply(unify_target)

# 全データを結合！
train_full = pd.concat([train, orig_1, orig_2], axis=0).reset_index(drop=True)
print(f"Total Training Data: {len(train_full)} samples")

# ==========================================
# 3. FEATURE ENGINEERING
# ==========================================
def create_features(df):
    df = df.copy()
    if 'BP' in df.columns and 'Max HR' in df.columns:
        df['BP_HR_Risk'] = df['BP'] * df['Max HR']
    return df

train_full = create_features(train_full)
test = create_features(test)

# ==========================================
# 4. AUTOGLUON TRAINING (The Magic)
# ==========================================
# AutoGluon用の目的変数は文字列のままでもOKです
predictor = TabularPredictor(
    label='Heart Disease',
    eval_metric='roc_auc',
    path='ag_models'
).fit(
    train_data=train_full.drop(columns=['id'], errors='ignore'),
    presets='best_quality', # 自動スタッキングを有効化
    time_limit=1800        # まずは30分で試行
)

# ==========================================
# 5. PREDICT & SUBMIT
# ==========================================
# 確率予測 (Presenceである確率を取得)
preds_proba = predictor.predict_proba(test)
final_preds = preds_proba['Presence']

submission = pd.DataFrame({'id': test['id'], 'Heart Disease': final_preds})
submission.to_csv('sub_v22_autogluon_zip.csv', index=False)

# モデルの貢献度を確認
print(predictor.leaderboard(silent=True))