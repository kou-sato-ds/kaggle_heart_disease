# !pip install autogluon

import pandas as pd
import zipfile
import os
from autogluon.tabular import TabularPredictor

# ==========================================
# 1. ZIP解凍 & データ読み込み
# ==========================================
def extract_and_read(zip_path, csv_name):
    # 解凍先のディレクトリ作成
    extract_dir = 'data/extracted'
    os.makedirs(extract_dir, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_dir)
    return pd.read_csv(os.path.join(extract_dir, csv_name))

# 基本データの読み込み
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# Zipからオリジナルデータを取得 (タイポ修正済み)
orig_1 = extract_and_read('data/archive (1).zip', 'heart_disease_uci.csv')
orig_2 = extract_and_read('data/archive (2).zip', 'heart_cleveland_upload.csv')

# ==========================================
# 2. 名寄せ (Data Alignment)
# ==========================================
# コンペ側のカラム名に合わせる変換辞書
rename_dict_1 = {
    'thalch': 'Max HR',
    'trestbps': 'BP',
    'num': 'Heart Disease'
}
rename_dict_2 = {
    'trestbps': 'BP',
    'thalach': 'Max HR',
    'condition': 'Heart Disease'
}

orig_1 = orig_1.rename(columns=rename_dict_1)
orig_2 = orig_2.rename(columns=rename_dict_2)

# ターゲット値の統一 (0: Absence, 1以上: Presence)
def unify_target(val):
    if val == 0 or val == 'Absence': return 'Absence'
    return 'Presence'

orig_1['Heart Disease'] = orig_1['Heart Disease'].apply(unify_target)
orig_2['Heart Disease'] = orig_2['Heart Disease'].apply(unify_target)

# 全データを結合
train_full = pd.concat([train, orig_1, orig_2], axis=0).reset_index(drop=True)
print(f"Total Training Data: {len(train_full)} samples")

# ==========================================
# 3. FEATURE ENGINEERING
# ==========================================
def create_features(df):
    df = df.copy()
    # 昨日の修正を反映：Systolic BP or BP
    bp_col = 'Systolic BP' if 'Systolic BP' in df.columns else 'BP'
    if bp_col in df.columns and 'Max HR' in df.columns:
        df['BP_HR_Risk'] = df[bp_col] * df['Max HR']
    return df

train_full = create_features(train_full)
test = create_features(test)

# ==========================================
# 4. AUTOGLUON TRAINING
# ==========================================
predictor = TabularPredictor(
    label='Heart Disease',
    eval_metric='roc_auc',
    path='ag_models'
).fit(
    train_data=train_full.drop(columns=['id'], errors='ignore'),
    presets='best_quality',
    time_limit=1800
)

# ==========================================
# 5. PREDICT & SUBMIT
# ==========================================
preds_proba = predictor.predict_proba(test)
final_preds = preds_proba['Presence']

submission = pd.DataFrame({'id': test['id'], 'Heart Disease': final_preds})
submission.to_csv('sub_v22_autogluon_zip.csv', index=False)

print(predictor.leaderboard(silent=True))
print("V22: Process Complete!")