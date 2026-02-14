# Kaggle Heart Disease Prediction (Playground Series S6E2)

This repository contains a robust machine learning pipeline for the "Heart Disease Prediction" competition on Kaggle. 
The primary goal of this project is to implement a production-ready, error-resistant pipeline rather than just chasing leaderboard scores.

## 🛠 Key Features (Iteration 17: Robust Pipeline)

In this version, I focused on building a "hard-to-break" implementation:

1. **Automated Missing Value Handling**
   - Implemented median imputation by dynamically selecting numerical columns using `select_dtypes`.
   - Designed to be generalized for any future data updates.

2. **Robust Categorical Encoding**
   - Solved the "Unseen Labels" error by fitting the `LabelEncoder` on a concatenated dataset of both train and test sets.
   - Ensures the script runs without failure even if new categories appear in the test set.

3. **Stability through Seed Averaging**
   - Used multiple random seeds (`42`, `2026`) and `StratifiedKFold` to reduce variance.
   - Aimed for better generalization performance across different data splits.

## 🚀 Technologies
- Python 3.x
- Pandas / NumPy
- Scikit-learn (RandomForestClassifier, StratifiedKFold, LabelEncoder)

## 📂 Project Structure
- `main.py`: Main script for preprocessing, training, and generating submission files.
- `data/`: Dataset folder (managed locally).

## 🚀 Update: 2026-02-14 (Iteration 18-20)
Today, I successfully implemented **Ensemble Learning (Blending)** and achieved a new **Personal Best score (AUC: 0.95339)**.

### 🛠️ Key Achievements
1. **Algorithm Diversification**:
   - Implemented and compared **LightGBM (V18)** and **XGBoost (V19)**.
   - Deepened my understanding of how different gradient boosting frameworks handle categorical data and missing values.

2. **Blending Ensemble (V20)**:
   - Combined predictions from LGBM and XGBoost using a 50:50 weighted average.
   - This technique effectively balanced the individual biases of each model, enhancing overall generalization.

3. **Feature Engineering**:
   - Created `BP_HR_Risk` (Interaction between Systolic BP and Max HR) to capture critical health indicators.

### 📈 Score Progression
- **V18 (LightGBM)**: AUC 0.95168
- **V19 (XGBoost)**: AUC 0.95326
- **V20 (Ensemble)**: **AUC 0.95339 (Personal Best)**

### 💡 Future Roadmap
- **External Data Integration**: Plan to merge the original UCI clinical dataset to improve model robustness.
- **AutoML Exploration**: Experiment with **AutoGluon** for automated hyperparameter tuning and advanced stacking.