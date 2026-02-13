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