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

## 🚀 Update: 2026-02-14 (Iteration 21)
Implemented a sophisticated **Stacking Ensemble** and prepared for **AutoML integration** with real-world clinical data.

### 🛠️ Key Achievements
1. **Advanced Stacking (V21)**:
   - Integrated **LightGBM** and **XGBoost (Native API)** as base models.
   - Utilized **Logistic Regression** as a meta-model to intelligently blend predictions based on Out-Of-Fold (OOF) validation scores.
   - Achieved a more robust prediction pipeline compared to simple averaging.

2. **Data Enrichment Strategy**:
   - Identified and prepared two major **UCI Heart Disease datasets** (373 & 207 votes) to supplement synthetic training data.
   - Focused on "Data Alignment" (feature renaming and mapping) to ensure seamless integration.

### 📈 Score Progression
- **V20 (Blending)**: AUC 0.95339
- **V21 (Stacking)**: **AUC 0.95348 (Current Best)**

### 💡 Next Mission: AutoML Conquest
Tomorrow, I will leverage **AutoGluon** to perform multi-layer stacking using the full dataset (Kaggle + UCI). This aims to break into the **Silver Tier** by maximizing both human domain knowledge and automated model optimization.

## 🚀 Update: 2026-02-14 (Iteration 22 - Final)
The final stage of the Kaggle Heart Disease challenge: **"Human Intel x AI Power."** Integrated external UCI datasets and leveraged AutoGluon for high-level automated stacking.

### 🛠️ Key Achievements
1. **Automated Multi-layer Stacking (V22)**:
   - Employed **AutoGluon (TabularPredictor)** with `best_quality` preset.
   - Automatically optimized an ensemble of various models (LightGBM, CatBoost, XGBoost, Neural Networks).
2. **Data Enrichment & Engineering**:
   - Automated ZIP extraction and data alignment for external sources.
   - Merged 500+ real-world clinical samples into the training pipeline.

### 📚 Data Sources & Acknowledgments
To improve model robustness, the following datasets were integrated:
- **Heart Disease Dataset (373 votes)** - [Redwan Karim Sony]
- **Heart Disease Cleveland Dataset (207 votes)** - [Cherngs]

### 📈 Final Roadmap
Completed Kaggle Playground Series. Next, I will apply these skills to **SIGNATE** (Japan's premier DS platform) to tackle real-world industrial and business challenges in Japan.

## 🚀 Update: 2026-02-14 (Iteration 22 - Final Achievement)
Successfully integrated real-world UCI clinical data and utilized **AutoGluon** for the ultimate ensemble strategy. This marks the completion of my Kaggle Heart Disease challenge.

### 🛠️ Key Achievements
1. **Robust Data Integration**: 
   - Automated extraction and mapping of two major UCI datasets (580+ samples).
   - Solved complex "Column Name Mismatch" issues, aligning external data with competition standards.
2. **AutoML Excellence (V22)**:
   - Implemented multi-layer stacking using AutoGluon's `best_quality` preset.
   - Achieved a highly generalized model capable of handling both synthetic and real patient data.

### 📈 Final Results & Reflection
- **Best Score (Manual Stacking)**: AUC 0.95348
- **Final Leap**: Transitioned from manual feature engineering to sophisticated automated pipelines.

### 🇯🇵 Next Challenge: SIGNATE
With the implementation skills gained here, I am moving to **SIGNATE** to solve Japan-specific industrial and business data challenges. The journey of "continuous transcription and implementation" continues.

## 🏁 Conclusion of Kaggle Heart Disease Challenge
- **Final Version**: V22 (AutoGluon with External UCI Data Integration)
- **Key Learning**: Mastered data alignment, automated ZIP handling, and high-performance AutoML pipelines.
- **Next Step**: Moving to **SIGNATE** to apply these implementation skills to Japanese industrial data. 

*Thank you, Kaggle! Ready for the next challenge in the Japanese data science community.*

## 🏁 Kaggle Completion & Transition
Successfully finalized the Heart Disease project with V22. 
Key takeaway: Advanced data preprocessing and AutoML integration.

### 🚗 New Goal: SIGNATE Car Mileage Prediction
Moving forward to the **SIGNATE** platform. 
I will tackle the **Car Mileage Prediction** challenge to master "Regression" tasks using my "Golden 6 Steps" framework.