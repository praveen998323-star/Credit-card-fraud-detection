"""
========================================================
  CREDIT CARD FRAUD DETECTION - MODEL TRAINING
========================================================
  This script trains a Machine Learning model to detect
  fraudulent credit card transactions.

  ALGORITHMS USED:
  ┌─────────────────────────────────────────────────┐
  │ 1. Random Forest    → Ensemble of decision trees│
  │ 2. XGBoost          → Gradient boosted trees   │
  │ 3. Logistic Regress → Baseline linear model    │
  └─────────────────────────────────────────────────┘

  TECHNIQUES USED:
  • SMOTE    → Oversampling to fix class imbalance
  • StandardScaler → Normalize Amount/Time features
  • Cross-Validation → Reliable performance estimate
  • Confusion Matrix → Detailed error analysis
  • ROC-AUC Score → Primary fraud detection metric
========================================================
"""

import numpy as np
import pandas as pd
import os, sys, joblib, warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection       import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing         import StandardScaler
from sklearn.ensemble              import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model          import LogisticRegression
from sklearn.metrics               import (classification_report, confusion_matrix,
                                           roc_auc_score, precision_recall_curve,
                                           average_precision_score, f1_score)
from imblearn.over_sampling        import SMOTE
from imblearn.pipeline             import Pipeline as ImbPipeline

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not installed. Skipping XGBoost model.")

print("=" * 65)
print("  CREDIT CARD FRAUD DETECTION — MODEL TRAINING")
print("=" * 65)
print()

# ───────────────────────────────────────────────────────────
# STEP 1: LOAD DATASET
#   The dataset has 31 columns:
#   - Time   : Seconds elapsed from first transaction
#   - V1-V28 : PCA-transformed anonymized features
#   - Amount : Transaction amount in dollars
#   - Class  : 0 = Legitimate, 1 = Fraudulent
# ───────────────────────────────────────────────────────────
print("📂 STEP 1: Loading Dataset...")
print("-" * 45)

dataset_path = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'credit_card_data.csv')

if not os.path.exists(dataset_path):
    print("❌ ERROR: Dataset not found!")
    print("   Please run: python dataset/generate_dataset.py first")
    sys.exit(1)

df = pd.read_csv(dataset_path)
print(f"   ✅ Loaded {len(df):,} transactions")
print(f"   ✅ Features: {df.shape[1]} columns")
print(f"   ✅ Fraud cases: {df['Class'].sum():,} ({df['Class'].mean()*100:.3f}%)")
print()

# ───────────────────────────────────────────────────────────
# STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
#   Before training, we understand the data distribution
# ───────────────────────────────────────────────────────────
print("🔍 STEP 2: Exploratory Data Analysis...")
print("-" * 45)

legit  = df[df['Class'] == 0]
fraud  = df[df['Class'] == 1]

print(f"   Legit Transactions  → Avg Amount: ${legit['Amount'].mean():.2f}")
print(f"   Fraud Transactions  → Avg Amount: ${fraud['Amount'].mean():.2f}")
print(f"   Legit Transactions  → Avg Time  : {legit['Time'].mean():.0f} sec")
print(f"   Fraud Transactions  → Avg Time  : {fraud['Time'].mean():.0f} sec")
print(f"   Missing Values      : {df.isnull().sum().sum()}")
print(f"   Duplicate Rows      : {df.duplicated().sum()}")
print()

# ───────────────────────────────────────────────────────────
# STEP 3: FEATURE ENGINEERING
#   - Scale 'Amount' and 'Time' (V1-V28 are already scaled)
#   - Drop original Amount/Time, use scaled versions
# ───────────────────────────────────────────────────────────
print("⚙️  STEP 3: Feature Engineering & Scaling...")
print("-" * 45)

scaler = StandardScaler()
df['scaled_amount'] = scaler.fit_transform(df[['Amount']])
df['scaled_time']   = scaler.fit_transform(df[['Time']])

# Save scaler for use in the web app
scaler_combined = StandardScaler()
df[['scaled_amount', 'scaled_time']] = scaler_combined.fit_transform(df[['Amount', 'Time']])

feature_cols = [f'V{i}' for i in range(1, 29)] + ['scaled_amount', 'scaled_time']
X = df[feature_cols]
y = df['Class']

print(f"   ✅ Scaled 'Amount' and 'Time' features")
print(f"   ✅ Feature matrix shape: {X.shape}")
print(f"   ✅ Target vector shape : {y.shape}")
print()

# ───────────────────────────────────────────────────────────
# STEP 4: TRAIN/TEST SPLIT
#   - 80% training, 20% testing
#   - Stratified: preserves fraud ratio in both splits
# ───────────────────────────────────────────────────────────
print("✂️  STEP 4: Train/Test Split...")
print("-" * 45)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size    = 0.20,
    random_state = 42,
    stratify     = y   # Ensures same fraud% in train & test
)

print(f"   Training set   : {len(X_train):,} samples")
print(f"   Testing set    : {len(X_test):,} samples")
print(f"   Training fraud : {y_train.sum():,} ({y_train.mean()*100:.2f}%)")
print(f"   Testing fraud  : {y_test.sum():,} ({y_test.mean()*100:.2f}%)")
print()

# ───────────────────────────────────────────────────────────
# STEP 5: HANDLE CLASS IMBALANCE WITH SMOTE
#   Problem: Only 0.17% are fraud → Model ignores fraud!
#   Solution: SMOTE creates SYNTHETIC fraud samples
#             by interpolating between real fraud cases
# ───────────────────────────────────────────────────────────
print("⚖️  STEP 5: Handling Class Imbalance with SMOTE...")
print("-" * 45)
print("   SMOTE = Synthetic Minority Oversampling TEchnique")
print("   → Creates artificial fraud samples from real ones")

smote = SMOTE(random_state=42, k_neighbors=5)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"   Before SMOTE → Fraud: {y_train.sum():,} | Legit: {(y_train==0).sum():,}")
print(f"   After  SMOTE → Fraud: {y_train_resampled.sum():,} | Legit: {(y_train_resampled==0).sum():,}")
print()

# ───────────────────────────────────────────────────────────
# STEP 6: TRAIN MULTIPLE MODELS
# ───────────────────────────────────────────────────────────
print("🤖 STEP 6: Training Machine Learning Models...")
print("-" * 45)

models = {}
results = {}

# ── MODEL A: Random Forest ──────────────────────────────────
print("\n   [A] Training Random Forest Classifier...")
print("       → An ensemble of 200 decision trees")
print("       → Each tree votes; majority wins")
rf = RandomForestClassifier(
    n_estimators     = 200,    # 200 decision trees
    max_depth        = 10,     # Maximum depth per tree
    min_samples_leaf = 2,      # Minimum samples at leaf
    class_weight     = 'balanced',
    random_state     = 42,
    n_jobs           = -1      # Use all CPU cores
)
rf.fit(X_train_resampled, y_train_resampled)
models['random_forest'] = rf
rf_pred  = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)[:, 1]
rf_auc   = roc_auc_score(y_test, rf_proba)
rf_f1    = f1_score(y_test, rf_pred)
results['Random Forest'] = {'auc': rf_auc, 'f1': rf_f1, 'pred': rf_pred, 'proba': rf_proba}
print(f"       ✅ AUC-ROC: {rf_auc:.4f} | F1-Score: {rf_f1:.4f}")

# ── MODEL B: Logistic Regression ────────────────────────────
print("\n   [B] Training Logistic Regression (Baseline)...")
print("       → Linear model; fast and interpretable")
lr = LogisticRegression(
    C            = 0.01,    # Regularization strength
    class_weight = 'balanced',
    max_iter     = 1000,
    random_state = 42
)
lr.fit(X_train_resampled, y_train_resampled)
models['logistic_regression'] = lr
lr_pred  = lr.predict(X_test)
lr_proba = lr.predict_proba(X_test)[:, 1]
lr_auc   = roc_auc_score(y_test, lr_proba)
lr_f1    = f1_score(y_test, lr_pred)
results['Logistic Regression'] = {'auc': lr_auc, 'f1': lr_f1, 'pred': lr_pred, 'proba': lr_proba}
print(f"       ✅ AUC-ROC: {lr_auc:.4f} | F1-Score: {lr_f1:.4f}")

# ── MODEL C: XGBoost ────────────────────────────────────────
if XGBOOST_AVAILABLE:
    print("\n   [C] Training XGBoost Classifier...")
    print("       → Gradient Boosting: builds trees sequentially")
    print("       → Each tree corrects errors of the previous")
    xgb = XGBClassifier(
        n_estimators    = 200,
        max_depth       = 6,
        learning_rate   = 0.05,
        scale_pos_weight= (y_train_resampled==0).sum() / (y_train_resampled==1).sum(),
        random_state    = 42,
        eval_metric     = 'logloss',
        use_label_encoder=False
    )
    xgb.fit(X_train_resampled, y_train_resampled)
    models['xgboost'] = xgb
    xgb_pred  = xgb.predict(X_test)
    xgb_proba = xgb.predict_proba(X_test)[:, 1]
    xgb_auc   = roc_auc_score(y_test, xgb_proba)
    xgb_f1    = f1_score(y_test, xgb_pred)
    results['XGBoost'] = {'auc': xgb_auc, 'f1': xgb_f1, 'pred': xgb_pred, 'proba': xgb_proba}
    print(f"       ✅ AUC-ROC: {xgb_auc:.4f} | F1-Score: {xgb_f1:.4f}")

# ───────────────────────────────────────────────────────────
# STEP 7: SELECT BEST MODEL & SHOW DETAILED RESULTS
# ───────────────────────────────────────────────────────────
print()
print("📊 STEP 7: Model Comparison & Selection...")
print("-" * 45)

best_model_name = max(results, key=lambda k: results[k]['auc'])
best_model = models[{
    'Random Forest'      : 'random_forest',
    'Logistic Regression': 'logistic_regression',
    'XGBoost'            : 'xgboost'
}[best_model_name]]

print(f"\n   {'Model':<22} {'AUC-ROC':>8} {'F1-Score':>10}")
print(f"   {'─'*22} {'─'*8} {'─'*10}")
for name, res in results.items():
    marker = " ← BEST" if name == best_model_name else ""
    print(f"   {name:<22} {res['auc']:>8.4f} {res['f1']:>10.4f}{marker}")

best_pred  = results[best_model_name]['pred']
best_proba = results[best_model_name]['proba']

print(f"\n   🏆 Best Model: {best_model_name}")
print()
print("   CLASSIFICATION REPORT:")
print(classification_report(y_test, best_pred, target_names=['Legit', 'Fraud'], digits=4))

cm = confusion_matrix(y_test, best_pred)
print("   CONFUSION MATRIX:")
print("                 Predicted")
print("                 Legit   Fraud")
print(f"   Actual Legit  {cm[0,0]:>6,}  {cm[0,1]:>6,}  ← False Positives = {cm[0,1]:,}")
print(f"   Actual Fraud  {cm[1,0]:>6,}  {cm[1,1]:>6,}  ← Caught fraud = {cm[1,1]:,}/{cm[1,0]+cm[1,1]:,}")
print()

# ───────────────────────────────────────────────────────────
# STEP 8: CROSS-VALIDATION
# ───────────────────────────────────────────────────────────
print("🔄 STEP 8: 5-Fold Stratified Cross-Validation...")
print("-" * 45)
print("   → Tests model stability across 5 different data splits")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(
    RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1),
    X, y,
    cv=cv, scoring='roc_auc'
)
print(f"   CV AUC Scores: {[f'{s:.4f}' for s in cv_scores]}")
print(f"   Mean AUC     : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print()

# ───────────────────────────────────────────────────────────
# STEP 9: FEATURE IMPORTANCE
# ───────────────────────────────────────────────────────────
print("📈 STEP 9: Feature Importance (Random Forest)...")
print("-" * 45)

importances = rf.feature_importances_
feat_imp_df = pd.DataFrame({
    'Feature'   : feature_cols,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print("   TOP 10 MOST IMPORTANT FEATURES:")
for _, row in feat_imp_df.head(10).iterrows():
    bar = '█' * int(row['Importance'] * 300)
    print(f"   {row['Feature']:>14}: {row['Importance']:.4f}  {bar}")
print()

# ───────────────────────────────────────────────────────────
# STEP 10: SAVE MODEL AND ARTIFACTS
# ───────────────────────────────────────────────────────────
print("💾 STEP 10: Saving Model & Artifacts...")
print("-" * 45)

model_dir = os.path.dirname(__file__)

joblib.dump(best_model, os.path.join(model_dir, 'fraud_model.pkl'))
joblib.dump(scaler_combined, os.path.join(model_dir, 'scaler.pkl'))
joblib.dump(feature_cols, os.path.join(model_dir, 'feature_cols.pkl'))

# Save feature importance for the dashboard
feat_imp_df.to_csv(os.path.join(model_dir, 'feature_importance.csv'), index=False)

# Save model comparison results
results_data = []
for name, res in results.items():
    results_data.append({'model': name, 'auc': res['auc'], 'f1': res['f1']})
pd.DataFrame(results_data).to_csv(os.path.join(model_dir, 'model_results.csv'), index=False)

# Save confusion matrix
np.save(os.path.join(model_dir, 'confusion_matrix.npy'), cm)

print(f"   ✅ Model saved       → model/fraud_model.pkl")
print(f"   ✅ Scaler saved      → model/scaler.pkl")
print(f"   ✅ Features saved    → model/feature_cols.pkl")
print(f"   ✅ Feature importance → model/feature_importance.csv")
print()
print("=" * 65)
print("  🎉 TRAINING COMPLETE!")
print("=" * 65)
print(f"  Best Model  : {best_model_name}")
print(f"  AUC-ROC     : {results[best_model_name]['auc']:.4f}")
print(f"  F1-Score    : {results[best_model_name]['f1']:.4f}")
print(f"  Fraud Caught: {cm[1,1]}/{cm[1,0]+cm[1,1]} ({cm[1,1]/(cm[1,0]+cm[1,1])*100:.1f}%)")
print()
print("  ➡️  NEXT STEP: Run 'python app.py' to launch the web dashboard")
print("=" * 65)
