"""
========================================================
  CREDIT CARD FRAUD DETECTION - DATASET GENERATOR
========================================================
  This script generates a REALISTIC synthetic dataset
  that mimics real credit card transaction patterns.

  WHAT IT CREATES:
  - 50,000 transactions (normal + fraudulent)
  - 30 features (V1-V28 PCA features + Amount + Time)
  - Only ~0.17% fraud cases (realistic imbalance)
  
  HOW IT WORKS:
  - Normal transactions follow regular spending patterns
  - Fraudulent transactions have anomalous feature values
  - Amount and Time are raw features (not PCA-transformed)
========================================================
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os

# ─────────────────────────────────────────────
# STEP 1: Set random seed for reproducibility
# ─────────────────────────────────────────────
np.random.seed(42)
random.seed(42)

print("=" * 60)
print("  CREDIT CARD FRAUD DATASET GENERATOR")
print("=" * 60)
print()

# ─────────────────────────────────────────────
# STEP 2: Define dataset parameters
# ─────────────────────────────────────────────
TOTAL_TRANSACTIONS = 50000
FRAUD_RATE = 0.0017          # 0.17% fraud rate (realistic)
N_FRAUD = int(TOTAL_TRANSACTIONS * FRAUD_RATE)
N_NORMAL = TOTAL_TRANSACTIONS - N_FRAUD
N_FEATURES = 28              # V1 through V28 (PCA components)

print(f"📊 Dataset Configuration:")
print(f"   Total Transactions  : {TOTAL_TRANSACTIONS:,}")
print(f"   Normal Transactions : {N_NORMAL:,}")
print(f"   Fraud Transactions  : {N_FRAUD:,}")
print(f"   Fraud Rate          : {FRAUD_RATE*100:.2f}%")
print(f"   PCA Features        : V1 - V{N_FEATURES}")
print()

# ─────────────────────────────────────────────
# STEP 3: Generate TIME feature
#   - Represents seconds elapsed from first transaction
#   - 2-day period = 172800 seconds
# ─────────────────────────────────────────────
print("⏱️  Step 1/6: Generating TIME feature...")
normal_time = np.sort(np.random.uniform(0, 172800, N_NORMAL))
fraud_time  = np.random.uniform(0, 172800, N_FRAUD)   # Fraud at random times

# ─────────────────────────────────────────────
# STEP 4: Generate AMOUNT feature
#   - Normal: follows log-normal distribution
#     (most purchases small, some large)
#   - Fraud: smaller amounts (fraudsters test cards)
# ─────────────────────────────────────────────
print("💰 Step 2/6: Generating AMOUNT feature...")
normal_amount = np.random.lognormal(mean=3.0, sigma=1.5, size=N_NORMAL)
normal_amount = np.clip(normal_amount, 0.01, 25691.16)

fraud_amount = np.random.lognormal(mean=2.5, sigma=1.2, size=N_FRAUD)
fraud_amount = np.clip(fraud_amount, 0.01, 2000)

# ─────────────────────────────────────────────
# STEP 5: Generate V1-V28 PCA features
#   - These represent anonymized card features
#   - Normal: centered around 0 (standard normal)
#   - Fraud: shifted means to simulate anomalies
# ─────────────────────────────────────────────
print("🔢 Step 3/6: Generating V1-V28 PCA features...")

# Normal transactions: near-zero mean features
normal_features = np.random.randn(N_NORMAL, N_FEATURES)

# Fraud transactions: shifted/anomalous features
# Key features that typically differ in fraud:
fraud_shifts = {
    'V1':  -3.0,  'V2':  2.5,   'V3':  -4.0,  'V4':  3.0,
    'V5':  -2.5,  'V6':  -1.5,  'V7':  -5.0,  'V8':  2.0,
    'V9':  -2.0,  'V10': -4.5,  'V11': 3.5,   'V12': -6.0,
    'V13': 0.5,   'V14': -7.0,  'V15': 0.3,   'V16': -3.5,
    'V17': -5.5,  'V18': -2.8,  'V19': 1.0,   'V20': 0.8,
    'V21': 1.5,   'V22': 0.6,   'V23': -0.5,  'V24': 0.2,
    'V25': 0.4,   'V26': -0.3,  'V27': 1.8,   'V28': 0.9,
}

fraud_means = np.array([fraud_shifts[f'V{i+1}'] for i in range(N_FEATURES)])
fraud_features = np.random.randn(N_FRAUD, N_FEATURES) + fraud_means

# ─────────────────────────────────────────────
# STEP 6: Add realistic noise and patterns
# ─────────────────────────────────────────────
print("🎲 Step 4/6: Adding realistic patterns and noise...")

# Daytime vs nighttime spending patterns for normal transactions
for i, t in enumerate(normal_time):
    hour = (t % 86400) / 3600
    if 22 <= hour or hour <= 6:   # Late night → smaller amounts
        normal_amount[i] *= 0.4
    elif 11 <= hour <= 14:         # Lunch hours → medium amounts
        normal_amount[i] *= 1.2

# Add correlated features for normal transactions
normal_features[:, 0] -= 0.3 * normal_features[:, 3]   # V1 correlated with V4
normal_features[:, 6] -= 0.5 * normal_features[:, 9]   # V7 correlated with V10

# ─────────────────────────────────────────────
# STEP 7: Combine and create DataFrame
# ─────────────────────────────────────────────
print("📦 Step 5/6: Combining data into DataFrame...")

# Normal transactions
normal_df = pd.DataFrame(normal_features, columns=[f'V{i+1}' for i in range(N_FEATURES)])
normal_df['Time']   = normal_time
normal_df['Amount'] = np.round(normal_amount, 2)
normal_df['Class']  = 0   # 0 = Legitimate

# Fraud transactions
fraud_df = pd.DataFrame(fraud_features, columns=[f'V{i+1}' for i in range(N_FEATURES)])
fraud_df['Time']   = fraud_time
fraud_df['Amount'] = np.round(fraud_amount, 2)
fraud_df['Class']  = 1   # 1 = Fraudulent

# Combine and shuffle
full_df = pd.concat([normal_df, fraud_df], ignore_index=True)
full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Reorder columns: Time, V1-V28, Amount, Class
cols = ['Time'] + [f'V{i+1}' for i in range(N_FEATURES)] + ['Amount', 'Class']
full_df = full_df[cols]

# ─────────────────────────────────────────────
# STEP 8: Save to CSV
# ─────────────────────────────────────────────
print("💾 Step 6/6: Saving dataset to CSV...")
output_path = os.path.join(os.path.dirname(__file__), 'credit_card_data.csv')
full_df.to_csv(output_path, index=False)

# ─────────────────────────────────────────────
# STEP 9: Print summary statistics
# ─────────────────────────────────────────────
print()
print("=" * 60)
print("  ✅ DATASET GENERATED SUCCESSFULLY!")
print("=" * 60)
print(f"  📁 Saved to: {output_path}")
print(f"  📐 Shape   : {full_df.shape[0]:,} rows × {full_df.shape[1]} columns")
print()
print("  📊 CLASS DISTRIBUTION:")
class_counts = full_df['Class'].value_counts()
for cls, count in class_counts.items():
    label = "FRAUD (1)" if cls == 1 else "LEGIT (0)"
    pct   = count / len(full_df) * 100
    print(f"     {label}: {count:,} transactions ({pct:.3f}%)")
print()
print("  💰 AMOUNT STATISTICS:")
print(f"     Min    : ${full_df['Amount'].min():.2f}")
print(f"     Max    : ${full_df['Amount'].max():,.2f}")
print(f"     Mean   : ${full_df['Amount'].mean():.2f}")
print(f"     Median : ${full_df['Amount'].median():.2f}")
print()
print("  🔍 FEATURE PREVIEW (first 3 rows):")
print(full_df[['Time', 'V1', 'V2', 'V3', 'Amount', 'Class']].head(3).to_string())
print()
print("  ➡️  NEXT STEP: Run 'python model/train_model.py'")
print("=" * 60)
