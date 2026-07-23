# FIX for model_evaluation.ipynb, cell 1 --------------------------------
# The original version of this cell reloaded the data with skiprows=...(1/10
# of rows) for a quick SGD test, then OVERWROTE X_train/X_test/y_train/y_test.
# Cells 2 and 3 (Random Forest and the final LightGBM model) then trained on
# that 10%-sized subsample instead of the full split from cell 0 -- nobody
# intended that, and it's why the saved model behaves erratically on inputs
# slightly outside the dense part of the training distribution.
#
# Fix: give the quick SGD test its OWN variable names so it can't clobber the
# real split used by every model after it.

import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import time
import gc

# Load a subsample ONLY for this quick SGD sanity check
df_sample = pd.read_csv(
    "credit_risk_cleaned.csv",
    skiprows=lambda i: i != 0 and i % 10 != 0,
    low_memory=False
)
print(f"Loaded (subsample for SGD test only): {df_sample.shape[0]:,} rows")

df_sample = df_sample.dropna(subset=["default"])
df_sample["default"] = df_sample["default"].astype(int)

keep_cols = [
    "loan_amnt", "int_rate", "installment", "annual_inc",
    "dti", "fico_range_low", "fico_range_high", "revol_util",
    "revol_bal", "open_acc", "total_acc", "pub_rec",
    "emp_length", "grade", "sub_grade", "delinq_2yrs",
    "inq_last_6mths", "mort_acc", "pub_rec_bankruptcies",
    "total_rev_hi_lim", "avg_cur_bal", "bc_util",
    "pct_tl_nvr_dlq", "num_actv_bc_tl", "num_actv_rev_tl",
    "default"
]
keep_cols = [c for c in keep_cols if c in df_sample.columns]
df_sample = df_sample[keep_cols]

# IMPORTANT: use the SAME keep_cols to subset the REAL X_train/X_test that
# were already built in cell 0 from the full dataset -- do NOT re-split here.
X_train_sgd = X_train[[c for c in keep_cols if c != "default"]].copy()
X_test_sgd = X_test[[c for c in keep_cols if c != "default"]].copy()

for col in X_train_sgd.columns:
    if X_train_sgd[col].isnull().any():
        med = X_train_sgd[col].median()
        X_train_sgd[col] = X_train_sgd[col].fillna(med)
        X_test_sgd[col] = X_test_sgd[col].fillna(med)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_sgd)
X_test_scaled = scaler.transform(X_test_sgd)
gc.collect()

print("\nTraining Logistic Regression (SGD)...")
start = time.time()
lr = SGDClassifier(loss="log_loss", class_weight="balanced", random_state=26,
                    max_iter=1000, tol=1e-4, n_jobs=-1)
lr.fit(X_train_scaled, y_train)  # y_train is the REAL full-size target from cell 0
print(f"Done in {time.time() - start:.1f}s")

lr_probs = lr.predict_proba(X_test_scaled)[:, 1]
lr_preds = lr.predict(X_test_scaled)
print(f"\nAUC-ROC: {roc_auc_score(y_test, lr_probs):.4f}")
print(classification_report(y_test, lr_preds, target_names=["Fully Paid", "Default"]))

# ------------------------------------------------------------------------
# From here on, cells 2 (Random Forest) and 3 (LightGBM) should subset
# X_train/X_test (the real, full-size ones from cell 0) down to keep_cols
# instead of relying on anything this cell defined:
#
#   X_train_model = X_train[[c for c in keep_cols if c != "default"]]
#   X_test_model  = X_test[[c for c in keep_cols if c != "default"]]
#
# ...and train on X_train_model / y_train (full-size), then re-save
# credit_risk_model.lgb and re-download it before redeploying.
