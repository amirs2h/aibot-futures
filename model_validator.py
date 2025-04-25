# model_validator.py
# Walk-Forward Testing + Cross-Validation ماژول

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight
import joblib

class WalkForwardValidator:
    def __init__(self, df, model_fn, features_cols, label_col='signal', n_splits=5):
        self.df = df.copy()
        self.features_cols = features_cols
        self.label_col = label_col
        self.n_splits = n_splits
        self.model_fn = model_fn  # تابعی که مدل XGBoost جدید می‌سازه در هر تکرار
        self.metrics_log = []

    def evaluate_split(self, X_train, y_train, X_test, y_test):
        model = self.model_fn()
        sample_weight = compute_sample_weight(class_weight='balanced', y=y_train)
        model.fit(X_train, y_train, sample_weight=sample_weight)
        y_pred = model.predict(X_test)
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='macro'),
            'recall': recall_score(y_test, y_pred, average='macro'),
            'f1': f1_score(y_test, y_pred, average='macro')
        }

    def run_walk_forward(self):
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        X = self.df[self.features_cols].replace([np.inf, -np.inf], np.nan).dropna()
        y = self.df.loc[X.index, self.label_col]

        print(f"📆 اجرای Walk-Forward Validation با {self.n_splits} تکه زمانی:")

        for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
            metrics = self.evaluate_split(X_train, y_train, X_test, y_test)
            self.metrics_log.append(metrics)

            print(f"\n🔁 Fold {i+1}:")
            for k, v in metrics.items():
                print(f"{k.capitalize()}: {v:.4f}")

        return self.metrics_log

    def aggregate_results(self):
        df = pd.DataFrame(self.metrics_log)
        print("\n📊 میانگین نتایج Walk-Forward:")
        print(df.mean())
        return df.mean()

# تابع کمکی برای ساخت مدل جدید XGBoost در هر تکرار

def create_model():
    return xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='mlogloss'
    )

# طرز استفاده در فایل اصلی:
# validator = WalkForwardValidator(df, create_model, features.columns.tolist())
# validator.run_walk_forward()
# validator.aggregate_results()
