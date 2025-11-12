"""
Disease Prediction Model Training Script
Author: Priyanka Malavade
Description: Train multiple ML models for disease prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix, 
                            classification_report)
from sklearn.datasets import load_breast_cancer
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("DISEASE PREDICTION MODEL TRAINING")
print("="*70)

# ============================================================================
# 1. DIABETES PREDICTION MODEL
# ============================================================================
print("\n" + "="*70)
print("1. TRAINING DIABETES PREDICTION MODEL")
print("="*70)

# Load diabetes dataset
print("\ Loading Diabetes Dataset...")
df_diabetes = pd.read_csv('data/diabetes.csv')
print(f" Loaded: {df_diabetes.shape[0]} samples, {df_diabetes.shape[1]} features")
print(f"\nDataset Info:")
print(df_diabetes.info())
print(f"\nTarget Distribution:")
print(df_diabetes['Outcome'].value_counts())

# Prepare data
X_diabetes = df_diabetes.drop('Outcome', axis=1)
y_diabetes = df_diabetes['Outcome']

# Split data
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
    X_diabetes, y_diabetes, test_size=0.2, random_state=42, stratify=y_diabetes
)

# Scale features
scaler_diabetes = StandardScaler()
X_train_d_scaled = scaler_diabetes.fit_transform(X_train_d)
X_test_d_scaled = scaler_diabetes.transform(X_test_d)

print(f"\n Training Set: {X_train_d_scaled.shape[0]} samples")
print(f" Test Set: {X_test_d_scaled.shape[0]} samples")

# Train multiple models
print("\n Training Multiple Models...")

models_diabetes = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss'),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

results_diabetes = {}

for name, model in models_diabetes.items():
    print(f"\n   Training {name}...")
    
    # Train model
    model.fit(X_train_d_scaled, y_train_d)
    
    # Predictions
    y_pred = model.predict(X_test_d_scaled)
    y_pred_proba = model.predict_proba(X_test_d_scaled)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_d, y_pred)
    precision = precision_score(y_test_d, y_pred)
    recall = recall_score(y_test_d, y_pred)
    f1 = f1_score(y_test_d, y_pred)
    roc_auc = roc_auc_score(y_test_d, y_pred_proba)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_d_scaled, y_train_d, cv=5)
    
    results_diabetes[name] = {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }
    
    print(f"   {name}: Accuracy={accuracy:.4f}, F1={f1:.4f}, ROC-AUC={roc_auc:.4f}")

# Select best model
best_model_name_d = max(results_diabetes, key=lambda x: results_diabetes[x]['accuracy'])
best_model_d = results_diabetes[best_model_name_d]['model']

print(f"\n Best Model: {best_model_name_d}")
print(f"   Accuracy: {results_diabetes[best_model_name_d]['accuracy']:.4f}")
print(f"   F1-Score: {results_diabetes[best_model_name_d]['f1_score']:.4f}")
print(f"   ROC-AUC: {results_diabetes[best_model_name_d]['roc_auc']:.4f}")

# Save models
print("\n Saving Diabetes Models...")
joblib.dump(best_model_d, 'models/diabetes_model.pkl')
joblib.dump(scaler_diabetes, 'models/diabetes_scaler.pkl')
joblib.dump(results_diabetes, 'models/diabetes_results.pkl')
print("Models saved successfully!")

# ============================================================================
# 2. HEART DISEASE PREDICTION MODEL
# ============================================================================
print("\n" + "="*70)
print("2. TRAINING HEART DISEASE PREDICTION MODEL")
print("="*70)

# Load heart disease dataset
print("\n Loading Heart Disease Dataset...")
df_heart = pd.read_csv('data/heart_disease.csv')
print(f"Loaded: {df_heart.shape[0]} samples, {df_heart.shape[1]} features")
print(f"\nDataset Info:")
print(df_heart.info())
print(f"\nTarget Distribution:")
print(df_heart['target'].value_counts())

# Prepare data
X_heart = df_heart.drop('target', axis=1)
y_heart = df_heart['target']

# Split data
X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
    X_heart, y_heart, test_size=0.2, random_state=42, stratify=y_heart
)

# Scale features
scaler_heart = StandardScaler()
X_train_h_scaled = scaler_heart.fit_transform(X_train_h)
X_test_h_scaled = scaler_heart.transform(X_test_h)

print(f"\n Training Set: {X_train_h_scaled.shape[0]} samples")
print(f" Test Set: {X_test_h_scaled.shape[0]} samples")

# Train multiple models
print("\n Training Multiple Models...")

models_heart = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss'),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

results_heart = {}

for name, model in models_heart.items():
    print(f"\n   Training {name}...")
    
    # Train model
    model.fit(X_train_h_scaled, y_train_h)
    
    # Predictions
    y_pred = model.predict(X_test_h_scaled)
    y_pred_proba = model.predict_proba(X_test_h_scaled)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_h, y_pred)
    precision = precision_score(y_test_h, y_pred)
    recall = recall_score(y_test_h, y_pred)
    f1 = f1_score(y_test_h, y_pred)
    roc_auc = roc_auc_score(y_test_h, y_pred_proba)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_h_scaled, y_train_h, cv=5)
    
    results_heart[name] = {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }
    
    print(f"   {name}: Accuracy={accuracy:.4f}, F1={f1:.4f}, ROC-AUC={roc_auc:.4f}")

# Select best model
best_model_name_h = max(results_heart, key=lambda x: results_heart[x]['accuracy'])
best_model_h = results_heart[best_model_name_h]['model']

print(f"\n Best Model: {best_model_name_h}")
print(f"   Accuracy: {results_heart[best_model_name_h]['accuracy']:.4f}")
print(f"   F1-Score: {results_heart[best_model_name_h]['f1_score']:.4f}")
print(f"   ROC-AUC: {results_heart[best_model_name_h]['roc_auc']:.4f}")

# Save models
print("\ Saving Heart Disease Models...")
joblib.dump(best_model_h, 'models/heart_model.pkl')
joblib.dump(scaler_heart, 'models/heart_scaler.pkl')
joblib.dump(results_heart, 'models/heart_results.pkl')
print(" Models saved successfully!")

# ============================================================================
# 3. BREAST CANCER PREDICTION MODEL
# ============================================================================
print("\n" + "="*70)
print("3. TRAINING BREAST CANCER PREDICTION MODEL")
print("="*70)

# Load breast cancer dataset (built-in)
print("\n Loading Breast Cancer Dataset...")
cancer_data = load_breast_cancer()
X_cancer = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
y_cancer = cancer_data.target

print(f" Loaded: {X_cancer.shape[0]} samples, {X_cancer.shape[1]} features")
print(f"\nTarget Distribution:")
print(pd.Series(y_cancer).value_counts())

# Split data
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_cancer, y_cancer, test_size=0.2, random_state=42, stratify=y_cancer
)

# Scale features
scaler_cancer = StandardScaler()
X_train_c_scaled = scaler_cancer.fit_transform(X_train_c)
X_test_c_scaled = scaler_cancer.transform(X_test_c)

print(f"\n Training Set: {X_train_c_scaled.shape[0]} samples")
print(f" Test Set: {X_test_c_scaled.shape[0]} samples")

# Train multiple models
print("\n Training Multiple Models...")

models_cancer = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss'),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

results_cancer = {}

for name, model in models_cancer.items():
    print(f"\n   Training {name}...")
    
    # Train model
    model.fit(X_train_c_scaled, y_train_c)
    
    # Predictions
    y_pred = model.predict(X_test_c_scaled)
    y_pred_proba = model.predict_proba(X_test_c_scaled)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_c, y_pred)
    precision = precision_score(y_test_c, y_pred)
    recall = recall_score(y_test_c, y_pred)
    f1 = f1_score(y_test_c, y_pred)
    roc_auc = roc_auc_score(y_test_c, y_pred_proba)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_c_scaled, y_train_c, cv=5)
    
    results_cancer[name] = {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }
    
    print(f"    {name}: Accuracy={accuracy:.4f}, F1={f1:.4f}, ROC-AUC={roc_auc:.4f}")

# Select best model
best_model_name_c = max(results_cancer, key=lambda x: results_cancer[x]['accuracy'])
best_model_c = results_cancer[best_model_name_c]['model']

print(f"\n Best Model: {best_model_name_c}")
print(f"   Accuracy: {results_cancer[best_model_name_c]['accuracy']:.4f}")
print(f"   F1-Score: {results_cancer[best_model_name_c]['f1_score']:.4f}")
print(f"   ROC-AUC: {results_cancer[best_model_name_c]['roc_auc']:.4f}")

# Save models
print("\n Saving Breast Cancer Models...")
joblib.dump(best_model_c, 'models/cancer_model.pkl')
joblib.dump(scaler_cancer, 'models/cancer_scaler.pkl')
joblib.dump(results_cancer, 'models/cancer_results.pkl')
joblib.dump(cancer_data.feature_names, 'models/cancer_feature_names.pkl')
print(" Models saved successfully!")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("TRAINING COMPLETE - SUMMARY")
print("="*70)

print(f"\n1. DIABETES PREDICTION:")
print(f"   Best Model: {best_model_name_d}")
print(f"   Accuracy: {results_diabetes[best_model_name_d]['accuracy']:.4f}")
print(f"   ROC-AUC: {results_diabetes[best_model_name_d]['roc_auc']:.4f}")

print(f"\n2. HEART DISEASE PREDICTION:")
print(f"   Best Model: {best_model_name_h}")
print(f"   Accuracy: {results_heart[best_model_name_h]['accuracy']:.4f}")
print(f"   ROC-AUC: {results_heart[best_model_name_h]['roc_auc']:.4f}")

print(f"\n3. BREAST CANCER PREDICTION:")
print(f"   Best Model: {best_model_name_c}")
print(f"   Accuracy: {results_cancer[best_model_name_c]['accuracy']:.4f}")
print(f"   ROC-AUC: {results_cancer[best_model_name_c]['roc_auc']:.4f}")

print("\n" + "="*70)
print(" ALL MODELS TRAINED AND SAVED SUCCESSFULLY!")
print("="*70)
print("\nSaved files in 'models/' directory:")
print("   - diabetes_model.pkl, diabetes_scaler.pkl")
print("   - heart_model.pkl, heart_scaler.pkl")
print("   - cancer_model.pkl, cancer_scaler.pkl")
print("\nYou can now run the Streamlit app: streamlit run app.py")
print("="*70)