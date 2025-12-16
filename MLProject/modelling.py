"""
MLflow Project - Ultra Simple Training Script
No tracking URI setting - let MLflow Project handle everything!
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, matthews_corrcoef, cohen_kappa_score
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


def plot_confusion_matrix(y_test, y_pred, save_path='confusion_matrix.png'):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return cm


def plot_feature_importance(model, feature_names, save_path='feature_importance.png', top_n=15):
    """Plot and save feature importance"""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    plt.figure(figsize=(10, 6))
    plt.title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
    plt.bar(range(top_n), importances[indices], color='skyblue')
    plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_estimators', type=int, default=200)
    parser.add_argument('--max_depth', type=int, default=15)
    parser.add_argument('--min_samples_split', type=int, default=2)
    parser.add_argument('--min_samples_leaf', type=int, default=1)
    parser.add_argument('--max_features', type=str, default='sqrt')
    parser.add_argument('--random_state', type=int, default=42)
    args = parser.parse_args()
    
    print("="*80)
    print("PARAMETERS")
    print("="*80)
    print(f"n_estimators:      {args.n_estimators}")
    print(f"max_depth:         {args.max_depth}")
    print(f"min_samples_split: {args.min_samples_split}")
    print(f"min_samples_leaf:  {args.min_samples_leaf}")
    print(f"max_features:      {args.max_features}")
    print(f"random_state:      {args.random_state}")
    print("="*80)
    
    # Load data
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    X_train = pd.read_csv('telco_churn_train.csv')
    X_test = pd.read_csv('telco_churn_test.csv')
    
    y_train = X_train['Churn']
    X_train = X_train.drop('Churn', axis=1)
    y_test = X_test['Churn']
    X_test = X_test.drop('Churn', axis=1)
    
    print(f"✓ Training data: {X_train.shape}")
    print(f"✓ Testing data: {X_test.shape}")
    print("="*80)
    
    # Train model
    print("\n" + "="*80)
    print("TRAINING MODEL")
    print("="*80)
    
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        random_state=args.random_state,
        n_jobs=-1
    )
    
    print("Training Random Forest...")
    model.fit(X_train, y_train)
    print("✓ Model trained successfully!")
    
    # Predictions
    print("\nMaking predictions...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    print("✓ Predictions complete!")
    
    # Calculate metrics
    print("\nCalculating metrics...")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    mcc = matthews_corrcoef(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    print("✓ Metrics calculated!")
    
    # Log to MLflow
    print("\n" + "="*80)
    print("LOGGING TO MLFLOW")
    print("="*80)
    
    print("Logging parameters...")
    mlflow.log_params({
        'n_estimators': args.n_estimators,
        'max_depth': args.max_depth,
        'min_samples_split': args.min_samples_split,
        'min_samples_leaf': args.min_samples_leaf,
        'max_features': args.max_features,
        'random_state': args.random_state,
        'total_features': X_train.shape[1],
        'training_samples': len(X_train),
        'testing_samples': len(X_test)
    })
    print("✓ Parameters logged!")
    
    print("Logging metrics...")
    mlflow.log_metrics({
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'matthews_corrcoef': mcc,
        'cohen_kappa': kappa,
        'specificity': specificity,
        'negative_predictive_value': npv,
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp)
    })
    print("✓ Metrics logged!")
    
    # Print results
    print("\n" + "="*80)
    print("MODEL PERFORMANCE")
    print("="*80)
    print("STANDARD METRICS:")
    print(f"  Accuracy:   {accuracy:.4f}")
    print(f"  Precision:  {precision:.4f}")
    print(f"  Recall:     {recall:.4f}")
    print(f"  F1-Score:   {f1:.4f}")
    print(f"  ROC-AUC:    {roc_auc:.4f}")
    print("\nADDITIONAL METRICS:")
    print(f"  Matthews CC: {mcc:.4f}")
    print(f"  Cohen Kappa: {kappa:.4f}")
    print(f"  Specificity: {specificity:.4f}")
    print(f"  NPV:         {npv:.4f}")
    print("\nCONFUSION MATRIX:")
    print(f"  TN: {tn:4d}  |  FP: {fp:4d}")
    print(f"  FN: {fn:4d}  |  TP: {tp:4d}")
    print("="*80)
    
    # Generate visualizations (SAVE FIRST, LOG LATER!)
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    print("Creating confusion matrix...")
    plot_confusion_matrix(y_test, y_pred, 'confusion_matrix.png')
    print("✓ Confusion matrix saved!")
    
    print("Creating feature importance plot...")
    plot_feature_importance(model, X_train.columns.tolist(), 'feature_importance.png')
    print("✓ Feature importance saved!")
    
    # Try to log artifacts (with error handling)
    print("\nLogging artifacts to MLflow...")
    try:
        mlflow.log_artifact('confusion_matrix.png')
        print("✓ Confusion matrix logged!")
    except Exception as e:
        print(f"⚠ Could not log confusion matrix: {e}")
    
    try:
        mlflow.log_artifact('feature_importance.png')
        print("✓ Feature importance logged!")
    except Exception as e:
        print(f"⚠ Could not log feature importance: {e}")
    
    # Save model
    print("\n" + "="*80)
    print("SAVING MODEL")
    print("="*80)
    
    print("Saving model locally...")
    try:
        mlflow.sklearn.save_model(model, "model")
        print("✓ Model saved to: ./model/")
    except Exception as e:
        print(f"⚠ Could not save model: {e}")
    
    # Try to log model
    print("Logging model to MLflow...")
    try:
        mlflow.sklearn.log_model(
            model, 
            "model",
            input_example=X_train.iloc[:5]
        )
        print("✓ Model logged to MLflow!")
    except Exception as e:
        print(f"⚠ Could not log model: {e}")
    
    # Done
    print("\n" + "="*80)
    print("✓ TRAINING COMPLETED!")
    print("="*80)
    
    print("\n✓ Artifacts generated:")
    print("  - confusion_matrix.png")
    print("  - feature_importance.png")
    print("  - model/ (if save succeeded)")
    
    print("\n✓ Script execution complete!")
