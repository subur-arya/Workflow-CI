"""
MLflow Project - Model Training Script
Author: [ISI_NAMA_ANDA]
Description: Training script for MLflow Project with parameterization
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import argparse
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, matthews_corrcoef, cohen_kappa_score
)
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


def load_data():
    """Load preprocessed training and testing data"""
    print("Loading preprocessed data...")
    
    # Try different paths
    paths = [
        'telco_churn_train.csv',
        './telco_churn_train.csv',
        '../telco_churn_train.csv'
    ]
    
    for path in paths:
        if os.path.exists(path):
            X_train = pd.read_csv(path)
            test_path = path.replace('train', 'test')
            X_test = pd.read_csv(test_path)
            break
    else:
        raise FileNotFoundError("Dataset files not found!")
    
    # Separate features and target
    y_train = X_train['Churn']
    X_train = X_train.drop('Churn', axis=1)
    
    y_test = X_test['Churn']
    X_test = X_test.drop('Churn', axis=1)
    
    print(f"✓ Training data: {X_train.shape}")
    print(f"✓ Testing data: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test


def plot_confusion_matrix(y_test, y_pred, save_path='confusion_matrix.png'):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
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
    plt.savefig(save_path, dpi=150)
    plt.close()


def train_model(n_estimators=200, max_depth=15, min_samples_split=2, 
                min_samples_leaf=1, max_features='sqrt', random_state=42):
    """
    Train Random Forest model with specified parameters
    
    Args:
        n_estimators (int): Number of trees
        max_depth (int): Maximum depth of trees
        min_samples_split (int): Minimum samples to split node
        min_samples_leaf (int): Minimum samples in leaf
        max_features (str): Number of features to consider
        random_state (int): Random seed
    """
    
    print("\n" + "="*80)
    print("MLFLOW PROJECT - MODEL TRAINING")
    print("="*80)
    
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    
    # Set MLflow tracking
    mlflow.set_experiment("Telco_Churn_CI_Workflow")
    
    print("\n[1] Starting MLflow Run...")
    with mlflow.start_run(run_name="RandomForest_CI", nested=True):
        
        # Log parameters
        print("\n[2] Logging parameters...")
        params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_features': max_features,
            'random_state': random_state,
            'total_features': X_train.shape[1],
            'training_samples': len(X_train),
            'testing_samples': len(X_test)
        }
        mlflow.log_params(params)
        print("✓ Parameters logged!")
        
        # Initialize and train model
        print("\n[3] Training Random Forest model...")
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        print("✓ Model trained successfully!")
        
        # Make predictions
        print("\n[4] Making predictions...")
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        print("\n[5] Calculating metrics...")
        
        # Standard metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Additional metrics
        mcc = matthews_corrcoef(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp)
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        # Log all metrics
        print("\n[6] Logging metrics...")
        metrics = {
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
        }
        mlflow.log_metrics(metrics)
        print("✓ All metrics logged!")
        
        # Print metrics
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
        
        # Generate visualizations
        print("\n[7] Generating visualizations...")
        
        # Confusion Matrix
        cm_path = 'confusion_matrix.png'
        plot_confusion_matrix(y_test, y_pred, cm_path)
        mlflow.log_artifact(cm_path)
        print(f"✓ Logged: {cm_path}")
        
        # Feature Importance
        fi_path = 'feature_importance.png'
        plot_feature_importance(model, X_train.columns.tolist(), fi_path)
        mlflow.log_artifact(fi_path)
        print(f"✓ Logged: {fi_path}")
        
        # Log model
        print("\n[8] Logging model...")
        mlflow.sklearn.log_model(
            model, 
            "model",
            input_example=X_train.iloc[:5],
            signature=mlflow.models.infer_signature(X_train, y_pred)
        )
        print("✓ Model logged!")
        
        # Save model locally for Docker
        print("\n[9] Saving model locally...")
        model_path = "model"
        mlflow.sklearn.save_model(model, model_path)
        print(f"✓ Model saved to: {model_path}")
        
        # Get run info
        run = mlflow.active_run()
        print(f"\n✓ MLflow Run ID: {run.info.run_id}")
        print(f"✓ Artifact URI: {run.info.artifact_uri}")
        
        return run.info.run_id
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED!")
    print("="*80)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train model with MLflow Project')
    parser.add_argument('--n_estimators', type=int, default=200)
    parser.add_argument('--max_depth', type=int, default=15)
    parser.add_argument('--min_samples_split', type=int, default=2)
    parser.add_argument('--min_samples_leaf', type=int, default=1)
    parser.add_argument('--max_features', type=str, default='sqrt')
    parser.add_argument('--random_state', type=int, default=42)
    
    args = parser.parse_args()
    
    # Train model
    run_id = train_model(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        random_state=args.random_state
    )
    
    print(f"\n✓ Training completed! Run ID: {run_id}")
