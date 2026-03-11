"""
Simplified JurisTransformer that works with Python 3.6 and older scikit-learn
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import warnings
warnings.filterwarnings('ignore')

def run_juris_transformer(loader, text_columns=None, n_folds=3, save_plots_flag=True):
    """
    Run a simplified but powerful ensemble model for legal judgment prediction
    
    Args:
        loader: JudicialDataLoader instance with loaded data
        text_columns: List of text column names to use
        n_folds: Number of cross-validation folds
        save_plots_flag: Whether to save plots
    
    Returns:
        dict: Results including accuracy and f1 score
    """
    
    print("\n" + "="*60)
    print("🚀 RUNNING SIMPLIFIED JURISTRANSFORMER (ENSEMBLE MODEL)")
    print("="*60)
    
    # Get data from loader
    df = loader.df.copy()
    
    # Default text columns if none provided
    if text_columns is None:
        text_columns = ['name', 'issue.text', 'lower court.disposition', 
                        'lower court.reasons', 'arguments.petitioner.entity', 
                        'arguments.respondent.entity']
    
    # Filter to only existing columns
    text_columns = [col for col in text_columns if col in df.columns]
    print(f"\n📝 Using text columns: {text_columns}")
    
    # Create simple text features
    print("\n🔧 Creating text-based features...")
    df['combined_text'] = df[text_columns].fillna('').agg(' '.join, axis=1)
    
    # Simple text statistics
    df['text_length'] = df['combined_text'].str.len()
    df['word_count'] = df['combined_text'].str.split().str.len()
    df['avg_word_length'] = df['text_length'] / (df['word_count'] + 1)
    
    # Legal term frequencies
    legal_terms = ['court', 'law', 'judge', 'justice', 'appeal', 'decision', 
                   'ruling', 'plaintiff', 'defendant', 'petitioner', 'respondent',
                   'constitutional', 'statute', 'precedent', 'jurisdiction']
    
    for term in legal_terms:
        df[f'count_{term}'] = df['combined_text'].str.lower().str.count(term)
    
    print(f"   Added {len(legal_terms) + 3} text-based features")
    
    # Prepare features and target
    target_col = 'target_clean'
    y = loader.label_encoder.transform(df[target_col])
    
    # Get all numeric features (including newly created ones)
    exclude_cols = [target_col, 'combined_text'] + text_columns
    feature_cols = [col for col in df.columns 
                   if col not in exclude_cols 
                   and col in df.columns
                   and df[col].dtype in ['int64', 'float64', 'int8']]
    
    X = df[feature_cols].fillna(0).values
    
    print(f"\n📊 Final feature matrix: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"📊 Number of classes: {len(np.unique(y))}")
    
    # Cross-validation setup
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    all_predictions = np.zeros(len(y))
    all_true = y.copy()
    fold_results = []
    
    # Define ensemble models
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=300, 
            max_depth=20,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        'Extra Trees': ExtraTreesClassifier(
            n_estimators=300,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        ),
        'Logistic Regression': LogisticRegression(
            max_iter=2000,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
    }
    
    print(f"\n🔄 Running {n_folds}-fold cross-validation with ensemble...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n{'='*40}")
        print(f"FOLD {fold + 1}/{n_folds}")
        print(f"{'='*40}")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Train individual models and collect predictions
        fold_preds = []
        model_accuracies = []
        
        for name, model in models.items():
            start_time = time.time()
            model.fit(X_train_scaled, y_train)
            pred = model.predict(X_val_scaled)
            acc = accuracy_score(y_val, pred)
            model_accuracies.append(acc)
            fold_preds.append(pred)
            print(f"   {name:20} → Accuracy: {acc:.4f}  ({time.time()-start_time:.1f}s)")
        
        # Convert to numpy array for voting
        fold_preds = np.array(fold_preds)
        
        # Majority voting
        final_preds = []
        for i in range(len(y_val)):
            votes = fold_preds[:, i].astype(int)
            final_pred = np.bincount(votes).argmax()
            final_preds.append(final_pred)
        
        final_preds = np.array(final_preds)
        all_predictions[val_idx] = final_preds
        
        # Fold metrics
        fold_acc = accuracy_score(y_val, final_preds)
        fold_f1 = f1_score(y_val, final_preds, average='macro')
        
        fold_results.append({
            'fold': fold + 1,
            'accuracy': fold_acc,
            'f1_score': fold_f1,
            'model_accuracies': model_accuracies
        })
        
        print(f"\n   ✅ Ensemble Accuracy: {fold_acc:.4f}")
        print(f"   ✅ Ensemble Macro F1: {fold_f1:.4f}")
    
    # Overall results
    overall_acc = accuracy_score(all_true, all_predictions)
    overall_f1 = f1_score(all_true, all_predictions, average='macro')
    
    print("\n" + "="*60)
    print("📈 FINAL JURISTRANSFORMER RESULTS")
    print("="*60)
    print(f"Overall Accuracy: {overall_acc:.4f}")
    print(f"Overall Macro F1: {overall_f1:.4f}")
    
    print("\n📋 Per-fold Results:")
    for result in fold_results:
        print(f"   Fold {result['fold']}: Acc={result['accuracy']:.4f}, F1={result['f1_score']:.4f}")
    
    print("\n📋 Classification Report:")
    print(classification_report(all_true, all_predictions, target_names=loader.label_encoder.classes_))
    
    # Save plots if requested
    if save_plots_flag:
        os.makedirs('plots', exist_ok=True)
        
        # Confusion Matrix
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(all_true, all_predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=loader.label_encoder.classes_,
                   yticklabels=loader.label_encoder.classes_)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('JurisTransformer - Confusion Matrix')
        plt.tight_layout()
        plt.savefig('plots/juris_transformer_confusion_matrix.png', dpi=300)
        plt.close()
        
        # Feature importance (if possible)
        try:
            # Train a final Random Forest for feature importance
            rf_final = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
            scaler_final = StandardScaler()
            X_scaled = scaler_final.fit_transform(X)
            rf_final.fit(X_scaled, y)
            
            plt.figure(figsize=(12, 6))
            importances = rf_final.feature_importances_
            indices = np.argsort(importances)[::-1][:20]  # Top 20 features
            
            plt.barh(range(20), importances[indices])
            plt.yticks(range(20), [feature_cols[i] for i in indices])
            plt.xlabel('Feature Importance')
            plt.title('Top 20 Feature Importances')
            plt.tight_layout()
            plt.savefig('plots/juris_transformer_feature_importance.png', dpi=300)
            plt.close()
            print("\n✅ Feature importance plot saved")
        except:
            pass
        
        print("\n✅ Plots saved to 'plots/' directory")
    
    return {
        'accuracy': overall_acc,
        'f1_score': overall_f1,
        'fold_results': fold_results,
        'predictions': all_predictions,
        'true_labels': all_true,
        'class_names': loader.label_encoder.classes_
    }