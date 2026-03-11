"""
Simplified JurisTransformer that works with older Python versions
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# Try to import transformers (optional)
try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except:
    TRANSFORMERS_AVAILABLE = False
    print("Note: transformers not available. Using simplified version without BERT.")

class SimplifiedJurisTransformer:
    """
    A simplified version of JurisTransformer that works with Python 3.6
    """
    
    def __init__(self, n_folds=3):
        self.n_folds = n_folds
        self.label_encoder = LabelEncoder()
        
    def extract_text_features(self, df, text_columns):
        """
        Simple text features (word counts, etc.) instead of BERT
        """
        print("\n📝 Extracting simple text features...")
        
        # Combine text columns
        df['combined_text'] = df[text_columns].fillna('').agg(' '.join, axis=1)
        
        # Simple text features
        df['text_length'] = df['combined_text'].str.len()
        df['word_count'] = df['combined_text'].str.split().str.len()
        df['avg_word_length'] = df['text_length'] / (df['word_count'] + 1)
        
        # Count of legal terms (simple approach)
        legal_terms = ['court', 'law', 'judge', 'justice', 'appeal', 'decision', 
                      'ruling', 'plaintiff', 'defendant', 'petitioner', 'respondent']
        
        for term in legal_terms:
            df[f'has_{term}'] = df['combined_text'].str.contains(term, case=False).astype(int)
            df[f'count_{term}'] = df['combined_text'].str.lower().str.count(term)
        
        print(f"   Added {len(legal_terms)*2 + 3} text-based features")
        return df
    
    def create_ensemble_model(self):
        """Create an ensemble of multiple models"""
        models = {
            'rf': RandomForestClassifier(n_estimators=200, max_depth=15, 
                                        class_weight='balanced', random_state=42, n_jobs=-1),
            'gb': GradientBoostingClassifier(n_estimators=150, max_depth=5, 
                                            random_state=42),
            'lr': LogisticRegression(max_iter=2000, class_weight='balanced', 
                                     random_state=42, n_jobs=-1)
        }
        return models
    
    def fit_predict_cv(self, X, y, n_splits=3):
        """
        Cross-validated training and prediction
        """
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        all_predictions = np.zeros(len(y))
        all_probabilities = np.zeros((len(y), len(np.unique(y))))
        all_true = y.copy()
        
        fold_scores = []
        
        print(f"\n🔄 Running {n_splits}-fold cross-validation...")
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"\n📊 Fold {fold+1}/{n_splits}")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Train ensemble
            models = self.create_ensemble_model()
            fold_preds = np.zeros((len(y_val), len(models)))
            
            for i, (name, model) in enumerate(models.items()):
                model.fit(X_train_scaled, y_train)
                fold_preds[:, i] = model.predict(X_val_scaled)
                
                # Individual model accuracy
                acc = accuracy_score(y_val, fold_preds[:, i])
                print(f"   {name}: Accuracy = {acc:.4f}")
            
            # Majority voting
            final_preds = []
            for j in range(len(y_val)):
                votes = fold_preds[j, :].astype(int)
                final_pred = np.bincount(votes).argmax()
                final_preds.append(final_pred)
            
            final_preds = np.array(final_preds)
            all_predictions[val_idx] = final_preds
            
            # Fold accuracy
            fold_acc = accuracy_score(y_val, final_preds)
            fold_f1 = f1_score(y_val, final_preds, average='macro')
            fold_scores.append({'fold': fold+1, 'accuracy': fold_acc, 'f1': fold_f1})
            
            print(f"   ✅ Fold {fold+1} Ensemble Accuracy: {fold_acc:.4f}, F1: {fold_f1:.4f}")
        
        # Overall results
        overall_acc = accuracy_score(all_true, all_predictions)
        overall_f1 = f1_score(all_true, all_predictions, average='macro')
        
        return {
            'predictions': all_predictions,
            'true_labels': all_true,
            'accuracy': overall_acc,
            'f1_score': overall_f1,
            'fold_scores': fold_scores
        }
    
    def run(self, df, text_columns, target_col='target_clean'):
        """
        Main method to run the simplified JurisTransformer
        """
        print("\n" + "="*60)
        print("🚀 SIMPLIFIED JURISTRANSFORMER")
        print("="*60)
        
        # Prepare data
        df_processed = df.copy()
        
        # Extract text features
        df_processed = self.extract_text_features(df_processed, text_columns)
        
        # Get target
        y = self.label_encoder.fit_transform(df_processed[target_col])
        
        # Get features (exclude target and text columns)
        exclude_cols = [target_col, 'combined_text'] + text_columns
        feature_cols = [col for col in df_processed.columns 
                       if col not in exclude_cols and col in df_processed.columns]
        
        X = df_processed[feature_cols].fillna(0).values
        
        print(f"\n📊 Feature matrix shape: {X.shape}")
        print(f"📊 Number of classes: {len(np.unique(y))}")
        
        # Run cross-validation
        results = self.fit_predict_cv(X, y, n_splits=self.n_folds)
        
        # Print results
        print("\n" + "="*60)
        print("📈 FINAL RESULTS")
        print("="*60)
        print(f"Overall Accuracy: {results['accuracy']:.4f}")
        print(f"Overall Macro F1: {results['f1_score']:.4f}")
        
        print("\n📋 Classification Report:")
        print(classification_report(results['true_labels'], results['predictions'],
                                   target_names=self.label_encoder.classes_))
        
        # Plot confusion matrix
        self.plot_confusion_matrix(results['true_labels'], results['predictions'],
                                  self.label_encoder.classes_)
        
        return results
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names):
        """Plot and save confusion matrix"""
        os.makedirs('plots', exist_ok=True)
        
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Simplified JurisTransformer - Confusion Matrix')
        plt.tight_layout()
        plt.savefig('plots/simplified_juris_confusion_matrix.png', dpi=300)
        plt.show()
        print("\n✅ Confusion matrix saved to 'plots/simplified_juris_confusion_matrix.png'")