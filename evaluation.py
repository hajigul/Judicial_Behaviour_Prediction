"""
Evaluation module for model comparison and visualization.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, classification_report, 
                           confusion_matrix, roc_curve, auc)
from sklearn.preprocessing import label_binarize
import time

class ModelEvaluator:
    """Handles model evaluation and visualization."""
    
    def __init__(self, label_encoder=None):
        """
        Initialize evaluator.
        
        Args:
            label_encoder: sklearn LabelEncoder for class names
        """
        self.label_encoder = label_encoder
        self.results = []
        
    def evaluate_model(self, model, X_test, y_test, model_name="Model"):
        """
        Evaluate a single model.
        
        Args:
            model: Trained model (with predict method)
            X_test: Test features
            y_test: Test labels
            model_name: Name for the model
            
        Returns:
            dict: Evaluation metrics
        """
        start = time.time()
        y_pred = model.predict(X_test)
        pred_time = time.time() - start
        
        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        
        result = {
            'Model': model_name,
            'Accuracy': round(acc, 4),
            'Macro F1': round(f1_macro, 4),
            'Weighted F1': round(f1_weighted, 4),
            'Prediction Time (sec)': round(pred_time, 4)
        }
        
        self.results.append(result)
        return result, y_pred
    
    def plot_model_comparison(self, results_df=None, metric='Accuracy', figsize=(14, 7)):
        """
        Plot bar chart comparing models.
        
        Args:
            results_df: DataFrame with results (if None, uses stored results)
            metric: Metric to plot ('Accuracy' or 'Macro F1')
            figsize: Figure size
        """
        if results_df is None:
            if hasattr(self, 'results_df'):
                results_df = self.results_df
            else:
                results_df = pd.DataFrame(self.results)
        
        plt.figure(figsize=figsize)
        sns.barplot(data=results_df.sort_values(by=metric, ascending=False), 
                   x=metric, y='Model', palette='viridis')
        plt.title(f'Model Comparison - {metric}')
        plt.xlabel(metric)
        plt.tight_layout()
        plt.show()
        
    def plot_confusion_matrix(self, y_true, y_pred, class_names=None, figsize=(10, 8)):
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names
            figsize: Figure size
        """
        if class_names is None and self.label_encoder is not None:
            class_names = self.label_encoder.classes_
        
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.show()
        
    def print_classification_report(self, y_true, y_pred, class_names=None):
        """Print detailed classification report."""
        if class_names is None and self.label_encoder is not None:
            class_names = self.label_encoder.classes_
        
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=class_names))
        
    def plot_training_history(self, history, figsize=(12, 4)):
        """
        Plot training history for deep learning models.
        
        Args:
            history: Keras history object
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Val Accuracy')
        plt.title('Accuracy over epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title('Loss over epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curves(self, model, X_test, y_test, n_classes, class_names=None):
        """
        Plot ROC curves for multi-class classification.
        
        Args:
            model: Model with predict_proba method
            X_test: Test features
            y_test: Test labels
            n_classes: Number of classes
            class_names: List of class names
        """
        if not hasattr(model, "predict_proba"):
            print("Model doesn't support predict_proba")
            return
            
        y_score = model.predict_proba(X_test)
        y_test_bin = label_binarize(y_test, classes=range(n_classes))
        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Plot all ROC curves
        plt.figure(figsize=(10, 8))
        colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, n_classes))
        
        for i, color in zip(range(n_classes), colors):
            if class_names and i < len(class_names):
                label = f'{class_names[i]} (AUC = {roc_auc[i]:.2f})'
            else:
                label = f'Class {i} (AUC = {roc_auc[i]:.2f})'
            plt.plot(fpr[i], tpr[i], color=color, lw=2, label=label)
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for Multi-class Classification')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.show()