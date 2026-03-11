"""
Base models module containing all traditional ML models.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, 
                            GradientBoostingClassifier, AdaBoostClassifier, 
                            VotingClassifier)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
import time
import pandas as pd
import numpy as np
import warnings




# Add this import at the TOP of base_models.py
from sklearn.utils.class_weight import compute_class_weight




warnings.filterwarnings('ignore')

# Try to import HistGradientBoostingClassifier, but don't fail if not available
try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    HIST_GRADIENT_AVAILABLE = True
except ImportError:
    HIST_GRADIENT_AVAILABLE = False
    print("Note: HistGradientBoostingClassifier not available in your scikit-learn version.")
    print("You can upgrade scikit-learn with: pip install --upgrade scikit-learn")



class BaseModelTrainer:
    """Trains and evaluates multiple base ML models."""
    
    def __init__(self, preprocessor):
        """
        Initialize with preprocessor.
        
        Args:
            preprocessor: sklearn ColumnTransformer object
        """
        self.preprocessor = preprocessor
        self.models = None  # Will be defined later
        self.results = []
        self.y_train = None  # Add this to store y_train for class weights
    
    def _define_models(self, y_train=None):
        """Define all base models with proper class weights."""
        
        # Compute class weights if y_train is provided
        if y_train is not None and len(np.unique(y_train)) > 1:
            try:
                class_weights = compute_class_weight(
                    'balanced', 
                    classes=np.unique(y_train), 
                    y=y_train
                )
                class_weight_dict = dict(zip(np.unique(y_train), class_weights))
                print(f"   Using balanced class weights: {class_weight_dict}")
            except:
                class_weight_dict = 'balanced'
                print("   Using 'balanced' class weights")
        else:
            class_weight_dict = 'balanced'
        
        # Define models with proper class weights
        models = {
            "Logistic Regression": LogisticRegression(
                max_iter=2000,  # Increased iterations
                class_weight=class_weight_dict,
                random_state=42, 
                n_jobs=-1,
                solver='lbfgs'
            ),
            "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
            "Naive Bayes": GaussianNB(),
            "Random Forest": RandomForestClassifier(
                n_estimators=300, 
                max_depth=None, 
                class_weight=class_weight_dict, 
                random_state=42, 
                n_jobs=-1
            ),
            "Extra Trees": ExtraTreesClassifier(
                n_estimators=300, 
                class_weight=class_weight_dict, 
                random_state=42, 
                n_jobs=-1
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=150, 
                random_state=42
            ),
            "XGBoost": XGBClassifier(
                n_estimators=300, 
                learning_rate=0.1, 
                max_depth=8, 
                random_state=42, 
                eval_metric='mlogloss', 
                n_jobs=-1
            ),
            "Support Vector Machine (RBF)": SVC(
                kernel='rbf', 
                class_weight=class_weight_dict, 
                random_state=42, 
                probability=True
            ),
            "MLP Neural Network": MLPClassifier(
                hidden_layer_sizes=(100, 50), 
                max_iter=500, 
                random_state=42
            ),
            "Decision Tree": DecisionTreeClassifier(
                class_weight=class_weight_dict, 
                random_state=42
            ),
            "AdaBoost": AdaBoostClassifier(
                n_estimators=100, 
                random_state=42
            ),
        }
        
        # Add HistGradientBoosting if available
        try:
            from sklearn.ensemble import HistGradientBoostingClassifier
            models["Hist Gradient Boosting"] = HistGradientBoostingClassifier(
                max_iter=300, 
                random_state=42,
                class_weight=class_weight_dict  # If supported
            )
        except:
            pass
        
        return models
    
    def train_all_models(self, X_train, X_test, y_train, y_test, verbose=True):
        """
        Train all models and collect results.
        
        Args:
            X_train, X_test, y_train, y_test: train/test data
            verbose (bool): Whether to print progress
            
        Returns:
            pd.DataFrame: Results dataframe
        """
        self.results = []
        self.y_train = y_train  # Store for class weights
        
        # Initialize models with class weights based on y_train
        self.models = self._define_models(y_train)
        
        print("Training all base models with improved settings...\n")
        
        for name, clf in self.models.items():
            try:
                start = time.time()
                
                pipe = Pipeline([
                    ('preprocessor', self.preprocessor),
                    ('classifier', clf)
                ])
                
                pipe.fit(X_train, y_train)
                y_pred = pipe.predict(X_test)
                
                from sklearn.metrics import accuracy_score, f1_score
                acc = accuracy_score(y_test, y_pred)
                f1_macro = f1_score(y_test, y_pred, average='macro')
                
                self.results.append({
                    'Model': name,
                    'Accuracy': round(acc, 4),
                    'Macro F1': round(f1_macro, 4),
                    'Time (sec)': round(time.time() - start, 1)
                })
                
                if verbose:
                    print(f"{name:30} → Accuracy: {acc:.4f} | Macro F1: {f1_macro:.4f}  ({time.time()-start:.1f}s)")
            
            except Exception as e:
                print(f" Error training {name}: {str(e)}")
                continue
        
        self.results_df = pd.DataFrame(self.results).sort_values(by='Accuracy', ascending=False)
        return self.results_df
    
    def create_voting_ensemble(self, estimators=None):
        """
        Create a voting ensemble classifier.
        
        Args:
            estimators: List of (name, model) tuples. If None, uses default.
            
        Returns:
            VotingClassifier: Ensemble model
        """
        if estimators is None:
            estimators = [
                ('rf', RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42)),
                ('xgb', XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=8, 
                                      random_state=42, eval_metric='mlogloss'))
            ]
        
        voting_clf = VotingClassifier(estimators=estimators, voting='soft')
        return voting_clf
    
    def get_best_model(self):
        """Return the best performing model."""
        if not self.results:
            return None, None
        best_model_name = self.results_df.iloc[0]['Model']
        return best_model_name, self.models[best_model_name]