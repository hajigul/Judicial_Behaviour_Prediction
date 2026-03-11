"""
Data loading and preprocessing module for the judicial dataset.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PolynomialFeatures, KBinsDiscretizer
import warnings
warnings.filterwarnings('ignore')

class JudicialDataLoader:
    """Handles loading and preprocessing of the Supreme Court dataset."""
    
    def __init__(self, file_path):
        """
        Initialize the data loader with file path.
        
        Args:
            file_path (str): Path to the CSV file
        """
        self.file_path = file_path
        self.df = None
        self.X = None
        self.y = None
        self.y_encoded = None
        self.label_encoder = None
        self.preprocessor = None
        self.numeric_cols = None
        self.cat_cols = None
        self.boolean_cols = None
        
    def load_data(self):
        """Load the dataset from CSV file."""
        print(f"Loading data from: {self.file_path}")
        self.df = pd.read_csv(self.file_path)
        print(f"Dataset loaded! Shape: {self.df.shape}")
        print(f"Columns: {self.df.columns.tolist()}")
        return self.df
    
    def clean_target(self, target_col='decision.case.disposition'):
        """
        Clean and consolidate target column values.
        
        Args:
            target_col (str): Name of the target column
        """
        print(f"\nCleaning target column: {target_col}")
        print("Target column value counts (top 30):")
        print(self.df[target_col].value_counts(dropna=False).head(30))
        print(f"\nUnique values in target: {self.df[target_col].nunique()}")
        
        def clean_disposition(text):
            if pd.isna(text):
                return 'unknown'
            t = str(text).lower().strip()
            if 'affirmed' in t:
                return 'affirmed'
            elif 'reversed and remanded' in t:
                return 'reversed_and_remanded'
            elif 'reversed' in t:
                return 'reversed'
            elif 'vacated and remanded' in t:
                return 'vacated_and_remanded'
            elif 'vacated' in t:
                return 'vacated'
            elif 'dismissed' in t:
                return 'dismissed'
            elif 'petition denied' in t or 'appeal dismissed' in t:
                return 'petition_denied'
            else:
                return 'other'
        
        self.df['target_clean'] = self.df[target_col].apply(clean_disposition)
        print("\nCleaned target distribution:")
        print(self.df['target_clean'].value_counts())
        return self.df
    
    def drop_leakage_columns(self):
        """Drop columns that could cause data leakage."""
        leakage_cols = [
            'decision.case.disposition', 'decision.authority', 'decision.direction',
            'decision.dissent agrees', 'decision.jurisdiction', 'decision.precedent altered?',
            'decision.term', 'decision.type', 'decision.unconstitutional', 'decision.winning party',
            'decision.admin action.agency', 'decision.admin action.id', 'decision.case.unusual',
            'decision.date.day', 'decision.date.full', 'decision.date.month', 'decision.date.year',
            'voting.majority', 'voting.minority', 'voting.split on second', 'voting.unclear',
            'voting.majority assigner.id', 'voting.majority assigner.name',
            'voting.majority writer.id', 'voting.majority writer.name',
            'id.case', 'id.case issues', 'id.docket', 'id.vote',
            'citation.led', 'citation.lexis', 'citation.sct', 'citation.us',
            '3_judge_dc'
        ]
        
        drop_cols = [col for col in leakage_cols if col in self.df.columns]
        self.df = self.df.drop(columns=drop_cols)
        print(f"\nDropped {len(drop_cols)} leakage columns. New shape: {self.df.shape}")
        return self.df
    
    def convert_boolean_columns(self):
        """Detect and convert boolean columns to int8."""
        print("\n🔧 Detecting and converting TRUE/FALSE columns...")
        
        self.boolean_cols = []
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                unique_vals = self.df[col].dropna().astype(str).str.strip().str.upper().unique()
                unique_set = set(unique_vals)
                if len(unique_set) <= 8 and any(v in unique_set for v in ['TRUE', 'FALSE']):
                    self.boolean_cols.append(col)
        
        print(f"Detected {len(self.boolean_cols)} real boolean columns:")
        print(self.boolean_cols)
        
        for col in self.boolean_cols:
            self.df[col] = (self.df[col]
                           .astype(str)
                           .str.strip()
                           .str.upper()
                           .replace({'TRUE': 1, 'FALSE': 0, 'UNKNOWN': 0, 'NAN': 0, '': 0, 'NONE': 0})
                           .fillna(0)
                           .astype('int8'))
        
        print("\nAll boolean columns converted successfully!")
        return self.df
    
    def add_polynomial_features(self, degree=2):
        """Add polynomial features for important numeric columns."""
        print("\n🔧 Adding polynomial interaction features...")
        
        # First identify numeric columns
        numeric_cols = self.df.select_dtypes(include=['int64', 'float64', 'int8']).columns.tolist()
        
        # Select important numeric columns (those with enough unique values)
        important_numeric = []
        for col in numeric_cols:
            if col in self.df.columns and self.df[col].nunique() > 10:
                important_numeric.append(col)
        
        # Limit to top 5
        important_numeric = important_numeric[:5]
        
        if len(important_numeric) >= 2:
            poly = PolynomialFeatures(degree=degree, interaction_only=True, include_bias=False)
            
            # Create interaction features
            X_subset = self.df[important_numeric].fillna(0)
            poly_features = poly.fit_transform(X_subset)
            
            # Add to dataframe
            try:
                feature_names = poly.get_feature_names_out(important_numeric)
            except:
                # Fallback for older sklearn versions
                feature_names = []
                for i in range(len(important_numeric)):
                    for j in range(i, len(important_numeric)):
                        if i == j:
                            feature_names.append(f"{important_numeric[i]}^2")
                        else:
                            feature_names.append(f"{important_numeric[i]}_{important_numeric[j]}")
            
            new_features_added = 0
            for i, name in enumerate(feature_names):
                clean_name = str(name).replace(' ', '_').replace('[', '').replace(']', '').replace('<', '')
                col_name = f'poly_{clean_name}'
                if col_name not in self.df.columns:  # Avoid duplicates
                    self.df[col_name] = poly_features[:, i]
                    new_features_added += 1
            
            print(f"   Added {new_features_added} polynomial interaction features")
        else:
            print("   Not enough numeric columns for polynomial features (need at least 2)")
    
    def add_binned_features(self, n_bins=5):
        """Create binned versions of continuous features."""
        print("\n🔧 Adding binned features...")
        
        # Identify numeric columns
        numeric_cols = self.df.select_dtypes(include=['int64', 'float64', 'int8']).columns.tolist()
        
        continuous_cols = []
        for col in numeric_cols:
            if col in self.df.columns and self.df[col].nunique() > 20:
                continuous_cols.append(col)
        
        # Limit to top 5
        continuous_cols = continuous_cols[:5]
        
        if continuous_cols:
            binner = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
            
            X_cont = self.df[continuous_cols].fillna(0)
            binned = binner.fit_transform(X_cont)
            
            new_features_added = 0
            for i, col in enumerate(continuous_cols):
                self.df[f'binned_{col}'] = binned[:, i].astype(int)
                new_features_added += 1
                
            print(f"   Created {new_features_added} binned features")
        else:
            print("   No continuous columns found for binning")
    
    def prepare_features(self):
        """Prepare features and target for modeling."""
        self.X = self.df.drop(columns=['target_clean'])
        self.y = self.df['target_clean']
        
        self.label_encoder = LabelEncoder()
        self.y_encoded = self.label_encoder.fit_transform(self.y)
        
        print(f"\nClasses: {self.label_encoder.classes_}")
        print(f"Target distribution (%):")
        print(self.df['target_clean'].value_counts(normalize=True) * 100)
        
        return self.X, self.y_encoded
    
    def create_preprocessor(self):
        """Create column transformer for preprocessing."""
        self.numeric_cols = self.X.select_dtypes(include=['int64', 'float64', 'int8']).columns.tolist()
        self.cat_cols = self.X.select_dtypes(include=['object']).columns.tolist()
        
        # For older scikit-learn versions, use simple OrdinalEncoder without encoded_missing_value
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numeric_cols),
                ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), self.cat_cols)
            ])
        
        return self.preprocessor
    
    def get_train_test_split(self, test_size=0.2, random_state=42):
        """Split data into train and test sets."""
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y_encoded, test_size=test_size, 
            random_state=random_state, stratify=self.y_encoded
        )
        return X_train, X_test, y_train, y_test
    
    def run_full_pipeline(self, target_col='decision.case.disposition', 
                          add_advanced_features=True):
        """Run the complete data preprocessing pipeline with advanced features."""
        self.load_data()
        self.clean_target(target_col)
        self.drop_leakage_columns()
        self.convert_boolean_columns()
        
        # Add advanced feature engineering
        if add_advanced_features:
            print("\n" + "="*40)
            print("ADDING ADVANCED FEATURES")
            print("="*40)
            
            # Add polynomial and binned features
            self.add_polynomial_features(degree=2)
            self.add_binned_features(n_bins=5)
        
        self.prepare_features()
        self.create_preprocessor()
        
        print("\n" + "="*40)
        print("✅ ADVANCED DATA PIPELINE COMPLETED!")
        print("="*40)
        print(f"Final dataset shape: {self.df.shape}")
        print(f"Number of features: {len(self.X.columns) if self.X is not None else 0}")
        
        return self