"""
Main script for judicial ML project.
Runs complete pipeline from data loading to model evaluation.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sys
import os
warnings.filterwarnings('ignore')

# Import custom modules
from data_loader import JudicialDataLoader
from base_models import BaseModelTrainer
from evaluation import ModelEvaluator

# Import JurisTransformer
try:
    from juris_transformer import run_juris_transformer
    JURIS_AVAILABLE = True
except ImportError as e:
    JURIS_AVAILABLE = False
    print(f"⚠️ JurisTransformer not available: {e}")

# Create plots directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

def save_base_model_plots(results_df, evaluator, y_test, y_pred, model_name, 
                          y_test_best=None, y_pred_best=None):
    """Save all plots for base models"""
    
    # Plot model comparison
    plt.figure(figsize=(14, 7))
    sns.barplot(data=results_df.sort_values(by='Accuracy', ascending=False), 
               x='Accuracy', y='Model', palette='viridis')
    plt.title('Base Models Comparison - Accuracy')
    plt.xlabel('Accuracy')
    plt.tight_layout()
    plt.savefig('plots/base_models_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot F1 comparison
    plt.figure(figsize=(14, 7))
    sns.barplot(data=results_df.sort_values(by='Macro F1', ascending=False), 
               x='Macro F1', y='Model', palette='magma')
    plt.title('Base Models Comparison - Macro F1 Score')
    plt.xlabel('Macro F1 Score')
    plt.tight_layout()
    plt.savefig('plots/base_models_f1_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot confusion matrix for best model
    if y_test_best is not None and y_pred_best is not None:
        plt.figure(figsize=(10, 8))
        cm = pd.crosstab(y_test_best, y_pred_best, 
                         rownames=['Actual'], colnames=['Predicted'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - Best Model ({model_name})')
        plt.tight_layout()
        plt.savefig('plots/best_model_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print("\n✅ Base model plots saved to 'plots/' directory")

def main():
    """Main execution function."""
    
    # Configuration
    DATA_PATH = r"C:\Users\Haji\Documents\Judicial_adnan_amin\supreme_court.csv"
    TARGET_COL = 'decision.case.disposition'
    
    print("="*60)
    print("JUDICIAL MACHINE LEARNING PROJECT")
    print("="*60)
    
    # Step 1: Load and preprocess data
    print("\n📊 STEP 1: DATA LOADING & PREPROCESSING")
    print("-"*40)
    
    try:
        loader = JudicialDataLoader(DATA_PATH)
        loader.run_full_pipeline(target_col=TARGET_COL)
        
        # Get train-test split
        X_train, X_test, y_train, y_test = loader.get_train_test_split()
        print(f"\n✅ Train set: {X_train.shape}, Test set: {X_test.shape}")
    except Exception as e:
        print(f"❌ Error in data loading: {e}")
        return
    
    # Step 2: Train base models
    print("\n🤖 STEP 2: TRAINING BASE MODELS")
    print("-"*40)
    
    try:
        trainer = BaseModelTrainer(loader.preprocessor)
        results_df = trainer.train_all_models(X_train, X_test, y_train, y_test)
    except Exception as e:
        print(f"❌ Error in model training: {e}")
        return
    
    # Step 3: Evaluate base models
    print("\n📈 STEP 3: BASE MODEL EVALUATION")
    print("-"*40)
    
    try:
        evaluator = ModelEvaluator(label_encoder=loader.label_encoder)
        evaluator.results_df = results_df
        
        # Display results
        print("\n📋 Base Model Comparison:")
        print(results_df.to_string(index=False))
        
        # Get best model for detailed evaluation
        best_model_name, best_model = trainer.get_best_model()
        if best_model_name:
            from sklearn.pipeline import Pipeline
            best_pipe = Pipeline([
                ('preprocessor', loader.preprocessor),
                ('classifier', best_model)
            ])
            best_pipe.fit(X_train, y_train)
            y_pred_best = best_pipe.predict(X_test)
            
            # Save base model plots
            save_base_model_plots(results_df, evaluator, y_test, y_pred_best, 
                                 best_model_name, y_test, y_pred_best)
            
            # Print best model report
            print(f"\n🏆 Best Base Model: {best_model_name}")
            evaluator.print_classification_report(y_test, y_pred_best)
            
    except Exception as e:
        print(f"❌ Error in base model evaluation: {e}")
    
    # Step 4: Run JurisTransformer (if available)
    if JURIS_AVAILABLE:
        print("\n" + "="*60)
        print("🤖 STEP 4: RUNNING JURISTRANSFORMER (NOVEL HYBRID MODEL)")
        print("="*60)
        
        try:
            # Define text columns (adjust based on your dataset)
            text_columns = ['name', 'issue.text', 'lower court.disposition', 
                          'lower court.reasons', 'arguments.petitioner.entity', 
                          'arguments.respondent.entity']
            
            # Run JurisTransformer
            juris_results = run_juris_transformer(
                loader=loader,
                text_columns=text_columns,
                n_folds=3,  # Use 3 folds for faster training
                save_plots_flag=True
            )
            
            # Add JurisTransformer results to comparison
            juris_row = pd.DataFrame([{
                'Model': 'JurisTransformer (Novel)',
                'Accuracy': round(juris_results['accuracy'], 4),
                'Macro F1': round(juris_results['f1_score'], 4),
                'Time (sec)': 'N/A'
            }])
            
            final_results = pd.concat([results_df, juris_row], ignore_index=True)
            
            print("\n" + "="*60)
            print("📊 FINAL COMPARISON: ALL MODELS")
            print("="*60)
            print(final_results.sort_values(by='Accuracy', ascending=False).to_string(index=False))
            
            # Plot final comparison
            plt.figure(figsize=(14, 8))
            plot_df = final_results.copy()
            plot_df['Time (sec)'] = pd.to_numeric(plot_df['Time (sec)'], errors='coerce')
            
            # Create grouped bar chart
            x = np.arange(len(plot_df))
            width = 0.35
            
            fig, ax = plt.subplots(figsize=(14, 8))
            bars1 = ax.bar(x - width/2, plot_df['Accuracy'], width, label='Accuracy', color='steelblue')
            bars2 = ax.bar(x + width/2, plot_df['Macro F1'], width, label='Macro F1', color='coral')
            
            ax.set_xlabel('Models')
            ax.set_ylabel('Score')
            ax.set_title('Model Comparison: Accuracy vs Macro F1 Score')
            ax.set_xticks(x)
            ax.set_xticklabels(plot_df['Model'], rotation=45, ha='right')
            ax.legend()
            
            # Add value labels on bars
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(f'{height:.3f}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3), textcoords="offset points",
                               ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            plt.savefig('plots/final_model_comparison.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # Save final results to CSV
            final_results.to_csv('plots/final_model_comparison.csv', index=False)
            print("\n✅ Final results saved to 'plots/final_model_comparison.csv'")
            
            # Print improvement
            best_base_acc = results_df['Accuracy'].max()
            improvement = ((juris_results['accuracy'] - best_base_acc) / best_base_acc) * 100
            print(f"\n📈 JurisTransformer Improvement over best base model: {improvement:+.2f}%")
            
        except Exception as e:
            print(f"❌ Error running JurisTransformer: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n⚠️ JurisTransformer not available. Install required packages to run it.")
    
    print("\n✅ Project completed successfully!")

if __name__ == "__main__":
    # Set pandas display options
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)
    pd.set_option('display.max_colwidth', 100)
    
    # Run main pipeline
    main()