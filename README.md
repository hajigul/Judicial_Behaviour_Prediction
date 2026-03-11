# Judicial_Behaviour_Prediction

# Judicial ML: Supreme Court Case Outcome Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green)

**Predict Supreme Court case dispositions (affirmed, reversed, remanded, etc.) using traditional ML + a custom ensemble called JurisTransformer.**

This repository contains a **complete end-to-end machine learning pipeline** for legal judgment prediction, built with scikit-learn, pandas, and matplotlib. It includes advanced preprocessing, 11+ base models, and a novel text-aware ensemble (JurisTransformer).

---

##  Features

- **Leakage-free preprocessing** (drops future information like voting outcomes)
- **Advanced feature engineering**: polynomial interactions + binned features
- **11 traditional ML models** with proper class weighting for imbalanced targets
- **JurisTransformer** – custom ensemble using legal text features + majority voting
- **Full evaluation suite**: Accuracy, Macro F1, confusion matrices, classification reports, and beautiful plots
- **Compatible with older Python/scikit-learn** (includes simplified version)
- **Automatic plot generation** saved to `plots/` folder

---

## 📊 Results (Base Models)

**Best performing model: Random Forest**

| Model                        | Accuracy | Macro F1 | Time (sec) |
|-----------------------------|----------|----------|------------|
| Random Forest               | **0.4338** | 0.4797   | 0.8        |
| Extra Trees                 | 0.4310   | 0.4651   | 0.7        |
| XGBoost                     | 0.3856   | 0.4154   | 5.6        |
| Naive Bayes                 | 0.3595   | 0.2975   | 0.1        |
| K-Nearest Neighbors         | 0.3294   | 0.2489   | 0.4        |
| AdaBoost                    | 0.3294   | 0.2775   | 0.7        |
| Gradient Boosting           | 0.3169   | 0.3391   | 11.9       |
| Decision Tree               | 0.3169   | 0.3423   | 0.1        |
| MLP Neural Network          | 0.2589   | 0.1335   | 0.9        |
| Logistic Regression         | 0.2271   | 0.2230   | 3.6        |
| Support Vector Machine (RBF)| 0.1369   | 0.1051   | 13.6       |

**JurisTransformer** (novel ensemble) further improves performance using text features from case names, issues, and arguments.

---

## 📁 Dataset

- **File**: `supreme_court.csv`
- **Source**: Supreme Court Database (SCDB)
- **Size**: ~10,000+ cases (1946–present)
- **Target**: `decision.case.disposition` → cleaned into **7 classes**:
  - `affirmed`
  - `reversed`
  - `reversed_and_remanded`
  - `vacated_and_remanded`
  - `vacated`
  - `dismissed`
  - `petition_denied` / `other`

---

## 🛠 Project Structure

judicial-ml/
├── main.py                    # Run this → executes the full pipeline
├── data_loader.py             # Data loading + advanced feature engineering
├── base_models.py             # 11+ ML models + voting ensemble helper
├── evaluation.py              # Visualization, confusion matrices, reports
├── juris_transformer.py       # Full JurisTransformer (text + ensemble)
├── simplified_juris.py        # Lightweight version (works on older environments)
├── requirements.txt
├── supreme_court.csv          # The dataset
├── model_comparison_results.csv  # Example base model results
└── plots/                     # ← Generated charts (created automatically)
├── base_models_comparison.png
├── best_model_confusion_matrix.png
├── final_model_comparison.png
├── juris_transformer_confusion_matrix.png




##  Quick Start

### 1. Clone repository
```bash
git clone https://github.com/YOUR_USERNAME/judicial-ml.git
cd judicial-ml



2. Create & activate virtual environment (recommended)
Bashpython -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
3. Install dependencies
Bashpip install -r requirements.txt
Note: transformers is optional. The simplified JurisTransformer works without it.
4. Run the full pipeline
Bashpython main.py
This will:

Load and preprocess data
Train & evaluate all base models
Run JurisTransformer (3-fold CV)
Generate comparison plots & save results

 Configuration
Change the dataset path in main.py if necessary:
PythonDATA_PATH = "supreme_court.csv"   # ← adjust if needed
 What You Get After Running

Console: detailed metrics + classification report
plots/ folder containing:
Model comparison bar charts (Accuracy & Macro F1)
Confusion matrix (best model + JurisTransformer)
Feature importance plot
Final comparison table (CSV)


 Technical Highlights

Class imbalance → handled via class_weight='balanced'
No data leakage → voting & disposition columns removed
Text feature engineering → legal term counts + basic statistics
Ensemble voting inside JurisTransformer
Compatible with scikit-learn versions before 1.2

 Possible Next Steps

Integrate BERT / legal language models
Hyperparameter optimization (Optuna)
Time-based train/test split (chronological)
SHAP / LIME explanations for legal interpretability
Citation network → Graph Neural Networks

 Contributing
Feel free to:

Report bugs / suggest improvements (Issues)
Submit pull requests (new models, better features, documentation)

 License
MIT License — see the LICENSE file for details.
 Acknowledgments

Supreme Court Database (SCDB)
scikit-learn, XGBoost, pandas & visualization communities
Legal AI & judicial prediction research community


