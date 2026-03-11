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
│
├── main.py
│   └── Runs the full ML pipeline
│
├── data_loader.py
│   └── Data loading and feature engineering
│
├── base_models.py
│   └── 11+ ML models and voting ensemble utilities
│
├── evaluation.py
│   └── Model evaluation, plots, confusion matrices
│
├── juris_transformer.py
│   └── Full JurisTransformer implementation
│
├── simplified_juris.py
│   └── Lightweight JurisTransformer (older environments)
│
├── requirements.txt
│
├── supreme_court.csv
│   └── Dataset
│
├── model_comparison_results.csv
│   └── Example model results
│
└── plots/
    ├── base_models_comparison.png
    ├── best_model_confusion_matrix.png
    ├── final_model_comparison.png
    └── juris_transformer_confusion_matrix.png

