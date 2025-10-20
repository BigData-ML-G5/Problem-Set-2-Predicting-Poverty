# Problem-Set-2-Predicting-Poverty

## 🧩 Project Summary

This project applies supervised machine learning methods to predict household poverty status using Colombia’s **Gran Encuesta Integrada de Hogares (GEIH)** from DANE.  
The goal is to build models that improve upon traditional *proxy-means tests (PMT)* by capturing **non-linear interactions** and **multidimensional poverty indicators**.  
By combining interpretability and predictive performance, this pipeline aims to inform more efficient and data-driven social targeting policies.

All analyses were conducted in **R**, using multiple package for model training and validation. The workflow integrates data preprocessing, class rebalancing (SMOTE), cross-validation, and multiple model comparisons.

---

## ⚙️ Reproducibility Guide

The repository is structured for full reproducibility.  
All data processing, model training, and output generation can be replicated by running the scripts in order:

### 1. Data Download and Preparation
**Script:** `DataDownload.R`  
Downloads, unzips, and organizes the raw GEIH datasets (`train_hogares`, `test_hogares`, `train_personas`, `test_personas`).  
Performs necessary cleaning, merging, and feature engineering at the **household** level through aggregation of individual attributes.
It also handles missing values (including targeted imputations by group), scales numeric features, and constructs the final training and test sets used for modeling.

### 3. Model Training, Prediction and Validation
**Scripts:**  
- `Logit.R`  
- `CART`
- `EN.R`
- `RandomForest.R`  
- `Boosting.R`
- `Boosting_xgb.R`  

Each script runs its corresponding model using 10-fold cross-validation with SMOTE sampling.  
The metric optimized in all cases is the **F1 score**, balancing precision and recall under class imbalance.
Different methodologies were used to test different models, thus you can find different scripts of a same model, with different hyperparameters.

---

## 🧠 Models Implemented

| Model | Description | Key Hyperparameters | Optimization Metric |
|--------|--------------|---------------------|---------------------|
| **Logit** | Linear model with logit function | `Threshold` | F1-score |
| **Elastic Net** | Linear model with combined L1/L2 regularization | `alpha`, `lambda` | F1-score |
| **CART (rpart)** | Single decision tree optimized by pruning complexity (`cp`) and depth control | `cp`, `maxdepth` | F1-score |
| **Random Forest** | Ensemble of decision trees using bagging and random feature sampling | `mtry`, `ntree` | F1-score |
| **Boosting (GBM/XGBoost)** | Sequential ensemble emphasizing hard-to-classify cases | `n.trees`, `interaction.depth`, `shrinkage` | F1-score |

---

## 📊 Discussion of Results

The results section compares predictive performance across models, emphasizing:
- The trade-off between **interpretability** and **predictive strength** 
- The effect of **class rebalancing (SMOTE)** on minority class detection,
- The contribution of **non-linear modeling** to identifying poverty determinants.

Final evaluation metrics and variable importance rankings are discussed in the main report (`/Documentation/Final_Report.pdf`).

---

## 🧾 Citation

If you use or build upon this repository, please cite:

> Authors: Andrez Guerrero, Sergio Delgado, Gianluca Cicco, Adrián Suárez
> Title: *Poverty Prediction with Machine Learning: GEIH-based Analysis (2025)*  
> University of the Andes – Department of Economics and Systems Engineering  
> R version 4.5.1 | caret, randomForest, DMwR, MLmetrics


