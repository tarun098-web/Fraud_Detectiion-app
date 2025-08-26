# Fraud_Detectiion-app
.

📌 Repository Description (350 words)

This repository hosts the code, results, and interactive dashboard for the dissertation “Fairness-Aware Risk Modelling for Fraudulent Transaction Detection”, completed as part of the MSc in Artificial Intelligence and Machine Learning at the University of Limerick.

Fraudulent financial transactions continue to pose a significant challenge for banks and payment platforms, often costing billions in losses annually. While traditional fraud detection systems achieve strong accuracy, they frequently overlook fairness, leading to disproportionate misclassifications for certain demographic groups such as gender or age. This project addresses that gap by developing and evaluating a fairness-aware machine learning framework that balances high recall in fraud detection with equitable treatment across subgroups.

🔹 Key Features

Dataset: Kaggle’s large-scale fraud detection dataset (1.6M+ transactions, 27 features, ~1% fraud rate).

Data preprocessing: Feature engineering (age, distance), one-hot encoding for categorical features, outlier handling, and fairness-driven sample weighting.

Models: Random Forest, XGBoost, and Logistic Regression trained with custom class and demographic weighting.

Calibration: Integration of Platt Scaling to improve probability reliability and reduce calibration error (MACE).

Fairness metrics: Statistical Parity Difference (SPD), Demographic Parity Difference (DPD), Equal Opportunity Difference (EOD), Equalized Odds, and Mean Absolute Calibration Error (MACE) computed across gender and age groups.

Performance metrics: Precision, Recall, F1 score, and ROC–AUC used for model benchmarking.

Critical insights: Removal of SMOTE reduced overfitting, while sample weighting maintained recall ≥ 0.75 and improved fairness stability. Random Forest consistently emerged as the most effective and balanced model.

🔹 Interactive Dashboard

A Streamlit dashboard has been deployed to allow dynamic exploration of results. Users can:

Select models, performance metrics, and fairness metrics from dropdown menus.

View bar charts comparing fairness and performance across groups.

Explore calibration outcomes and subgroup disparities interactively. 
👉 [Click here to launch the app](https://frauddetectiion-app-exhfccadzdcauvkrxvlexg.streamlit.app/)

🔹 Structure

app.py: Streamlit dashboard code.

notebooks/: Jupyter notebooks for data preprocessing, modeling, evaluation.

results/: Saved plots, tables, and metrics.

references/: Supporting literature and dissertation materials.

This project demonstrates how fairness-aware techniques can be integrated into fraud detection pipelines, ensuring both accuracy and ethical responsibility in financial AI applications.
