# Fraud-detection-system

## Overview

This repository contains code for a fraud detection system developed using Python and machine learning techniques. The system analyzes credit card transaction data to identify fraudulent transactions.

## Files

1. **creditcard.csv**: This CSV file contains the dataset used for training and testing the fraud detection models. It consists of features such as transaction amount, time, and anonymized numerical variables (V1-V28), as well as a binary target variable indicating whether the transaction is fraudulent (Class).

2. **fraud_detection_system.ipynb**: This Jupyter Notebook file contains the Python code for the fraud detection system. It includes data loading, preprocessing, exploratory data analysis, model building, evaluation, optimization, and deployment steps.

3. **Fraud Detection System-Proposal.ipynb**: This Jupyter Notebook file contains the proposal for the fraud detection system.

4. **Fraud Detection System-Analysis report.ipynb**: This Jupyter Notebook file contains the analysis report for the fraud detection system.

5. **app.py**: This Python file contains the code for the web application of the fraud detection system.

6. **requirements.txt**: This file lists the dependencies required to run the code.

7. **boxplots_combined.png**: This image file contains the combined boxplots generated during the analysis.

8. **runtime.txt**: This file specifies the Python runtime version.

9. **Procfile**: This file specifies the commands that are executed by the app on startup.

10. **random_forest_model.pkl**: This file contains the trained Random Forest model.

## Usage

To use the fraud detection system:

1. Clone the repository to your local machine.
2. Ensure you have Python installed along with the required libraries listed in the notebook.
3. Open and run the `Fraud Detection System-Analysis report.ipynb` notebook using Jupyter or any compatible environment.
4. Follow the instructions in the notebook to load the dataset, preprocess the data, train and evaluate machine learning models, and deploy the system.

## Dependencies

The following Python libraries are required to run the code:

- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn
- joblib

You can install the dependencies using pip:
pip install -r requirements.txt

## Acknowledgments

The dataset used in this project is sourced from Kaggle: [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud).

