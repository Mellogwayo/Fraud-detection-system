# Fraud-detection-system

## Overview

This repository contains code for a fraud detection system developed using Python and machine learning techniques. The system analyzes credit card transaction data to identify fraudulent transactions.

## Files

1. **creditcard.csv**: This CSV file contains the dataset used for training and testing the fraud detection models. It consists of features such as transaction amount, time, and anonymized numerical variables (V1-V28), as well as a binary target variable indicating whether the transaction is fraudulent (Class).

2. **fraud_detection_system.ipynb**: This Jupyter Notebook file contains the Python code for the fraud detection system. It includes data loading, preprocessing, exploratory data analysis, model building, evaluation, optimization, and deployment steps.

## Usage

To use the fraud detection system:

1. Clone the repository to your local machine.
2. Ensure you have Python installed along with the required libraries listed in the notebook.
3. Open and run the `fraud_detection_system.ipynb` notebook using Jupyter or any compatible environment.
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

```
pip install pandas numpy seaborn matplotlib scikit-learn joblib
```

## Acknowledgments

The dataset used in this project is sourced from Kaggle: [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud).
