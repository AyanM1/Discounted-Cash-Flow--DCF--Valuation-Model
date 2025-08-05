# data_preprocessing.py
import pandas as pd
import numpy as np

# Functions for cleaning and processing financial/credit data
# Feature engineering: debt ratios, income, payment history, etc.
def clean_data(df):
    # Example cleaning steps
    df = df.dropna()
    # ...more cleaning...
    return df

def engineer_features(df):
    # Example feature engineering
    df['debt_to_income'] = df['loan_amount'] / df['annual_income']
    # ...more features...
    return df
