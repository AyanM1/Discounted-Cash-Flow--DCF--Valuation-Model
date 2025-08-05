# credit_model.py
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# Functions to train models: logistic regression, random forest, XGBoost
# Use SMOTE for imbalanced classes

def train_logistic_regression(X, y):
    model = LogisticRegression()
    model.fit(X, y)
    return model

def train_random_forest(X, y):
    model = RandomForestClassifier()
    model.fit(X, y)
    return model

def train_xgboost(X, y):
    model = XGBClassifier()
    model.fit(X, y)
    return model

def balance_data(X, y):
    smote = SMOTE()
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res
