# risk_metrics.py
import numpy as np

# Functions to compute default probability, credit scores, risk grades

def compute_default_probability(model, X):
    return model.predict_proba(X)[:, 1]

def compute_credit_score(prob_default):
    # Example: inverse of default probability scaled
    return (1 - prob_default) * 800

def assign_risk_grade(score):
    if score > 750:
        return 'A'
    elif score > 650:
        return 'B'
    elif score > 550:
        return 'C'
    else:
        return 'D'
