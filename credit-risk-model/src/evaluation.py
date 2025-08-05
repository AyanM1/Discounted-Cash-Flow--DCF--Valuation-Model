# evaluation.py
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score

# Model validation (AUC, accuracy, confusion matrix, etc.)
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    auc = roc_auc_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return {'auc': auc, 'accuracy': acc, 'confusion_matrix': cm, 'report': report}

# Cross-validation routines
def cross_validate_model(model, X, y, cv=5):
    scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
    return scores
