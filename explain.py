import joblib, os
import pandas as pd
import numpy as np

MODEL_PATH = os.path.join('storage','churn_model.pkl')
_model = None
def load_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model

def predict_proba(features_df):
    model = load_model()
    X = features_df.drop(columns=['customer_id'], errors='ignore').fillna(0)
    proba = model.predict_proba(X)[:,1]
    return proba

def explain_instance(instance):
    model = load_model()
    cols = [c for c in instance.index if c!='customer_id']
    x = instance[cols].fillna(0).values.astype(float)
    if hasattr(model, 'coef_'):
        coefs = model.coef_[0]
        contrib = list(zip(cols, (coefs * x).tolist()))
        contrib_sorted = sorted(contrib, key=lambda t: -abs(t[1]))
        return contrib_sorted[:5]
    else:
        return [(c, 0.0) for c in cols][:5]
