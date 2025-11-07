import pandas as pd
from features import compute_features
from sklearn.linear_model import LogisticRegression
import joblib
import os

def train_and_save(sample_csv='storage/sample_logs.csv', save_path='storage/churn_model.pkl'):
    df = pd.read_csv(sample_csv)
    feats = compute_features(df)
    merged = feats.copy()
    orig = df[['customer_id','churn']].drop_duplicates()
    merged = merged.merge(orig, on='customer_id', how='left')
    merged['churn'] = merged['churn'].fillna(0).astype(int)
    X = merged.drop(columns=['customer_id','churn'])
    y = merged['churn']
    if X.shape[0] < 2:
        print('Not enough data to train demo model.')
        return
    model = LogisticRegression(max_iter=1000)
    model.fit(X.fillna(0), y)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(model, save_path)
    print('Saved demo model to', save_path)

if __name__ == '__main__':
    train_and_save()
