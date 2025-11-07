import pandas as pd
import numpy as np

def compute_features(df, customer_col='customer_id', ts_col='timestamp'):
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col])
    features = []
    for cid, g in df.groupby(customer_col):
        g = g.sort_values(ts_col)
        total_logins = g['login_count'].sum() if 'login_count' in g else 0
        avg_session = g['session_duration'].mean() if 'session_duration' in g else 0
        tickets = g['support_tickets'].sum() if 'support_tickets' in g else 0
        last = g['login_count'].iloc[-1] if 'login_count' in g else 0
        first = g['login_count'].iloc[0] if 'login_count' in g else 0
        login_trend = (last - first) / (abs(first) + 1e-6)
        row = {
            'customer_id': cid,
            'total_logins': int(total_logins),
            'avg_session': float(avg_session if not np.isnan(avg_session) else 0.0),
            'support_tickets': int(tickets),
            'login_trend': float(login_trend)
        }
        for col in ['feature_a','feature_b']:
            if col in g:
                row[col] = int(g[col].sum())
        features.append(row)
    return pd.DataFrame(features)