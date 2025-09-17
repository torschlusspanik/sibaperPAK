import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os
rng = np.random.default_rng(42)

def synthesize(n=3000, seed=42):
    rng = np.random.default_rng(seed)
    jenis = rng.choice(['R2','R4'], size=n, p=[0.6, 0.4])
    pkb_pokok = np.where(jenis=='R2',
                         rng.integers(100_000, 600_000, size=n),
                         rng.integers(800_000, 3_500_000, size=n))
    usia_kend = rng.integers(0, 15, size=n)
    tunggakan_tahun = rng.integers(0, 4, size=n)
    # rule-based baseline similar to your HTML
    opsen = np.round(pkb_pokok * 0.66).astype(int)
    pkb_output = (pkb_pokok + opsen) * np.maximum(1, tunggakan_tahun)
    swdkllj = np.where(jenis=='R2', 35_000, 150_000)
    pl = np.where(jenis=='R2', 15_000, 25_000)
    biaya_admin = 150_000
    total = pkb_output + swdkllj + pl + biaya_admin
    # add slight noise to emulate real-world variability
    noise = rng.normal(0, total*0.02).astype(int)
    y = np.clip(total + noise, a_min=0, a_max=None)
    df = pd.DataFrame({
        'jns_kend': jenis,
        'pkb_pokok': pkb_pokok,
        'opsen' : opsen,
        'usia_kend': usia_kend,
        'tunggakan_tahun': tunggakan_tahun,
        'target_total': y
    })
    return df

def main():
    df = synthesize(4000, seed=42)
    X = df[['jns_kend','pkb_pokok','usia_kend','tunggakan_tahun']]
    y = df['target_total']

    pre = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['jns_kend']),
        ('num','passthrough',['pkb_pokok','usia_kend','tunggakan_tahun'])
    ])
    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    pipe = Pipeline(steps=[('pre', pre), ('model', model)])

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    mae = mean_absolute_error(y_te, pred)
    r2 = r2_score(y_te, pred)

    os.makedirs('model', exist_ok=True)
    joblib.dump(pipe, 'model/sibaper_total_pipeline.joblib')

    # Save metrics
    with open('model/metrics.json','w') as f:
        import json
        json.dump({'mae': float(mae), 'r2': float(r2)}, f, indent=2)

    print('Saved model to model/sibaper_total_pipeline.joblib')
    print(f'MAE: {mae:,.0f} | R^2: {r2:.4f}')

if __name__ == '__main__':
    main()
