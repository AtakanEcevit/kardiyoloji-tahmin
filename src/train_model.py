###################################
# train_model.py
###################################
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def train_model(
    data_path="../data/Kardiyoloji.xlsx",
    model_path="../models/best_wait_time_model.joblib"
):
    """
    Kardiyoloji bekleme süresi verisini okur, feature engineering yapar,
    outlier temizler, log transform uygular, 
    GridSearchCV ile en iyi RandomForest modelini bulur
    ve modeli kaydeder.
    """

    # 1) Veri Oku
    df = pd.read_excel(data_path)
    print("Veri sekli:", df.shape)

    # 2) Bekleme Süresi
    df["BeklemeSuresi"] = df["İşlem Öncesi Bekleme (dk)"]

    # Tarihsel eklemeler
    df["Gün"] = pd.to_datetime(df["Randevu Tarihi"], errors="coerce").dt.day_name()
    df["Saat"] = pd.to_datetime(df["Randevu Tarihi"], errors="coerce").dt.hour

    # Randevuya Geliş
    df["Randevuya_Gelis_ErkenMi"] = (df["Randevuya Geliş Süresi (dk)"] < 0).astype(int)
    df["Randevuya_Gelis_Mutlak"] = df["Randevuya Geliş Süresi (dk)"].abs()

    # Outlier => üst %95
    upper_lim = df["BeklemeSuresi"].quantile(0.95)
    df = df[df["BeklemeSuresi"] <= upper_lim]

    # Log Transform
    df["Bekleme_Log"] = np.log1p(df["BeklemeSuresi"])

    # Extra Feature
    df["Doluluk_GelisMutlak"] = df["Saatlik_Doluluk"] * df["Randevuya_Gelis_Mutlak"]
    df["Saat_GelisErken"] = df["Saat"] * df["Randevuya_Gelis_ErkenMi"]

    # Seçilen Feature'lar
    selected_features = [
        "Saatlik_Doluluk",
        "Randevuya_Gelis_Mutlak",
        "Randevuya_Gelis_ErkenMi",
        "Doluluk_GelisMutlak",
        "Saat_GelisErken",
        "Randevu",
        "Gün",
        "Saat",
        "Doktor ID"
    ]
    target_col = "Bekleme_Log"

    use_cols = selected_features + [target_col]
    df_model = df[use_cols].dropna()
    df_model["Doktor ID"] = df_model["Doktor ID"].astype(str)

    print("Model veri seti sekli:", df_model.shape)

    # Kategorik
    cat_cols = []
    for c in ["Randevu", "Gün", "Doktor ID"]:
        if c in df_model.columns:
            cat_cols.append(c)

    df_encoded = pd.get_dummies(df_model, columns=cat_cols, drop_first=True)

    # X,y
    X = df_encoded.drop(target_col, axis=1)
    y = df_encoded[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Train:", X_train.shape, "Test:", X_test.shape)

    # Model + GridSearch
    rf = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }

    grid = GridSearchCV(
        rf,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    print("Best params:", grid.best_params_)

    # Test
    y_pred_log = best_model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_test_orig = np.expm1(y_test)

    mae = mean_absolute_error(y_test_orig, y_pred)
    mse = mean_squared_error(y_test_orig, y_pred)
    rmse = mse**0.5
    r2 = r2_score(y_test_orig, y_pred)

    print("\n--- Test (Log->Orijinal) ---")
    print("MAE:", mae)
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("R2:", r2)

    # Kaydet
    joblib.dump(best_model, model_path)
    print("Model kaydedildi =>", model_path)

if __name__ == "__main__":
    train_model()
