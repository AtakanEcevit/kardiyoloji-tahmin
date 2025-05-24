###################################
# app.py
###################################
import os
import pandas as pd
import numpy as np
import joblib

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI()

# Model yükle
MODEL_PATH = "rf_pipe.joblib"
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("Model feature_in_:", model.feature_names_in_)
else:
    model = None
    print("Uyarı: Model dosyası yok! Lütfen train_model.py çalıştır.")

# Pydantic: Giriş verisi
class PatientInput(BaseModel):
    Gun: str
    Saat: int
    Doktor_ID: str
    Saatlik_Doluluk: int
    Randevu: str
    Randevuya_Gelis_Sure: float
    Randevuya_Gelis_ErkenMi: int
    Randevuya_Gelis_Mutlak: float

@app.get("/", response_class=HTMLResponse)
def serve_index():
    # index.html 'templates/' altinda
    template_path = os.path.join("templates", "index.html")
    try:
        with open(template_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    except FileNotFoundError:
        return HTMLResponse(content="<h3>index.html bulunamadı!</h3>", status_code=404)

@app.post("/tahmin")
def predict_wait_time(inp: PatientInput):
    if model is None:
        return {"detail": "Model yok. Lütfen once train_model.py ile eğitin."}

    # 1) DF
    df = pd.DataFrame([inp.dict()])
    print("\nGelen veri:\n", df)

    # 2) rename
    rename_map = {
        "Gun": "Gün",
        "Doktor_ID": "Doktor ID",
        "Randevuya_Gelis_Sure": "Randevuya Geliş Süresi (dk)",
        "Randevuya_Gelis_ErkenMi": "Randevuya_Gelis_ErkenMi",
        "Randevuya_Gelis_Mutlak": "Randevuya_Gelis_Mutlak",
    }
    df.rename(columns=rename_map, inplace=True)

    # 3) ek feature -> Doluluk_GelisMutlak, Saat_GelisErken
    df["Doluluk_GelisMutlak"] = df["Saatlik_Doluluk"] * df["Randevuya_Gelis_Mutlak"]
    df["Saat_GelisErken"] = df["Saat"] * df["Randevuya_Gelis_ErkenMi"]

    # 4) one-hot
    cat_cols = []
    for c in ["Randevu", "Gün", "Doktor ID"]:
        if c in df.columns:
            cat_cols.append(c)
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # 5) eksik columns => 0
    for col in model.feature_names_in_:
        if col not in df.columns:
            df[col] = 0
    df = df[model.feature_names_in_]
    print("Final DF:\n", df)

    # 6) predict
    y_pred_log = model.predict(df)[0]
    bekleme_tahmin = float(np.expm1(y_pred_log))  # log -> orijinal
    return {"Tahmini_Bekleme_Suresi (dk)": bekleme_tahmin}

# Local run
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
