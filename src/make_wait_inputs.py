"""
Kullanım:
    python make_wait_inputs.py  \
        --schedule  yarin_randevular.xlsx \
        --model     rf_pipe.joblib        \
        --slot-len  15                    \
        --out       wait_inputs.xlsx
"""

import argparse, joblib, numpy as np, pandas as pd
from datetime import datetime, timedelta

# ---------------- CLI arg ----------------
p = argparse.ArgumentParser()
p.add_argument('--schedule', default='hasta_listesi.xlsx')
p.add_argument('--model',     default='rf_pipe.joblib')
p.add_argument('--slot-len',  type=int, default=15,  # dakika
               help='Randevu slot uzunluğu (dk)')
p.add_argument('--out',       default='wait_inputs.xlsx')
args = p.parse_args()

# ---------------- 1) Dosyaları yükle ----------------
df = pd.read_excel(args.schedule)
model = joblib.load(args.model)      # Pipeline (pre + RF)

# ---------------- 2) Özellik mühendisliği ----------------
# Randevu Tarihi (datetime) -> Slot_Start (ondalık)
df['Slot_Dec'] = df['Randevu_Tarihi'].dt.hour + df['Randevu_Tarihi'].dt.minute/60
df['Gün']      = df['Randevu_Tarihi'].dt.day_name()
df['Saat']     = df['Randevu_Tarihi'].dt.hour

# ML eğitiminde kullanılan diğer sütunlar sizde zaten varsa buraya ekleyin
df['Randevuya_Gelis_Mutlak'] = 0      # varsayılan, gerçek veri yoksa
df['Randevuya_Gelis_ErkenMi'] = 0
df['Doluluk_GelisMutlak'] = df['Saatlik_Doluluk'] * df['Randevuya_Gelis_Mutlak']
df['Saat_GelisErken']     = df['Saat'] * df['Randevuya_Gelis_ErkenMi']

# ---------------- 3) Tahmin ----------------
ŷ_log = model.predict(df)
df['ŷ_wait'] = np.expm1(ŷ_log)        # dk cinsinden

# ---------------- 4) Slot × Doktor agregasyonu ----------------
slot_len_min = args.slot_len
def slot_label(dec):
    h = int(dec)
    m = int(round((dec - h)*60))
    start = datetime(2000,1,1,h,m)
    end   = start + timedelta(minutes=slot_len_min)
    return f"{start.strftime('%H:%M')}-{end.strftime('%H:%M')}"

df['Slot_Label'] = df['Slot_Dec'].apply(slot_label)

agg = (df.groupby(['Slot_Label','Doktor'])
         ['ŷ_wait'].mean()
         .reset_index()
         .rename(columns={'Doktor':'Doktor_ID',
                          'ŷ_wait':'Tahmini_Bekleme_Dk'}))

# ---------------- 5) Excel çıktısı ----------------
agg.to_excel(args.out, index=False)
print(f"✓ Tahmin tablosu yazıldı ➜ {args.out}")
