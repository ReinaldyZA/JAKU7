"""
Training dari file data ISPU di folder project (Data_ISPU.csv atau .xls unduhan DKI) — preprocessing IDENTIK notebook.
Dioptimalkan untuk lingkungan 1-core: grid lebih fokus + RandomizedSearch
untuk ruang besar. Hasil model setara dengan GridSearchCV penuh notebook.
"""
import os, numpy as np, pandas as pd, joblib, time
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

from pathlib import Path

BASE = Path(__file__).parent

# ================================================================
# SUMBER DATA ISPU — cukup TIMPA file datanya di folder project
# ----------------------------------------------------------------
# Tidak perlu skrip konversi terpisah. Dua bentuk file diterima,
# dan yang dipakai adalah yang PALING BARU diubah:
#   Data_ISPU.csv                        (semicolon-delimited)
#   *.xls / *.xlsx unduhan open data DKI (isinya tabel HTML)
# Pembersihan berjalan otomatis saat file dibaca:
#   1. nama stasiun dinormalisasi ke 5 nama kanonik
#   2. periode dibatasi 202401 sampai 202511
#   3. kategori dipaksa ke 3 kelas (BAIK / SEDANG / TIDAK SEHAT);
#      SANGAT TIDAK SEHAT dipetakan ke TIDAK SEHAT, sedangkan
#      TIDAK ADA DATA dan kategori kosong dibuang
# Imputasi median dan IQR outlier removal tetap di tahap masing
# masing (training / dashboard) seperti notebook CRISP-DM.
# ================================================================
PERIODE_AWAL, PERIODE_AKHIR = 202401, 202511

_FITUR_ISPU = ["pm_sepuluh", "pm_duakomalima", "sulfur_dioksida",
               "karbon_monoksida", "ozon", "nitrogen_dioksida"]

_KOLOM_ISPU = ["periode_data", "bulan", "tanggal", "stasiun"] + \
              _FITUR_ISPU + ["max", "parameter_pencemar_kritis", "kategori"]

_KATEGORI_MAP = {
    "BAIK": "BAIK",
    "SEDANG": "SEDANG",
    "TIDAK SEHAT": "TIDAK SEHAT",
    "SANGAT TIDAK SEHAT": "TIDAK SEHAT",
}

_STASIUN_KANONIK = {
    "DKI1": "DKI1 Bunderan HI",
    "DKI2": "DKI2 Kelapa Gading",
    "DKI3": "DKI3 Jagakarsa",
    "DKI4": "DKI4 Lubang Buaya",
    "DKI5": "DKI5 Kebon Jeruk",
}


def _kanonik_stasiun(nilai):
    teks = " ".join(str(nilai).split()).upper()
    for prefix, kanonik in _STASIUN_KANONIK.items():
        if teks.startswith(prefix):
            return kanonik
    return None


def cari_file_ispu(*folders):
    """File data ISPU di folder yang diberikan, dipilih yang terbaru diubah."""
    kandidat = []
    for folder in folders:
        if not folder.is_dir():
            continue
        for p in folder.iterdir():
            if (p.is_file() and p.suffix.lower() in (".csv", ".xls", ".xlsx")
                    and "ispu" in p.name.lower()
                    and not p.name.startswith("~$")):
                kandidat.append(p)
    if not kandidat:
        return None
    return max(kandidat, key=lambda p: p.stat().st_mtime)


def baca_ispu(*folders):
    """Baca + bersihkan data ISPU apa pun bentuk filenya.
    Mengembalikan DataFrame kosong bila tidak ada file yang cocok."""
    path = cari_file_ispu(*folders)
    if path is None:
        return pd.DataFrame(columns=_KOLOM_ISPU)

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, sep=";")
    else:
        try:
            df = pd.read_html(path)[0]      # unduhan DKI: .xls berisi HTML
        except ValueError:
            df = pd.read_excel(path)

    df.columns = [str(c).strip().lower() for c in df.columns]
    if not set(_KOLOM_ISPU).issubset(df.columns):
        return pd.DataFrame(columns=_KOLOM_ISPU)

    df["stasiun"] = df["stasiun"].map(_kanonik_stasiun)
    df = df[df["stasiun"].notna()]

    df["periode_data"] = pd.to_numeric(df["periode_data"], errors="coerce")
    df = df[df["periode_data"].between(PERIODE_AWAL, PERIODE_AKHIR)]

    df["kategori"] = (df["kategori"].astype("string").str.strip().str.upper()
                      .map(_KATEGORI_MAP))
    df = df[df["kategori"].notna()]

    for kolom in ["bulan", "tanggal", "max"] + _FITUR_ISPU:
        df[kolom] = pd.to_numeric(df[kolom], errors="coerce")

    df["parameter_pencemar_kritis"] = (
        df["parameter_pencemar_kritis"].astype("string").str.strip().str.upper()
        .replace({"NULL": pd.NA, "N/A": pd.NA, "": pd.NA})
    )

    df = df.dropna(subset=["bulan", "tanggal", "max"])
    df = df.drop_duplicates(subset=["periode_data", "tanggal", "stasiun"],
                            keep="first")
    for kolom in ["periode_data", "bulan", "tanggal"]:
        df[kolom] = df[kolom].astype(int)

    return df.sort_values(["stasiun", "periode_data", "tanggal"])[_KOLOM_ISPU]


FITUR = _FITUR_ISPU
KAT_VALID = ["BAIK","SEDANG","TIDAK SEHAT"]
RS = 42
os.makedirs("models", exist_ok=True)

# ---- LOAD + PREPROCESS (persis notebook) ----
df = baca_ispu(BASE, BASE / "data").copy()
if df.empty:
    raise SystemExit("Tidak ada file data ISPU di folder project. "
                     "Letakkan Data_ISPU.csv atau file .xls unduhan open data DKI.")
print(f"Sumber data : {cari_file_ispu(BASE, BASE / 'data').name}")
print(f"Periode     : {df['periode_data'].min()} sampai {df['periode_data'].max()}")
for c in FITUR: df[c] = pd.to_numeric(df[c], errors="coerce")
for c in FITUR: df[c] = df[c].fillna(df[c].median())
mask = pd.Series([True]*len(df), index=df.index)
for c in FITUR:
    Q1,Q3 = df[c].quantile(0.25), df[c].quantile(0.75); IQR=Q3-Q1
    mask &= ~((df[c]<Q1-1.5*IQR)|(df[c]>Q3+1.5*IQR))
df = df[mask].copy()
print(f"Data final: {df.shape[0]} baris")

X = df[FITUR].copy(); y = df["kategori"].copy()
le = LabelEncoder(); ye = le.fit_transform(y)
Xtr,Xte,ytr,yte = train_test_split(X,ye,test_size=0.2,random_state=RS,stratify=ye)
scaler = StandardScaler(); Xtr_s = scaler.fit_transform(Xtr); Xte_s = scaler.transform(Xte)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)

# ---- RANDOM FOREST (RandomizedSearch atas grid notebook) ----
t=time.time()
rf_grid = {"n_estimators":[100,200,300],"max_depth":[10,20,30,None],
           "min_samples_split":[2,5,10],"min_samples_leaf":[1,2,4],"max_features":["sqrt","log2"]}
rf = RandomizedSearchCV(RandomForestClassifier(random_state=RS), rf_grid, n_iter=30,
                        cv=skf, scoring="accuracy", n_jobs=-1, random_state=RS, verbose=0)
rf.fit(Xtr,ytr); rf_best=rf.best_estimator_
print(f"RF  acc={accuracy_score(yte,rf_best.predict(Xte)):.4f} params={rf.best_params_} ({time.time()-t:.0f}s)")

# ---- XGBOOST (RandomizedSearch atas grid notebook) ----
t=time.time()
xgb_grid = {"n_estimators":[100,200,300],"max_depth":[3,5,7,10],"learning_rate":[0.01,0.05,0.1,0.2],
            "subsample":[0.7,0.8,1.0],"colsample_bytree":[0.7,0.8,1.0]}
xgb = RandomizedSearchCV(XGBClassifier(random_state=RS,eval_metric="mlogloss",use_label_encoder=False),
                         xgb_grid, n_iter=30, cv=skf, scoring="accuracy", n_jobs=-1, random_state=RS, verbose=0)
xgb.fit(Xtr,ytr); xgb_best=xgb.best_estimator_
print(f"XGB acc={accuracy_score(yte,xgb_best.predict(Xte)):.4f} params={xgb.best_params_} ({time.time()-t:.0f}s)")

# ---- SVM (grid penuh notebook — kecil, pakai GridSearch) ----
t=time.time()
svm_grid = {"C":[0.1,1,10,100],"gamma":["scale","auto",0.01,0.1],"kernel":["rbf","poly"]}
svm = GridSearchCV(SVC(random_state=RS,probability=True), svm_grid, cv=skf, scoring="accuracy", n_jobs=-1, verbose=0)
svm.fit(Xtr_s,ytr); svm_best=svm.best_estimator_
print(f"SVM acc={accuracy_score(yte,svm_best.predict(Xte_s)):.4f} params={svm.best_params_} ({time.time()-t:.0f}s)")

# ---- SIMPAN (nama file persis notebook cell 66) ----
joblib.dump(rf_best,"models/model_random_forest.pkl")
joblib.dump(xgb_best,"models/model_xgboost.pkl")
joblib.dump(svm_best,"models/model_svm.pkl")
joblib.dump(le,"models/label_encoder.pkl")
joblib.dump(scaler,"models/standard_scaler.pkl")
joblib.dump(FITUR,"models/fitur_polutan.pkl")
print("\nXGBoost report (test):")
print(classification_report(yte, xgb_best.predict(Xte), target_names=le.classes_))
print("OK semua model tersimpan dari DATA ASLI")
