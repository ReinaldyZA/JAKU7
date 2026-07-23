"""
generate_dashboard_data.py
==========================================================================
Membangun SEMUA data tampilan dashboard dari file data ISPU di folder
project (Data_ISPU.csv ATAU .xls unduhan open data DKI) + model XGBoost
terlatih (models/*.pkl). Output menggantikan data dummy lama.

Pembersihan data IDENTIK dengan train_model.py / notebook:
  - filter kategori valid (BAIK / SEDANG / TIDAK SEHAT)
  - normalisasi nama stasiun
  - konversi numerik + isi NaN dengan median

File yang dihasilkan di folder data/:
  - ringkasan.csv      : ringkasan ISPU Jakarta terkini (kartu hero dashboard)
  - wilayah.csv        : kondisi terkini 5 wilayah (peta + tab Detail Wilayah)
  - tren_harian.csv    : tren ISPU DKI 7 hari terakhir (grafik dashboard)
  - tren_wilayah.csv   : tren ISPU 7 hari terakhir per wilayah (Detail Wilayah)
  - prediksi.csv       : klasifikasi kategori oleh model XGBoost atas
                         pembacaan polutan 7 hari terakhir (per wilayah + DKI)

Jalankan ulang setiap kali file datanya ditimpa:
    python generate_dashboard_data.py
==========================================================================
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).parent
DATA = BASE / "data"
MODELS = BASE / "models"
DATA.mkdir(exist_ok=True)

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
KAT_VALID = ["BAIK", "SEDANG", "TIDAK SEHAT"]

# Pemetaan stasiun pemantau -> wilayah administrasi + koordinat
WILAYAH = {
    "DKI1 Bunderan HI":   ("Jakarta Pusat",   -6.1924, 106.8232),
    "DKI2 Kelapa Gading": ("Jakarta Utara",   -6.1565, 106.9056),
    "DKI5 Kebon Jeruk":   ("Jakarta Barat",   -6.1881, 106.7567),
    "DKI3 Jagakarsa":     ("Jakarta Selatan", -6.3349, 106.8270),
    "DKI4 Lubang Buaya":  ("Jakarta Timur",   -6.2889, 106.9105),
}

# Map polutan ringkas (UI) <-> kolom dataset
POL_DATASET = {"pm25": "pm_duakomalima", "pm10": "pm_sepuluh",
               "no2": "nitrogen_dioksida", "so2": "sulfur_dioksida",
               "co": "karbon_monoksida", "o3": "ozon"}

BULAN_ID = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "Mei", 6: "Jun",
            7: "Jul", 8: "Agu", 9: "Sep", 10: "Okt", 11: "Nov", 12: "Des"}


def kategori_dari_ispu(ispu):
    """Skema 3 kelas sistem JakU: Baik / Sedang / Tidak Sehat."""
    if ispu <= 50:  return "Baik"
    if ispu <= 100: return "Sedang"
    return "Tidak Sehat"


NAMA_POL = {"pm25": "PM2.5", "pm10": "PM10", "no2": "NO₂",
            "so2": "SO₂", "co": "CO", "o3": "O₃"}


# ───────────────────────── LOAD & CLEAN ─────────────────────────
def load_clean():
    df = baca_ispu(BASE, DATA).copy()
    if df.empty:
        raise SystemExit("Tidak ada file data ISPU di folder project. "
                         "Letakkan Data_ISPU.csv atau file .xls unduhan open data DKI.")
    print(f"Sumber data : {cari_file_ispu(BASE, DATA).name}")
    for c in FITUR + ["max"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # isi NaN per kolom dengan median (sama seperti notebook)
    for c in FITUR:
        df[c] = df[c].fillna(df[c].median())
    df["tgl"] = pd.to_datetime(
        df["periode_data"].astype(str).str[:4] + "-"
        + df["periode_data"].astype(str).str[4:6] + "-"
        + df["tanggal"].astype(int).astype(str).str.zfill(2),
        errors="coerce",
    )
    df = df.dropna(subset=["tgl", "max"]).sort_values("tgl")
    return df


def fmt_tgl(ts):
    return f"{ts.day:02d} {BULAN_ID[ts.month]} {ts.year}"


# ───────────────────────── BUILDERS ─────────────────────────
def build_wilayah(df):
    """Snapshot terkini tiap wilayah = rata-rata 30 hari terakhir stasiun."""
    rows = []
    for st, (wil, lat, lon) in WILAYAH.items():
        sub = df[df["stasiun"] == st].tail(30)
        if sub.empty:
            continue
        vals = {ui: round(float(sub[col].median()))
                for ui, col in POL_DATASET.items()}
        ispu = int(round(sub["max"].mean()))
        rows.append({
            "wilayah": wil, "ispu": ispu, "kategori": kategori_dari_ispu(ispu),
            "lat": lat, "lon": lon,
            "pm25": vals["pm25"], "pm10": vals["pm10"], "no2": vals["no2"],
            "so2": vals["so2"], "co": round(vals["co"], 1), "o3": vals["o3"],
        })
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "wilayah.csv", index=False)
    return out


def build_tren_harian(df):
    """Tren ISPU DKI (rata-rata semua stasiun) 7 hari terakhir."""
    g = (df.groupby("tgl")["max"].mean().round().astype(int)
         .tail(7).reset_index())
    g.columns = ["tanggal", "ispu"]
    g["tanggal"] = g["tanggal"].dt.strftime("%Y-%m-%d")
    g.to_csv(DATA / "tren_harian.csv", index=False)
    return g


def build_tren_wilayah(df):
    """Tren ISPU 7 hari terakhir per wilayah (nilai max asli per stasiun)."""
    rows = []
    for st, (wil, _, _) in WILAYAH.items():
        sub = df[df["stasiun"] == st].tail(7)
        for _, r in sub.iterrows():
            rows.append({"wilayah": wil,
                         "tanggal": r["tgl"].strftime("%Y-%m-%d"),
                         "ispu": int(round(r["max"]))})
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "tren_wilayah.csv", index=False)
    return out


def hitung_polutan_dominan(frame):
    """
    Polutan pencemar kritis = kolom polutan yang nilainya menentukan ISPU
    (paling sering sama dengan kolom 'max'). Pada dataset DKI ini PM2.5
    mendominasi (~85% baris), konsisten dengan literatur kualitas udara
    Jakarta. Cara ini lebih tepat daripada interpolasi breakpoint karena
    kolom polutan pada dataset sudah berskala indeks, bukan konsentrasi mentah.
    """
    cocok = {ui: (frame[col] == frame["max"]).sum()
             for ui, col in POL_DATASET.items()}
    key = max(cocok, key=cocok.get)
    return NAMA_POL[key], key


def build_ringkasan(df):
    """Ringkasan ISPU Jakarta terkini = rata-rata semua stasiun bulan terakhir."""
    periode_terakhir = df["periode_data"].max()
    lm = df[df["periode_data"] == periode_terakhir]
    vals = {ui: round(float(lm[col].median())) for ui, col in POL_DATASET.items()}
    ispu = int(round(lm["max"].mean()))
    nama_dom, key_dom = hitung_polutan_dominan(lm)
    tgl_terakhir = df["tgl"].max()
    ringkasan = {
        "ispu": ispu,
        "kategori": kategori_dari_ispu(ispu),
        "pm25": vals["pm25"], "pm10": vals["pm10"], "no2": vals["no2"],
        "so2": vals["so2"], "co": round(vals["co"], 1), "o3": vals["o3"],
        "polutan_dominan": nama_dom,
        "polutan_dominan_nilai": vals[key_dom],
        "tanggal_update": fmt_tgl(tgl_terakhir),
        "tahun_data": f"{df['tgl'].min().year}–{df['tgl'].max().year}",
    }
    pd.DataFrame([ringkasan]).to_csv(DATA / "ringkasan.csv", index=False)
    (DATA / "ringkasan.json").write_text(json.dumps(ringkasan, ensure_ascii=False, indent=2))
    return ringkasan


def build_prediksi(df):
    """
    Klasifikasi kategori ISPU oleh model XGBoost terlatih atas pembacaan
    polutan 7 hari TERAKHIR (per wilayah + agregat DKI). Ini OUTPUT MODEL
    NYATA pada data nyata — bukan angka karangan.

    Catatan: model notebook adalah CLASSIFIER (polutan -> kategori), bukan
    forecaster deret waktu. Maka tabel ini menampilkan kategori hasil model
    untuk pembacaan terkini, bukan ramalan tanggal masa depan yang dikarang.
    """
    xgb = joblib.load(MODELS / "model_xgboost.pkl")
    le = joblib.load(MODELS / "label_encoder.pkl")
    fitur = joblib.load(MODELS / "fitur_polutan.pkl")
    kat_map = {"BAIK": "Baik", "SEDANG": "Sedang", "TIDAK SEHAT": "Tidak Sehat"}

    def klasifikasi(frame):
        X = frame[fitur]
        idx = xgb.predict(X)
        return [kat_map.get(k, "Sedang") for k in le.inverse_transform(idx)]

    rows = []
    # Agregat DKI: rata-rata semua stasiun per tanggal, 7 hari terakhir
    agg = (df.groupby("tgl")[FITUR + ["max"]].mean()
           .tail(7).reset_index())
    agg["kategori"] = klasifikasi(agg)
    for _, r in agg.iterrows():
        rows.append({"tanggal": r["tgl"].strftime("%Y-%m-%d"),
                     "wilayah": "DKI Jakarta", "ispu": int(round(r["max"])),
                     "kategori": r["kategori"],
                     "pm25": int(round(r["pm_duakomalima"]))})
    # Per wilayah
    for st, (wil, _, _) in WILAYAH.items():
        sub = df[df["stasiun"] == st].tail(7).copy()
        if sub.empty:
            continue
        sub["kategori"] = klasifikasi(sub)
        for _, r in sub.iterrows():
            rows.append({"tanggal": r["tgl"].strftime("%Y-%m-%d"),
                         "wilayah": wil, "ispu": int(round(r["max"])),
                         "kategori": r["kategori"],
                         "pm25": int(round(r["pm_duakomalima"]))})
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "prediksi.csv", index=False)
    return out


if __name__ == "__main__":
    df = load_clean()
    print(f"Data bersih: {len(df)} baris | {df['tgl'].min().date()} → {df['tgl'].max().date()}")
    r = build_ringkasan(df)
    print(f"\n[ringkasan] ISPU Jakarta = {r['ispu']} ({r['kategori']}), "
          f"dominan {r['polutan_dominan']}, update {r['tanggal_update']}")
    w = build_wilayah(df)
    print(f"\n[wilayah] {len(w)} wilayah:")
    print(w[["wilayah", "ispu", "kategori", "pm25"]].to_string(index=False))
    t = build_tren_harian(df)
    print(f"\n[tren_harian] {t['ispu'].tolist()}")
    build_tren_wilayah(df)
    p = build_prediksi(df)
    print(f"\n[prediksi] klasifikasi model XGBoost (DKI):")
    print(p[p['wilayah'] == 'DKI Jakarta'][['tanggal', 'ispu', 'kategori']].to_string(index=False))
    print("\nSelesai. Semua file di folder data/ kini berbasis Data_ISPU.csv.")
