# 🌤️ JakU — Dashboard Kualitas Udara DKI Jakarta

**JakU (Jakarta Kualitas Udara)** adalah platform monitoring kualitas udara (ISPU) DKI Jakarta berbasis **machine learning**, dibangun dengan **Streamlit**. Model klasifikasi dilatih dari dataset ISPU asli (`Data_ISPU.csv`) mengikuti metodologi **CRISP-DM**, dengan **XGBoost** sebagai model utama.

**Cakupan data:** periode `202401` sampai `202511` (1 Jan 2024 sampai 30 Nov 2025), 5 stasiun SPKU DKI Jakarta, 3.480 baris bersih.

**Skema kategori:** sistem ini memakai **3 kelas** yaitu **Baik** (ISPU 0 sampai 50), **Sedang** (51 sampai 100), dan **Tidak Sehat** (di atas 100).

> **Pantau Udara, Jaga Jakarta**

🔗 **Akses aplikasi (live demo):** https://jaku-dashboard5-reinzulfarkaan.streamlit.app/

---

## ✨ Fitur

| Halaman | Isi |
|---|---|
| **Dashboard** | Ringkasan ISPU DKI Jakarta, peta interaktif 5 wilayah (Folium), prediksi & tren ISPU 7 hari, pemilih tanggal (1 Jan 2024 sampai 30 Nov 2025) |
| **Detail Wilayah** | Kondisi per kota administratif: Jakarta Pusat, Utara, Barat, Selatan, Timur, dan Kep. Seribu |
| **Simulasi Prediksi ISPU** | Input 6 polutan lewat slider + 3 preset skenario (Baik / Sedang / Tidak Baik) → prediksi kategori real-time |
| **Edukasi & Insight** | 3 kategori ISPU, dampak kesehatan, sumber polusi, dan tips menjaga kualitas udara |

Popup **"Informasi Polutan"** tersedia pada beberapa halaman untuk menjelaskan tiap parameter.

---

## 📂 Struktur Project

```
JAKU5-main/
├── app.py                       # Aplikasi Streamlit (4 halaman)
├── train_model.py               # Training 3 model dari Data_ISPU.csv (CRISP-DM)
├── generate_dashboard_data.py   # Generate data tampilan dashboard dari Data_ISPU.csv + model
├── Data_ISPU.csv                # Dataset ISPU bersih (semicolon-delimited, 3.480 baris)
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml              # Tema & konfigurasi server
├── assets/                      # SVG, PNG, ikon, ilustrasi UI
├── data/                        # Data tampilan (hasil generate dari dataset asli)
│   ├── ringkasan.csv            # Ringkasan ISPU Jakarta terkini (kartu hero)
│   ├── ringkasan.json
│   ├── wilayah.csv              # Kondisi 5 wilayah (peta + Detail Wilayah)
│   ├── tren_harian.csv          # Tren ISPU DKI 7 hari terakhir
│   ├── tren_wilayah.csv         # Tren ISPU per wilayah
│   └── prediksi.csv             # Klasifikasi kategori oleh model XGBoost
└── models/                      # Artefak model terlatih (dari notebook CRISP-DM)
    ├── model_xgboost.pkl        # ⭐ model utama
    ├── model_random_forest.pkl
    ├── model_svm.pkl
    ├── label_encoder.pkl
    ├── standard_scaler.pkl      # dipakai khusus SVM
    └── fitur_polutan.pkl        # urutan fitur polutan
```

---

## ▶️ Menjalankan Lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

Buka http://localhost:8501

**Dependensi utama:** Streamlit · Pandas · NumPy · Plotly · Folium · streamlit-folium · streamlit-option-menu · scikit-learn · XGBoost · joblib

---

## 🧠 Tentang Model

Model dilatih dari `Data_ISPU.csv` dengan pipeline identik notebook penelitian (CRISP-DM):

1. Filter kategori valid (**BAIK / SEDANG / TIDAK SEHAT**)
2. Normalisasi nama stasiun pemantau ke 5 nama kanonik
3. Konversi numerik + **imputasi median** untuk nilai kosong
4. **IQR outlier removal** pada 6 fitur polutan
5. Split `test_size=0.2, random_state=42, stratify`
6. Hyperparameter tuning dengan `StratifiedKFold` (k=5):
   - Random Forest & XGBoost → `RandomizedSearchCV`
   - SVM → `GridSearchCV` (dengan `StandardScaler`)

**Urutan fitur (wajib sama saat prediksi):**

```
pm_sepuluh, pm_duakomalima, sulfur_dioksida, karbon_monoksida, ozon, nitrogen_dioksida
```

XGBoost menjadi model utama, dengan **PM2.5** sebagai fitur paling dominan. Random Forest dan SVM disediakan sebagai pembanding.

**Performa pada data periode 202401 sampai 202511 (test set, 3 kelas):**

| Model | Akurasi |
|---|---|
| XGBoost ⭐ | 0,9639 |
| Random Forest | 0,9670 |
| SVM | 0,9451 |

Data setelah pembersihan dan IQR outlier removal: 3.184 baris (2.547 latih, 637 uji).

### Melatih ulang model

```bash
python train_model.py
```

Script membaca `Data_ISPU.csv`, melatih ketiga model, dan menyimpan 6 file `.pkl` ke `models/`.

### Memperbarui data tampilan dashboard

```bash
python generate_dashboard_data.py
```

Membangun ulang seluruh file di `data/` (ringkasan, wilayah, tren, prediksi) dari `Data_ISPU.csv` + model XGBoost.

**Urutan lengkap saat dataset diperbarui:**

```bash
python train_model.py
python generate_dashboard_data.py
```

---

## 🔄 Memperbarui Dataset

**Cukup timpa file datanya di folder project, tidak ada skrip konversi tambahan.** Dua bentuk file diterima:

| File | Keterangan |
|---|---|
| `Data_ISPU.csv` | Semicolon-delimited |
| `*.xls` / `*.xlsx` | Unduhan langsung dari open data DKI (isinya sebenarnya tabel HTML) |

Nama file bebas asal mengandung kata **ispu**. Jika ada lebih dari satu file yang cocok, yang dipakai adalah yang **paling baru diubah**, sehingga menimpa file lama langsung berlaku. Nama file yang sedang dipakai dicetak saat `train_model.py` dan `generate_dashboard_data.py` dijalankan.

Pembersihan berjalan otomatis setiap kali file dibaca oleh `app.py`, `train_model.py`, maupun `generate_dashboard_data.py`:

1. Nama stasiun dinormalisasi ke 5 nama kanonik (varian penulisan Bundaran HI dan Kebon Jeruk digabung)
2. Periode dibatasi `202401` sampai `202511`
3. Kategori dipaksa ke 3 kelas. **SANGAT TIDAK SEHAT** dipetakan ke **TIDAK SEHAT**, sedangkan **TIDAK ADA DATA** dan kategori kosong dibuang karena memang tidak ada pengukuran

Setelah menimpa file, jalankan:

```bash
python train_model.py
python generate_dashboard_data.py
```

> Kalau rentang periodenya nanti berubah, ubah konstanta `PERIODE_AWAL` dan `PERIODE_AKHIR` yang ada di bagian atas ketiga file tersebut.

---

## 🗺️ Pemetaan Stasiun → Wilayah

| Stasiun Pemantau | Wilayah |
|---|---|
| DKI1 Bunderan HI | Jakarta Pusat |
| DKI2 Kelapa Gading | Jakarta Utara |
| DKI5 Kebon Jeruk | Jakarta Barat |
| DKI3 Jagakarsa | Jakarta Selatan |
| DKI4 Lubang Buaya | Jakarta Timur |

> Kepulauan Seribu ditampilkan di Detail Wilayah namun tidak memiliki stasiun pemantau pada dataset.

---

## 🚀 Deploy ke Streamlit Community Cloud

### 1. Push ke GitHub

```bash
git init
git add .
git commit -m "Initial commit: JakU dashboard kualitas udara"
git branch -M main
git remote add origin https://github.com/USERNAME-ANDA/NAMA-REPO.git
git push -u origin main
```

> Ganti `USERNAME-ANDA` dan `NAMA-REPO`. Saat login, gunakan **Personal Access Token** GitHub sebagai password.

### 2. Deploy

1. Buka [share.streamlit.io](https://share.streamlit.io/) → login dengan GitHub
2. **Create app** → **Deploy a public app from GitHub**
3. Isi:
   - **Repository:** `USERNAME-ANDA/NAMA-REPO`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. **Deploy** — tunggu beberapa menit hingga aplikasi online di `https://<nama-app>.streamlit.app/`

> Aplikasi ini sudah ter-deploy dan dapat diakses di: **https://jaku-dashboard5-reinzulfarkaan.streamlit.app/**

### 3. Update setelah perubahan

```bash
git add .
git commit -m "deskripsi perubahan"
git push
```

Streamlit Cloud otomatis re-deploy. Jika file `.pkl` di `models/` diganti, lakukan **Reboot app** dari menu (⋮) agar cache `@st.cache_resource` ter-clear.

---

## 🛠️ Stack

Streamlit · Pandas · NumPy · Plotly · Folium · scikit-learn · XGBoost · joblib
