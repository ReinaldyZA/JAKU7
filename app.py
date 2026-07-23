"""
================================================================
JakU - Dashboard Kualitas Udara DKI Jakarta
================================================================
Aplikasi Streamlit untuk monitoring kualitas udara DKI Jakarta
dengan integrasi model machine learning XGBoost.

Halaman:
    1. Dashboard          - Ringkasan kualitas udara provinsi
    2. Detail Wilayah     - Informasi per kota administratif
    3. Simulasi Prediksi  - Prediksi ISPU dari 6 polutan
    4. Edukasi & Insight  - Pengetahuan ISPU, dampak, dan tips
"""

import os
import io
import re
import math
import hashlib
import base64
from pathlib import Path
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium.plugins import Fullscreen
from streamlit_folium import st_folium
import streamlit.components.v1 as components
from branca.element import Figure
from streamlit_option_menu import option_menu
import joblib

# ================================================================
# KONFIGURASI HALAMAN
# ================================================================
st.set_page_config(
    page_title="JakU - Dashboard Kualitas Udara",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ================================================================
# KONSTANTA
# ================================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"

# ----------------------------------------------------------------
# LEBAR KONTEN TETAP (kunci tampilan agar konsisten di semua zoom)
# Konten dikunci ke lebar px tetap & dibuat rata-tengah, sehingga
# layout di zoom 50%-100% identik (tidak meregang/menyempit).
#
# Cara setel: angka ini sebaiknya <= lebar konten yang tersedia saat
# zoom 100% di monitormu (kalau kebesaran -> muncul scroll horizontal
# di zoom 100%). 1366 aman untuk mayoritas laptop/monitor. Naikkan
# (mis. 1500) kalau monitor besar & ingin tampilan selebar zoom 80%.
# ----------------------------------------------------------------
CONTENT_MAX_WIDTH = 1366

def _leaf_icon_svg() -> str:
    """Baca assets/leaf.svg dan kembalikan inline SVG berukuran kecil (~16px),
    dijadikan SATU BARIS agar tidak memicu salah-render Markdown Streamlit
    (newline + indentasi bisa membuat sisa HTML dianggap blok kode)."""
    try:
        svg = (ASSETS_DIR / "leaf.svg").read_text(encoding="utf-8")
    except OSError:
        return ""
    svg = re.sub(r"\n\s*", "", svg).strip()  # satukan jadi satu baris
    return re.sub(
        r"<svg\b",
        "<svg style='width:16px;height:auto;vertical-align:middle;flex-shrink:0;'",
        svg, count=1,
    )


LEAF_ICON_SVG = _leaf_icon_svg()

# Mapping kategori ISPU -> warna, emoji, deskripsi
KATEGORI_INFO = {
    "Baik": {
        "warna": "#16A34A", "warna_bg": "#DCFCE7", "emoji": "😊",
        "rentang": "0-50",
        "deskripsi": "Udara bersih, aman untuk beraktivitas sehari-hari.",
        "rekom_emoji": "🌿",
        "rekomendasi": "Cocok untuk olahraga, jalan kaki, dan aktivitas outdoor lainnya. Nikmati udara segar dan tetap jaga pola hidup sehat."
    },
    "Sedang": {
        "warna": "#4A6CF7", "warna_bg": "#DBEAFE", "emoji": "😐",
        "rentang": "51-100",
        "deskripsi": "Masih dapat diterima untuk beraktivitas di luar ruangan.",
        "rekom_emoji": "🚶",
        "rekomendasi": "Cocok untuk olahraga ringan dan aktivitas harian. Gunakan masker jika sensitif terhadap polusi dan hindari paparan terlalu lama."
    },
    "Tidak Sehat": {
        "warna": "#E5B93D", "warna_bg": "#FEF3C7", "emoji": "😷",
        "rentang": "101-200",
        "deskripsi": "Kurangi aktivitas luar ruangan, terutama bagi kelompok sensitif.",
        "rekom_emoji": "⚠️",
        "rekomendasi": "Kurangi aktivitas luar ruangan dalam waktu lama. Disarankan menggunakan masker terutama bagi anak-anak, lansia, dan penderita gangguan pernapasan."
    },
    "Sangat Tidak Sehat": {
        "warna": "#EF4444", "warna_bg": "#FEE2E2", "emoji": "🤢",
        "rentang": "201-300",
        "deskripsi": "Hindari aktivitas luar ruangan. Gunakan masker jika harus keluar.",
        "rekom_emoji": "😷",
        "rekomendasi": "Hindari aktivitas outdoor jika tidak mendesak. Tetap berada di dalam ruangan dan gunakan masker saat harus keluar rumah."
    },
    "Berbahaya": {
        "warna": "#7C3AED", "warna_bg": "#EDE9FE", "emoji": "☠️",
        "rentang": ">301",
        "deskripsi": "Hindari semua aktivitas luar ruangan. Tetap di dalam ruangan.",
        "rekom_emoji": "🚨",
        "rekomendasi": "Tetap berada di dalam ruangan dan hindari seluruh aktivitas luar. Tutup ventilasi udara dan gunakan pelindung pernapasan jika harus keluar."
    },
}

# Gambar "Rekomendasi Aktivitas" (hasil desain Figma) per kategori ISPU.
# File PNG transparan disimpan di folder assets/.
REKOM_IMG = {
    "Baik":               "rekom_baik.png",
    "Sedang":             "rekom_sedang.png",
    "Tidak Sehat":        "rekom_tidak_sehat.png",
    "Sangat Tidak Sehat": "rekom_sangat_tidak_sehat.png",
    "Berbahaya":          "rekom_berbahaya.png",
}

# Informasi 6 polutan untuk popup
INFO_POLUTAN = {
    "PM2.5": {
        "warna": "#2563EB",
        "satuan": "µg/m³",
        "deskripsi_pendek": "Partikel sangat halus berukuran ≤ 2.5 mikron",
        "deskripsi": "Partikel sangat halus yang dapat masuk jauh ke dalam paru-paru dan aliran darah."
    },
    "PM10": {
        "warna": "#60A5FA",
        "satuan": "µg/m³",
        "deskripsi_pendek": "Partikel halus berukuran ≤ 10 mikron",
        "deskripsi": "Partikel halus yang dapat masuk ke saluran pernapasan bagian atas dan menyebabkan iritasi."
    },
    "NO₂": {
        "warna": "#8B5CF6",
        "satuan": "µg/m³",
        "deskripsi_pendek": "Nitrogen dioksida, gas hasil pembakaran",
        "deskripsi": "Gas hasil pembakaran kendaraan bermotor dan industri, dapat mengiritasi paru-paru."
    },
    "SO₂": {
        "warna": "#F59E0B",
        "satuan": "µg/m³",
        "deskripsi_pendek": "Sulfur dioksida, gas dari pembakaran bahan bakar fosil",
        "deskripsi": "Gas dari pembakaran bahan bakar fosil, dapat menyebabkan iritasi mata dan saluran pernapasan."
    },
    "CO": {
        "warna": "#10B981",
        "satuan": "mg/m³",
        "deskripsi_pendek": "Karbon monoksida, gas tidak berwarna dan tidak berbau",
        "deskripsi": "Gas tidak berwarna dan tidak berbau yang dapat mengganggu pasokan oksigen dalam tubuh."
    },
    "O₃": {
        "warna": "#06B6D4",
        "satuan": "µg/m³",
        "deskripsi_pendek": "Ozon, terbentuk dari reaksi kimia di atmosfer",
        "deskripsi": "Ozon terbentuk dari reaksi kimia polutan dengan sinar matahari, dapat menyebabkan sesak napas."
    },
}


# ================================================================
# CUSTOM CSS
# ================================================================
def inject_css():
    st.markdown("""
    <style>
    /* Import font modern */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"], .stApp, .main, .block-container {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Background utama */
    .stApp {
        background-color: #FFFFFF;
    }

    /* Hilangkan top padding default.
       (max-width konten diatur terpisah lewat inject_layout_lock()
        agar layout konsisten di semua level zoom) */
    .block-container {
        padding-top: 24px !important;
        padding-bottom: 48px !important;
    }

    /* Hilangkan header & footer Streamlit */
    header[data-testid="stHeader"] {
        background: transparent;
        height: 0;
    }
    #MainMenu, footer {visibility: hidden;}

    /* FIX TAMBAHAN — hilangkan SEMUA chrome Streamlit yang masih muncul
       (toolbar Share/star/edit/GitHub di kanan atas + "Manage app" di kanan bawah) */
    [data-testid="stToolbar"],
    [data-testid="stActionButton"],
    [data-testid="stStatusWidget"],
    [data-testid="stDecoration"],
    .stDeployButton,
    .stAppDeployButton,
    button[kind="header"],
    button[kind="headerNoPadding"],
    div[class*="viewerBadge"],
    iframe[title="streamlit_app"] {
        display: none !important;
        visibility: hidden !important;
    }
    /* Toolbar wrapper kosong tetap memakan tinggi → set 0 */
    .stApp > header { height: 0 !important; }

    /* ============ SIDEBAR ============ */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
        padding-top: 0 !important;
        min-width: 300px;
    }
    /* Header bawaan sidebar (area tombol collapse) menyisakan ruang kosong
       di atas — nol-kan supaya logo benar-benar mentok ke atas. */
    [data-testid="stSidebarHeader"],
    div[data-testid="stSidebarHeader"] {
        padding: 0 !important;
        height: 0 !important;
        min-height: 0 !important;
    }
    [data-testid="stSidebarUserContent"],
    [data-testid="stSidebarContent"] {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }
    /* Streamlit membungkus isi sidebar dalam beberapa div bersarang —
       semua layer harus di-reset ke 0 supaya logo benar-benar mentok atas */
    [data-testid="stSidebar"] > div:first-child,
    [data-testid="stSidebar"] > div > div:first-child,
    [data-testid="stSidebar"] > div > div > div:first-child,
    section[data-testid="stSidebar"] > div,
    section[data-testid="stSidebar"] > div > div {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0 !important;
        padding-top: 0 !important;
    }
    [data-testid="stSidebar"] .element-container {
        margin-bottom: 0 !important;
    }
    [data-testid="stSidebar"] .stMarkdown {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }

    .sidebar-logo {
        text-align: center;
        padding: 8px 16px 4px 16px;
    }
    .sidebar-subtitle {
        text-align: center;
        font-size: 12px;
        color: #64748B;
        font-weight: 500;
        margin-bottom: 24px;
        letter-spacing: 0.02em;
    }

    .sidebar-footer {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 14px 16px;
        margin: 16px 8px;
    }
    .sidebar-footer-title {
        font-size: 14px;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 6px;
    }
    .sidebar-footer-desc {
        font-size: 12px;
        color: #64748B;
        line-height: 1.45;
        margin-bottom: 10px;
    }
    .sidebar-footer-ts-label {
        font-size: 11px;
        color: #94A3B8;
        margin-bottom: 2px;
    }
    .sidebar-footer-ts {
        font-size: 12px;
        font-weight: 700;
        color: #0F172A;
    }

    /* ============ HEADER HALAMAN ============ */
    .page-title {
        font-size: 26px;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 4px;
        letter-spacing: -0.01em;
    }
    .page-subtitle {
        font-size: 15px;
        color: #64748B;
        margin-bottom: 24px;
    }

    .updated-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 14px 20px;
        display: inline-block;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }
    .updated-card-label {
        font-size: 12px;
        color: #64748B;
        margin-bottom: 2px;
    }
    .updated-card-value {
        font-size: 15px;
        font-weight: 700;
        color: #0F172A;
    }

    /* ============ CARD UMUM ============ */
    .card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
        transition: all 0.25s ease;
        height: 100%;
    }
    .card:hover {
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08);
        transform: translateY(-1px);
    }
    .card-title {
        font-size: 16px;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 8px;
    }

    /* ============ CARD via st.container(border=True) — FIX #3 ============
       Pattern lama (st.markdown("<div class='card'>") ... </div>) bocor
       karena tiap st.markdown jadi DOM container terpisah. Solusi: pakai
       st.container(border=True) native + style border wrapper-nya. */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 16px !important;
        border: 1px solid #E5E7EB !important;
        background-color: #FFFFFF !important;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
        padding: 20px 22px !important;
        transition: all 0.25s ease;
    }
    [data-testid="stVerticalBlockBorderWrapper"]:hover {
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
    }

    /* ============ MODAL "Informasi Polutan" (st.dialog) ============
       Default width="large" membuat kotak modal kelebaran/penuh layar dan
       top-aligned. Di-override: compact (max 720px), center vertikal &
       horizontal, overlay gelap transparan, shadow halus, close button
       minimalis. */
    /* Overlay gelap transparan + blur ringan, dan center vertikal */
    [data-testid="stDialog"] {
        background: rgba(0, 0, 0, 0.35) !important;
        backdrop-filter: blur(2px) !important;
        align-items: center !important;
    }
    [data-testid="stDialog"] > div {
        align-items: center !important;
        padding-top: 2vh !important;
        padding-bottom: 2vh !important;
    }
    /* Kotak modal: compact (bukan full-screen), rounded, padding rapi */
    [data-testid="stDialog"] div[role="dialog"] {
        width: min(720px, 94vw) !important;
        max-width: 720px !important;
        border-radius: 20px !important;
        padding: 22px 26px 26px !important;
        box-shadow: 0 24px 60px rgba(15, 23, 42, 0.22) !important;
        border: none !important;
    }
    /* Judul "Informasi Polutan" — bold, ringkas, tidak nabrak tombol close */
    [data-testid="stDialog"] div[role="dialog"] h1,
    [data-testid="stDialog"] div[role="dialog"] h2,
    [data-testid="stDialog"] div[role="dialog"] h3 {
        font-size: 22px !important;
        font-weight: 700 !important;
        color: #0F172A !important;
        padding-right: 34px !important;
        line-height: 1.25 !important;
    }
    /* Tombol close: ikon X minimalis tanpa kotak outline, hover halus */
    [data-testid="stDialog"] div[role="dialog"] button[aria-label="Close"] {
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
        border-radius: 8px !important;
        transition: background 0.15s ease !important;
    }
    [data-testid="stDialog"] div[role="dialog"] button[aria-label="Close"]:hover {
        background: #F1F5F9 !important;
    }

    /* ============ DATE PICKER "Pilih Tanggal" ============
       Dibuat lebih besar & menonjol supaya mudah di-notice user
       (dipakai di Dashboard & Detail Wilayah). */
    [data-testid="stDateInput"] input {
        font-size: 16px !important;
        font-weight: 600 !important;
        color: #0F172A !important;
        padding-top: 11px !important;
        padding-bottom: 11px !important;
    }
    /* SATU outline saja: border + shadow hanya di kotak terluar baseweb. */
    [data-testid="stDateInput"] [data-baseweb="input"] {
        border-radius: 12px !important;
        border: 1.5px solid #C7D2FE !important;
        background: #FFFFFF !important;
        box-shadow: 0 2px 10px rgba(37, 99, 235, 0.08) !important;
        transition: border-color 0.18s ease, box-shadow 0.18s ease !important;
    }
    /* Elemen dalam (base-input & wrappernya) -> TANPA border/shadow/bg
       supaya tidak terjadi outline bertumpuk (double outline). */
    [data-testid="stDateInput"] [data-baseweb="base-input"],
    [data-testid="stDateInput"] [data-baseweb="input"] > div {
        border: none !important;
        box-shadow: none !important;
        background: transparent !important;
    }
    [data-testid="stDateInput"] [data-baseweb="input"]:hover {
        border-color: #4A6CF7 !important;
        box-shadow: 0 4px 16px rgba(37, 99, 235, 0.16) !important;
    }
    [data-testid="stDateInput"] [data-baseweb="input"]:focus-within {
        border-color: #4A6CF7 !important;
        box-shadow: 0 0 0 3px rgba(74, 108, 247, 0.18) !important;
    }
    /* Ikon kalender bawaan baseweb -> biru aksen */
    [data-testid="stDateInput"] svg {
        fill: #4A6CF7 !important;
        width: 20px !important;
        height: 20px !important;
    }

    /* Map container — rounded corners untuk iframe folium.
       st_folium pakai title khusus; components.html (Figure) pakai srcdoc.
       Keduanya distyle supaya sudut membulat & ada border halus. */
    iframe[title="streamlit_folium.st_folium"],
    iframe[srcdoc] {
        border-radius: 16px;
        border: 1px solid #EEF2F7;
    }

    /* ============ ISPU BESAR ============ */
    .ispu-hero {
        display: flex;
        align-items: center;
        gap: 24px;
    }
    .ispu-number {
        font-size: 72px;
        font-weight: 800;
        line-height: 1;
        color: #2563EB;
        letter-spacing: -0.04em;
    }
    .ispu-label {
        font-size: 15px;
        font-weight: 600;
        color: #64748B;
        margin-top: 4px;
    }
    .ispu-status {
        font-size: 24px;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .ispu-desc {
        font-size: 14px;
        color: #475569;
        line-height: 1.5;
        max-width: 384px;
    }
    .ispu-emoji {
        font-size: 48px;
        margin-bottom: 8px;
    }

    /* ============ POLUTAN DOMINAN ============ */
    .polutan-dominan-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
        margin-top: 24px;
        padding-top: 16px;
        border-top: 1px solid #F1F5F9;
    }
    .polutan-dominan-text {
        font-size: 14px;
        color: #0F172A;
    }
    .polutan-dominan-icon {
        color: #16A34A;
    }

    /* ============ METRIC POLUTAN ROW ============ */
    .pollutant-grid {
        display: grid;
        grid-template-columns: repeat(6, 1fr);
        gap: 16px;
        margin-top: 16px;
    }
    .pollutant-cell {
        text-align: center;
    }
    .pollutant-name {
        font-size: 13px;
        font-weight: 600;
        color: #64748B;
        margin-bottom: 4px;
    }
    .pollutant-value {
        font-size: 27px;
        font-weight: 800;
        color: #0F172A;
        line-height: 1.1;
    }
    .pollutant-unit {
        font-size: 11px;
        color: #94A3B8;
        margin-top: 2px;
    }

    /* ============ PREDIKSI LIST ============ */
    .pred-row {
        display: grid;
        /* FIX — kolom fixed-width supaya semua baris align presisi: tanggal | badge | kategori | µg/m³
           sebelumnya pakai fr-ratio → spasi tidak konsisten antar baris */
        grid-template-columns: 105px 70px 1fr 155px;
        align-items: center;
        gap: 12px;
        padding: 10px 0;
        border-bottom: 1px solid #F1F5F9;
    }
    .pred-row:last-child { border-bottom: none; }
    .pred-date {
        font-size: 14px;
        color: #334155;
        font-weight: 500;
        white-space: nowrap;
    }
    .pred-pill {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 8px;
        font-size: 14px;
        font-weight: 700;
        color: #FFFFFF;
        text-align: center;
        min-width: 48px;
    }
    .pred-cat {
        font-size: 14px;
        font-weight: 600;
        white-space: nowrap;
    }
    .pred-pm {
        font-size: 13px;
        color: #64748B;
        text-align: right;
        white-space: nowrap;
    }

    /* ============ REKOMENDASI CARD ============ */
    .rekom-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 18px 19px;
        display: flex;
        gap: 14px;
        align-items: flex-start;
        transition: all 0.25s ease;
        height: 100%;
    }
    .rekom-card:hover {
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08);
        transform: translateY(-1px);
    }
    .rekom-icon {
        font-size: 32px;
        flex-shrink: 0;
        line-height: 1;
    }
    .rekom-title {
        font-size: 15px;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 3px;
    }
    .rekom-desc {
        font-size: 12px;
        color: #64748B;
        line-height: 1.45;
    }

    /* ============ INFO BOX (ML) ============ */
    .info-box {
        background-color: #EFF6FF;
        border: 1px solid #DBEAFE;
        border-radius: 12px;
        padding: 14px 18px;
        display: flex;
        gap: 10px;
        align-items: flex-start;
        margin-top: 16px;
    }
    .info-box-icon { color: #2563EB; font-size: 18px; line-height: 1.4; flex-shrink: 0;}
    .info-box-text {
        font-size: 14px;
        color: #1E40AF;
        line-height: 1.5;
    }

    /* ============ STEP BAR (Simulasi) ============ */
    .step-bar {
        background: #EFF6FF;
        border: 1px solid #DBEAFE;
        border-radius: 16px;
        padding: 24px 30px;
        display: grid;
        grid-template-columns: auto repeat(3, 1fr);
        gap: 32px;
        align-items: center;
        margin-bottom: 24px;
    }
    .step-title {
        display: flex;
        align-items: center;
        gap: 9px;
        font-weight: 700;
        color: #2563EB;
        font-size: 17px;
    }
    .step-item {
        display: flex;
        gap: 12px;
        align-items: flex-start;
    }
    .step-num {
        background: #FFFFFF;
        border: 1px solid #DBEAFE;
        color: #2563EB;
        width: 31px;
        height: 31px;
        border-radius: 999px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 15px;
        flex-shrink: 0;
    }
    .step-text {
        font-size: 15px;
        color: #1E40AF;
        line-height: 1.55;
    }

    /* ============ HASIL PREDIKSI ============ */
    .hasil-hero {
        display: flex;
        align-items: flex-start;
        gap: 24px;
        margin-bottom: 24px;
    }
    .hasil-num {
        font-size: 64px;
        font-weight: 800;
        line-height: 1;
        color: #2563EB;
        letter-spacing: -0.04em;
    }
    .hasil-label-ispu {
        font-size: 15px;
        color: #64748B;
        font-weight: 600;
        text-align: center;
    }
    .rekom-box {
        background-color: #EFF6FF;
        border: 1px solid #DBEAFE;
        border-radius: 14px;
        padding: 18px 21px;
    }
    .rekom-box-title {
        font-size: 16px;
        font-weight: 700;
        color: #2563EB;
        margin-bottom: 6px;
    }
    .rekom-box-text {
        font-size: 14px;
        color: #1E40AF;
        line-height: 1.5;
    }

    /* ============ SIM PAGE — MODERN CARD SYSTEM ============ */
    /* Wrapper card untuk Komposisi Polutan & Hasil Prediksi */
    .sim-card {
        background: #FFFFFF;
        border-radius: 20px;
        padding: 26px 28px;
        border: 1px solid #E5E7EB;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04),
                    0 4px 14px -4px rgba(15, 23, 42, 0.06);
        margin-bottom: 16px;
    }
    .sim-card-header {
        display: flex;
        align-items: flex-start;
        gap: 14px;
        margin-bottom: 20px;
        padding-bottom: 16px;
        border-bottom: 1px solid #F1F5F9;
    }
    .sim-card-icon {
        width: 40px; height: 40px;
        border-radius: 12px;
        background: linear-gradient(135deg, #DBEAFE 0%, #BFDBFE 100%);
        color: #2563EB;
        display: flex; align-items: center; justify-content: center;
        font-size: 19px; flex-shrink: 0;
    }
    .sim-card-icon.icon-result {
        background: linear-gradient(135deg, #DCFCE7 0%, #BBF7D0 100%);
        color: #16A34A;
    }
    .sim-card-title {
        font-size: 17px; font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.015em;
        line-height: 1.3;
    }
    .sim-card-desc {
        font-size: 13px; color: #64748B;
        line-height: 1.5;
        margin-top: 2px;
    }
    .sim-section-label {
        font-size: 12px; font-weight: 700;
        color: #475569;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 9px;
        display: flex; align-items: center; gap: 6px;
    }
     .sim-section-label {
        font-size: 12px; font-weight: 700;
        color: #475569;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 9px;
    }

    /* ============ PRESET PILLS (warna sesuai kategori ISPU) ============ */
    /* Pakai marker div sebagai sibling supaya bisa target tombol Streamlit
       lewat CSS combinator (Streamlit tidak expose class langsung di button). */
    /* ============ PRESET PILLS — :has() selector ============ */
    /* Karena Streamlit nesting marker dalam stMarkdown 4 level dalam, kita
       harus naik ke stElementContainer (level 4) lalu cari sibling-nya yang
       memuat button. Modern :has() bekerja di Chrome/Edge/Safari ≥2022,
       Firefox ≥2023 — aman untuk Streamlit Community Cloud users. */
    /* Marker hanya penanda CSS (tak terlihat) → container-nya dikolapskan
       supaya tidak menambah celah/baris kosong di atas tombol. Elemen tetap
       ada di DOM sehingga selektor sibling (+) di bawah tetap bekerja. */
    [data-testid="stElementContainer"]:has(.pmkr) {
        display: none !important;
    }
    [data-testid="stElementContainer"]:has(.pmkr) + [data-testid="stElementContainer"] button[kind="secondary"],
    [data-testid="stElementContainer"]:has(.pmkr) + [data-testid="stElementContainer"] [data-testid="stBaseButton-secondary"] {
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 9px 11px !important;
        font-size: 12px !important;
        transition: all 0.2s ease-in-out !important;
        border: 1.5px solid #E5E7EB !important;
        background: #FFFFFF !important;
        color: #475569 !important;
        min-height: 38px !important;
    }
    /* Idle hover — outline pakai warna kategori */
    [data-testid="stElementContainer"]:has(.pmkr-baik) + [data-testid="stElementContainer"] button[kind="secondary"]:hover {
        border-color: #16A34A !important; background: #F0FDF4 !important; color: #15803D !important;
    }
    [data-testid="stElementContainer"]:has(.pmkr-sedang) + [data-testid="stElementContainer"] button[kind="secondary"]:hover {
        border-color: #EAB308 !important; background: #FEFCE8 !important; color: #A16207 !important;
    }
    [data-testid="stElementContainer"]:has(.pmkr-tdksehat) + [data-testid="stElementContainer"] button[kind="secondary"]:hover {
        border-color: #EA580C !important; background: #FFF7ED !important; color: #C2410C !important;
    }
    /* Active — fill gradient sesuai kategori */
    [data-testid="stElementContainer"]:has(.pmkr-baik.active) + [data-testid="stElementContainer"] button[kind="secondary"] {
        background: linear-gradient(135deg, #16A34A 0%, #15803D 100%) !important;
        border-color: #15803D !important; color: #FFFFFF !important;
        box-shadow: 0 4px 12px -2px rgba(22,163,74,0.4) !important;
        font-weight: 700 !important; transform: translateY(-1px);
    }
    [data-testid="stElementContainer"]:has(.pmkr-sedang.active) + [data-testid="stElementContainer"] button[kind="secondary"] {
        background: linear-gradient(135deg, #EAB308 0%, #CA8A04 100%) !important;
        border-color: #CA8A04 !important; color: #FFFFFF !important;
        box-shadow: 0 4px 12px -2px rgba(234,179,8,0.4) !important;
        font-weight: 700 !important; transform: translateY(-1px);
    }
    [data-testid="stElementContainer"]:has(.pmkr-tdksehat.active) + [data-testid="stElementContainer"] button[kind="secondary"] {
        background: linear-gradient(135deg, #EA580C 0%, #C2410C 100%) !important;
        border-color: #C2410C !important; color: #FFFFFF !important;
        box-shadow: 0 4px 12px -2px rgba(234,88,12,0.4) !important;
        font-weight: 700 !important; transform: translateY(-1px);
    }

    /* ============ RESET BUTTON — :has() selector ============ */
    [data-testid="stElementContainer"]:has(.reset-marker) + [data-testid="stElementContainer"] button[kind="secondary"] {
        background: #FFFFFF !important;
        border: 1.5px solid #FCA5A5 !important;
        color: #B91C1C !important;
        font-weight: 600 !important;
        border-radius: 12px !important;
        padding: 9px 19px !important;
        font-size: 14px !important;
        transition: all 0.2s ease-in-out !important;
    }
    [data-testid="stElementContainer"]:has(.reset-marker) + [data-testid="stElementContainer"] button[kind="secondary"]:hover {
        background: #FEF2F2 !important;
        border-color: #EF4444 !important;
        color: #991B1B !important;
        box-shadow: 0 4px 12px -2px rgba(239, 68, 68, 0.25) !important;
        transform: translateY(-1px);
    }

    /* ============ SLIDER MINI-CARDS ============ */
    .slider-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 11px 15px 9px;
        margin-bottom: 2px;
        transition: all 0.2s ease-in-out;
    }
    .slider-card:hover {
        border-color: #CBD5E1;
        background: #FFFFFF;
        box-shadow: 0 2px 8px -2px rgba(15, 23, 42, 0.08);
    }
    .slider-card-head {
        display: flex; justify-content: space-between; align-items: center;
        gap: 8px;
        margin-bottom: 3px;
    }
    .slider-card-label {
        display: flex; align-items: center; gap: 8px;
        font-weight: 700; color: #0F172A; font-size: 15px;
    }
    .slider-card-dot {
        width: 10px; height: 10px; border-radius: 999px;
        flex-shrink: 0;
    }
    .slider-card-value {
        font-weight: 700; font-variant-numeric: tabular-nums;
        color: #0F172A; font-size: 15px;
        white-space: nowrap;
    }
    .slider-card-unit {
        font-size: 11px; color: #94A3B8;
        font-weight: 500; margin-left: 3px;
    }
    .slider-card-desc {
        font-size: 12px; color: #64748B;
        line-height: 1.4;
    }
    /* Group spacing antar item polutan */
    .polutan-block {
        margin-bottom: 18px;
    }

    /* ============ HERO RESULT ============ */
    .hero-result {
        background: #FFFFFF;
        border-radius: 16px; padding: 24px 20px 21px;
        border: 1px solid #F1F5F9;
        text-align: center;
        margin-bottom: 16px;
        position: relative;
    }
    .hero-status-pill {
        display: inline-flex; align-items: center; gap: 8px;
        padding: 6px 15px;
        border-radius: 999px;
        font-size: 13px; font-weight: 700;
        letter-spacing: 0.01em;
        margin-bottom: 14px;
    }
    .hero-emoji-inline { font-size: 17px; line-height: 1; }
    .hero-result-num {
        font-size: 67px;
        font-weight: 800;
        line-height: 0.95;
        margin: 3px 0 2px;
        letter-spacing: -0.04em;
        font-variant-numeric: tabular-nums;
    }
    .hero-result-label {
        font-size: 12px; color: #94A3B8;
        text-transform: uppercase; letter-spacing: 0.1em;
        font-weight: 700;
        margin-bottom: 10px;
    }
    .hero-result-desc {
        color: #475569;
        font-size: 13px;
        line-height: 1.55;
        margin-top: 11px;
        padding: 0 5px;
    }

    /* ============ REKOMENDASI MODERN BOX ============ */
    .rekom-modern {
        border-radius: 14px;
        padding: 15px 17px;
        display: flex; gap: 11px; align-items: flex-start;
        margin-top: 14px;
        border: 1px solid;
    }
    .rekom-modern-icon {
        width: 27px; height: 27px; border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
        font-size: 15px; font-weight: 700;
        flex-shrink: 0;
    }
    .rekom-modern-title {
        font-size: 12px; font-weight: 700;
        letter-spacing: 0.02em;
        margin-bottom: 3px;
    }
    .rekom-modern-text {
        font-size: 13px; color: #334155;
        line-height: 1.55;
    }

    /* ============ SUB-INDEKS PROGRESS BARS ============ */
    .subindex-section {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 16px 18px 14px;
        margin-top: 16px;
    }
    .subindex-section-title {
        font-size: 12px; font-weight: 700;
        color: #0F172A;
        margin-bottom: 11px;
        display: flex; justify-content: space-between; align-items: center;
    }
    .subindex-section-hint {
        font-size: 11px; color: #94A3B8;
        font-weight: 500; letter-spacing: 0.03em;
    }
    .subindex-bar-card {
        background: #FFFFFF;
        border-radius: 10px;
        padding: 9px 12px 8px;
        margin-bottom: 7px;
        border: 1px solid #F1F5F9;
        transition: all 0.2s ease-in-out;
    }
    .subindex-bar-card.dominan {
        border-color: #FBBF24;
        background: linear-gradient(135deg, #FFFBEB 0%, #FEF3C7 100%);
        box-shadow: 0 2px 6px -2px rgba(245, 158, 11, 0.25);
    }
    .subindex-bar-head {
        display: flex; justify-content: space-between; align-items: center;
        margin-bottom: 6px;
        font-size: 12px;
    }
    .subindex-bar-name {
        font-weight: 700; color: #0F172A;
        display: flex; align-items: center; gap: 6px;
    }
    .subindex-bar-val {
        font-variant-numeric: tabular-nums; color: #0F172A;
        font-weight: 700; font-size: 14px;
    }
    .subindex-bar-track {
        background: #F1F5F9; border-radius: 999px;
        height: 6px;
        overflow: hidden;
        margin-bottom: 6px;
    }
    .subindex-bar-fill {
        height: 100%; border-radius: 999px;
        transition: width 0.4s ease-out;
    }
    .subindex-bar-foot {
        display: flex; justify-content: flex-start;
        align-items: center; gap: 6px;
    }

    /* Pill kategori (Baik/Sedang/Tidak Sehat/dst) */
    .kat-pill {
        font-size: 10px;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 999px;
        border: 1px solid;
        white-space: nowrap;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    /* Badge polutan dominan */
    .dom-badge {
        font-size: 9px;
        font-weight: 800;
        background: linear-gradient(135deg, #F59E0B, #D97706);
        color: #FFFFFF;
        padding: 2px 6px;
        border-radius: 4px;
        letter-spacing: 0.05em;
        box-shadow: 0 1px 3px rgba(245, 158, 11, 0.4);
    }

    /* Badge preset aktif di header */
    .active-preset-badge {
        display: inline-flex; align-items: center; gap: 6px;
        background: #EFF6FF; color: #1D4ED8;
        font-size: 12px; font-weight: 600;
        padding: 5px 12px; border-radius: 999px;
        border: 1px solid #BFDBFE;
        margin-bottom: 13px;
    }
    .active-preset-badge .dot {
        width: 8px; height: 8px; border-radius: 50%;
        background: #2563EB;
        box-shadow: 0 0 0 3px rgba(37,99,235,0.18);
    }

    /* Fade-in halus tiap kali hasil di-recompute */
    @keyframes sim-fade-in {
        from { opacity: 0; transform: translateY(4px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .sim-fade { animation: sim-fade-in 0.25s ease-out; }

    /* Responsive — mobile: stack sliders, kurangi padding */
    @media (max-width: 768px) {
        .sim-card { padding: 18px; border-radius: 16px; }
        .hero-result-num { font-size: 51px; }
        .slider-card { padding: 11px 14px; }
    }

    /* ============ TABS WILAYAH ============ */
    /* Streamlit lama memakai BaseWeb ([data-baseweb="tab"]), Streamlit >= 1.5x
       memakai react-aria ([data-testid="stTab"], [data-selected]). Kedua
       selektor ditulis agar tampilan pill tetap muncul di versi mana pun. */
    .stTabs [data-baseweb="tab-list"],
    .stTabs [role="tablist"] {
        gap: 10px !important;
        border-bottom: none;
        flex-wrap: wrap;
        overflow: visible !important;
    }
    /* Garis abu bawaan di bawah baris tab (pseudo ::after milik tab-list) */
    .stTabs [role="tablist"]::after {
        background-color: transparent !important;
    }
    /* Garis bawah biru bawaan pada tab aktif */
    .stTabs .react-aria-SelectionIndicator { display: none !important; }
    .stTabs [data-baseweb="tab-highlight"] { display: none; }
    .stTabs [data-baseweb="tab-border"] { display: none; }

    .stTabs [data-baseweb="tab"],
    .stTabs [data-testid="stTab"] {
        display: inline-flex !important;
        align-items: center;
        height: auto !important;
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 8px !important;
        padding: 10px 20px !important;
        font-weight: 600;
        color: #64748B !important;
        font-size: 14px;
    }
    /* Ikon status (penunjuk) di kiri label tiap wilayah — sesuai desain Figma.
       inline-block + vertical-align agar ukuran 16x16 selalu berlaku
       (elemen inline biasa mengabaikan width/height). Gambar per-wilayah
       disuntik dinamis di page_detail_wilayah(). */
    .stTabs [data-baseweb="tab"]::before,
    .stTabs [data-testid="stTab"]::before {
        content: "";
        display: inline-block;
        width: 16px;
        height: 16px;
        margin-right: 8px;
        vertical-align: middle;
        flex-shrink: 0;
        background-repeat: no-repeat;
        background-position: center;
        background-size: contain;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"],
    .stTabs [data-testid="stTab"][aria-selected="true"],
    .stTabs [data-testid="stTab"][data-selected] {
        background-color: #DBEAFE !important;
        color: #2563EB !important;
        border-color: #BFDBFE !important;
    }
    .stTabs [data-testid="stTab"]:hover {
        color: #2563EB !important;
        border-color: #BFDBFE !important;
    }

    /* ============ BUTTONS ============ */
    .stButton > button {
        border-radius: 999px;
        font-weight: 600;
        padding: 8px 22px;
        border: 1px solid #E2E8F0;
        transition: all 0.2s ease;
    }
    .stButton > button[kind="primary"] {
        background-color: #2563EB;
        color: white;
        border: none;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #1D4ED8;
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(37, 99, 235, 0.25);
    }
    .stButton > button[kind="secondary"]:hover {
        border-color: #2563EB;
        color: #2563EB;
    }

    /* FIX TAMBAHAN — outline pill button (sesuai mockup):
       latar putih, teks biru, border biru tipis. Dipakai untuk tombol
       "Lihat penjelasan polutan" dan "Lihat Selengkapnya". */
    .stButton > button.outline-pill,
    div[data-testid="stButton"] > button {
        /* default semua button non-primary jadi outline pill modern */
    }
    /* Khusus tombol "Lihat penjelasan polutan" & "Lihat Selengkapnya":
       outline biru saja (latar putih, teks & border biru), TANPA efek hover.
       Flag " i" = case-insensitive, agar cocok di semua halaman termasuk
       label "Penjelasan" (P kapital) di halaman Simulasi Prediksi ISPU. */
    /* Tombol "Lihat penjelasan polutan" di Dashboard, Detail Wilayah, &
       Simulasi (key diawali "btn_info_") + tombol "Lihat Selengkapnya".
       Target utama lewat class st-key-* (paling andal di Streamlit modern),
       dengan fallback aria-label. Outline biru saja — TANPA hover/focus/active. */
    [class*="st-key-btn_info"] button,
    [class*="st-key-btn_info"] button:hover,
    [class*="st-key-btn_info"] button:focus,
    [class*="st-key-btn_info"] button:focus-visible,
    [class*="st-key-btn_info"] button:active,
    [class*="st-key-btn_reset"] button,
    [class*="st-key-btn_reset"] button:hover,
    [class*="st-key-btn_reset"] button:focus,
    [class*="st-key-btn_reset"] button:focus-visible,
    [class*="st-key-btn_reset"] button:active,
    div[data-testid="stButton"]:has(button[aria-label*="penjelasan" i]) > button,
    div[data-testid="stButton"]:has(button[aria-label*="penjelasan" i]) > button:hover,
    div[data-testid="stButton"]:has(button[aria-label*="penjelasan" i]) > button:focus,
    div[data-testid="stButton"]:has(button[aria-label*="penjelasan" i]) > button:focus-visible,
    div[data-testid="stButton"]:has(button[aria-label*="penjelasan" i]) > button:active,
    div[data-testid="stButton"]:has(button[aria-label*="Selengkapnya" i]) > button,
    div[data-testid="stButton"]:has(button[aria-label*="Selengkapnya" i]) > button:hover,
    div[data-testid="stButton"]:has(button[aria-label*="Selengkapnya" i]) > button:focus,
    div[data-testid="stButton"]:has(button[aria-label*="Selengkapnya" i]) > button:active {
        background: #FFFFFF !important;
        background-color: #FFFFFF !important;
        color: #2563EB !important;
        border: 1px solid #2563EB !important;
        border-color: #2563EB !important;
        font-weight: 600 !important;
        transform: none !important;
        box-shadow: none !important;
        transition: none !important;
    }

    /* FIX TAMBAHAN — right-align tombol "Lihat penjelasan polutan"
       di dalam kolomnya. Tanpa ini, tombol rapat kiri di column 1/3
       dengan whitespace di kanan (floating effect yang tidak rapi). */
    div[data-testid="stHorizontalBlock"]:has(button[aria-label*="penjelasan"])
        > div:last-child > div[data-testid="stVerticalBlock"] {
        align-items: flex-end !important;
    }
    div[data-testid="stHorizontalBlock"]:has(button[aria-label*="penjelasan"])
        > div:last-child div[data-testid="stButton"] {
        display: flex !important;
        justify-content: flex-end !important;
        width: 100%;
    }

    /* Tombol "Lihat Selengkapnya" (di bawah peta) — rounded pill, border
       biru tipis, latar putih, padding besar seperti desain Figma. */
    [class*="st-key-btn_selengkapnya"] button,
    [class*="st-key-btn_selengkapnya"] button:hover,
    [class*="st-key-btn_selengkapnya"] button:focus,
    [class*="st-key-btn_selengkapnya"] button:focus-visible,
    [class*="st-key-btn_selengkapnya"] button:active {
        background: #FFFFFF !important;
        color: #4A6CF7 !important;
        border: 1.5px solid #4A6CF7 !important;
        border-radius: 999px !important;
        padding: 11px 30px !important;
        font-weight: 600 !important;
        box-shadow: none !important;
        transform: none !important;
        transition: none !important;
    }

    /* ============ SLIDER ============ */
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background-color: #2563EB;
        box-shadow: 0 2px 6px rgba(37, 99, 235, 0.3);
    }

    /* ============ EXPANDER (popup polutan) ============ */
    .streamlit-expanderHeader {
        background-color: #FFFFFF !important;
        border-radius: 14px !important;
        font-weight: 600 !important;
        border: 1px solid #E2E8F0 !important;
    }

    /* ============ DONUT LEGEND CUSTOM ============ */
    .donut-legend-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 5px 0;
        font-size: 13px;
        gap: 8px;
    }
    .donut-legend-left {
        display: flex;
        align-items: center;
        gap: 8px;
        color: #0F172A;
        white-space: nowrap;
    }
    .donut-legend-dot {
        width: 10px;
        height: 10px;
        border-radius: 999px;
        flex-shrink: 0;
    }
    .donut-legend-pct {
        font-weight: 700;
        color: #0F172A;
        flex-shrink: 0;
    }
    /* Paksa kolom Dampak & Sumber Polusi sejajar bawah */
    div[data-testid="column"] > div[data-testid="stVerticalBlock"] {
        height: 100%;
    }
    .info-card {
        display: flex;
        align-items: flex-start;
        gap: 10px;
        background: #F1F5F9;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 10px 14px;
        margin-top: 14px;
        margin-bottom: 4px;
        font-size: 13px;
        color: #475569;
        line-height: 1.5;
    }
    .info-card .info-icon {
        font-size: 15px;
        color: #64748B;
        margin-top: 1px;
        flex-shrink: 0;
    }

    /* Responsivitas tablet/mobile */
    @media (max-width: 768px) {
        .ispu-number { font-size: 48px; }
        .pollutant-value { font-size: 22px; }
        .pollutant-grid { grid-template-columns: repeat(3, 1fr); }
        .step-bar { grid-template-columns: 1fr; }
    }
    </style>
    """, unsafe_allow_html=True)


# ================================================================
# UTILITIES
# ================================================================
@st.cache_data
def load_data():
    """Memuat semua data dummy (ditanam langsung di app.py, tanpa file CSV eksternal)."""
    ispu_csv = """tanggal,ispu
2025-10-25,44
2025-10-26,60
2025-10-27,61
2025-10-28,69
2025-10-29,74
2025-10-30,68
2025-10-31,69
"""
    wilayah_csv = """wilayah,ispu,kategori,lat,lon,pm25,pm10,no2,so2,co,o3
Jakarta Pusat,71,Sedang,-6.1924,106.8232,68,49,35,26,18.6,19
Jakarta Utara,65,Sedang,-6.1565,106.9056,62,53,22,47,13.8,20
Jakarta Barat,102,Tidak Sehat,-6.1881,106.7567,102,60,30,28,15.0,40
Jakarta Selatan,68,Sedang,-6.2615,106.8106,64,47,43,48,12.3,18
Jakarta Timur,71,Sedang,-6.225,106.9004,67,59,17,32,16.7,22
Kep. Seribu,42,Baik,-5.75,106.6,12,22,8,3,0.3,28
"""
    prediksi_csv = """tanggal,wilayah,ispu,kategori,pm25
2024-06-16,DKI Jakarta,74,Sedang,28
2024-06-17,DKI Jakarta,78,Sedang,31
2024-06-18,DKI Jakarta,82,Sedang,34
2024-06-19,DKI Jakarta,95,Sedang,41
2024-06-20,DKI Jakarta,108,Tidak Sehat,48
2024-06-21,DKI Jakarta,102,Tidak Sehat,45
2024-06-22,DKI Jakarta,89,Sedang,37
2024-06-16,Jakarta Pusat,68,Sedang,26
2024-06-17,Jakarta Pusat,72,Sedang,29
2024-06-18,Jakarta Pusat,108,Tidak Sehat,45
2024-06-19,Jakarta Pusat,77,Sedang,32
2024-06-20,Jakarta Pusat,102,Tidak Sehat,49
2024-06-21,Jakarta Pusat,72,Sedang,45
2024-06-22,Jakarta Pusat,60,Sedang,37
2024-06-16,Jakarta Utara,65,Sedang,24
2024-06-17,Jakarta Utara,70,Sedang,27
2024-06-18,Jakarta Utara,88,Sedang,38
2024-06-19,Jakarta Utara,75,Sedang,30
2024-06-20,Jakarta Utara,82,Sedang,35
2024-06-21,Jakarta Utara,69,Sedang,26
2024-06-22,Jakarta Utara,58,Sedang,22
2024-06-16,Jakarta Barat,102,Tidak Sehat,46
2024-06-17,Jakarta Barat,115,Tidak Sehat,52
2024-06-18,Jakarta Barat,125,Tidak Sehat,58
2024-06-19,Jakarta Barat,110,Tidak Sehat,50
2024-06-20,Jakarta Barat,118,Tidak Sehat,54
2024-06-21,Jakarta Barat,105,Tidak Sehat,48
2024-06-22,Jakarta Barat,95,Sedang,42
2024-06-16,Jakarta Selatan,71,Sedang,26
2024-06-17,Jakarta Selatan,78,Sedang,30
2024-06-18,Jakarta Selatan,95,Sedang,42
2024-06-19,Jakarta Selatan,82,Sedang,34
2024-06-20,Jakarta Selatan,89,Sedang,38
2024-06-21,Jakarta Selatan,76,Sedang,29
2024-06-22,Jakarta Selatan,64,Sedang,24
2024-06-16,Jakarta Timur,68,Sedang,25
2024-06-17,Jakarta Timur,74,Sedang,28
2024-06-18,Jakarta Timur,92,Sedang,40
2024-06-19,Jakarta Timur,80,Sedang,33
2024-06-20,Jakarta Timur,86,Sedang,36
2024-06-21,Jakarta Timur,72,Sedang,27
2024-06-22,Jakarta Timur,62,Sedang,23
2024-06-16,Kep. Seribu,42,Baik,15
2024-06-17,Kep. Seribu,45,Baik,16
2024-06-18,Kep. Seribu,52,Sedang,19
2024-06-19,Kep. Seribu,48,Baik,17
2024-06-20,Kep. Seribu,55,Sedang,21
2024-06-21,Kep. Seribu,44,Baik,16
2024-06-22,Kep. Seribu,38,Baik,14
"""
    edukasi_csv = (
        'kategori,rentang,deskripsi,warna,emoji\n'
        'Baik,0-50,"Udara bersih, aman untuk beraktivitas sehari-hari.",#16A34A,😊\n'
        'Sedang,51-100,Masih dapat diterima untuk beraktivitas luar ruangan.,#2563EB,😐\n'
        'Tidak Sehat,> 100,"Kurangi aktivitas luar ruangan, terutama bagi kelompok sensitif.",#E5B93D,😷\n'
    )
    return {
        "ispu":     pd.read_csv(io.StringIO(ispu_csv)),
        "wilayah":  pd.read_csv(io.StringIO(wilayah_csv)),
        "prediksi": pd.read_csv(io.StringIO(prediksi_csv)),
        "edukasi":  pd.read_csv(io.StringIO(edukasi_csv)),
    }


@st.cache_resource
def load_model():
    """
    Memuat SEMUA artefak model terlatih (sama seperti yang disimpan notebook
    di cell [66]): XGBoost, Random Forest, SVM, LabelEncoder, StandardScaler,
    dan daftar fitur. SVM butuh scaler agar prediksinya identik dengan notebook.
    """
    try:
        return {
            "xgb":    joblib.load(MODELS_DIR / "model_xgboost.pkl"),
            "rf":     joblib.load(MODELS_DIR / "model_random_forest.pkl"),
            "svm":    joblib.load(MODELS_DIR / "model_svm.pkl"),
            "le":     joblib.load(MODELS_DIR / "label_encoder.pkl"),
            "scaler": joblib.load(MODELS_DIR / "standard_scaler.pkl"),
            "fitur":  joblib.load(MODELS_DIR / "fitur_polutan.pkl"),
            "ok": True,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


def get_logo_b64():
    """Logo SVG ke base64 untuk disisipkan sebagai <img>."""
    logo_path = ASSETS_DIR / "logo.svg"
    if logo_path.exists():
        return base64.b64encode(logo_path.read_bytes()).decode()
    return ""


@st.cache_data
def rekom_img_b64(filename: str) -> str:
    """Baca PNG rekomendasi dari assets/ lalu encode base64."""
    if not filename:
        return ""
    p = ASSETS_DIR / filename
    if not p.exists():
        # Fallback: coba cari file dengan nama case-insensitive
        for f in ASSETS_DIR.iterdir():
            if f.name.lower() == filename.lower():
                return base64.b64encode(f.read_bytes()).decode()
        return ""
    return base64.b64encode(p.read_bytes()).decode()


def svg_inline(filename: str) -> str:
    """Baca SVG dari assets/ dan kembalikan markup inline (skala 100% lebar).
    Lebih andal daripada <img data:base64> untuk SVG kompleks."""
    if not filename:
        return ""
    p = ASSETS_DIR / filename
    if not p.exists():
        for f in ASSETS_DIR.iterdir():
            if f.name.lower() == filename.lower():
                p = f
                break
        else:
            return ""
    svg = p.read_text(encoding="utf-8")
    # Sisipkan style width:100% pada tag <svg ...> pertama (override width/height tetap)
    svg = re.sub(
        r'<svg\b',
        '<svg style="width:100%;height:auto;display:block"',
        svg, count=1,
    )
    return svg


def kategori_dari_ispu(ispu):
    """Konversi nilai ISPU ke kategori (skema 3 kelas sistem JakU)."""
    if ispu <= 50:    return "Baik"
    if ispu <= 100:   return "Sedang"
    return "Tidak Sehat"


# ─────────────────────────────────────────────────────────────────
# ISPU SUB-INDEX — PerMenLHK No. 14/2020
# ─────────────────────────────────────────────────────────────────
# Setiap polutan punya 5 pita konsentrasi yang dipetakan ke 5 pita
# indeks ISPU. Tuple format: (bp_low, bp_high, idx_low, idx_high)
#   bp_low/bp_high : batas bawah/atas konsentrasi polutan
#   idx_low/idx_high : batas bawah/atas indeks ISPU yang sesuai
#
# Boundary di-overlap (mis. band 0 hi=15.5, band 1 lo=15.5) supaya
# tidak ada gap numerik — formula interpolasi linear menghasilkan
# nilai yang sama di titik boundary, jadi aman untuk dimatch ke
# band manapun.
#
# Rumus interpolasi (PerMenLHK):
#     I = ((Ia - Ib) / (Xa - Xb)) * (Xx - Xb) + Ib
#   I  = sub-indeks polutan
#   Xx = konsentrasi aktual
#   Xb, Xa = batas konsentrasi bawah/atas pita
#   Ib, Ia = batas indeks bawah/atas pita
# ─────────────────────────────────────────────────────────────────
BREAKPOINTS = {
    # PM2.5 — µg/m³, rata-rata 24 jam
    "pm25": [
        (0,     15.5,  0,   50),
        (15.5,  55.4,  50,  100),
        (55.4,  150.4, 100, 200),
        (150.4, 250.4, 200, 300),
        (250.4, 500,   300, 500),
    ],
    # PM10 — µg/m³, rata-rata 24 jam
    "pm10": [
        (0,   50,  0,   50),
        (50,  150, 50,  100),
        (150, 350, 100, 200),
        (350, 420, 200, 300),
        (420, 500, 300, 500),
    ],
    # SO₂ — µg/m³, rata-rata 24 jam
    "so2": [
        (0,   52,   0,   50),
        (52,  180,  50,  100),
        (180, 400,  100, 200),
        (400, 800,  200, 300),
        (800, 1200, 300, 500),
    ],
    # CO — mg/m³, rata-rata 8 jam (catatan: sangat sensitif,
    # 9 mg/m³ sudah masuk band Tidak Sehat)
    "co": [
        (0,  4,  0,   50),
        (4,  8,  50,  100),
        (8,  15, 100, 200),
        (15, 30, 200, 300),
        (30, 45, 300, 500),
    ],
    # O₃ — µg/m³, rata-rata 8 jam
    "o3": [
        (0,   120,  0,   50),
        (120, 235,  50,  100),
        (235, 400,  100, 200),
        (400, 800,  200, 300),
        (800, 1000, 300, 500),
    ],
    # NO₂ — µg/m³, rata-rata 1 jam
    "no2": [
        (0,    80,   0,   50),
        (80,   200,  50,  100),
        (200,  1130, 100, 200),
        (1130, 2260, 200, 300),
        (2260, 3000, 300, 500),
    ],
}

# Threshold kategori ISPU — sistem JakU memakai 3 kelas
ISPU_CATEGORY_THRESHOLDS = [
    (50,    "Baik"),
    (100,   "Sedang"),
    (float("inf"), "Tidak Sehat"),
]


def calculate_subindex(value: float, breakpoints: list) -> float:
    """
    Hitung sub-indeks ISPU untuk SATU polutan dengan interpolasi linear.

    Args:
        value: konsentrasi aktual polutan (sesuai satuannya)
        breakpoints: list tuple (bp_low, bp_high, idx_low, idx_high)

    Returns:
        nilai sub-indeks (0–500). Di luar range, di-clamp ke 0 atau 500.

    Rumus PerMenLHK 14/2020:
        I = ((idx_high - idx_low) / (bp_high - bp_low)) * (value - bp_low) + idx_low
    """
    # Edge case: nilai nol atau negatif → sub-indeks 0
    if value <= 0:
        return 0.0
    # Edge case: nilai melebihi breakpoint maksimum → cap 500
    last_bp_high = breakpoints[-1][1]
    if value >= last_bp_high:
        return 500.0
    # Cari band yang memuat nilai ini, lalu interpolasi linear
    for bp_low, bp_high, idx_low, idx_high in breakpoints:
        if bp_low <= value <= bp_high:
            return ((idx_high - idx_low) / (bp_high - bp_low)) * (value - bp_low) + idx_low
    # Fallback (seharusnya tak terjangkau karena range BREAKPOINTS kontinu)
    return 500.0


def calculate_final_ispu(values: dict) -> tuple:
    """
    Hitung sub-indeks SEMUA polutan + ISPU final + polutan dominan.

    Args:
        values: dict {"pm25": float, "pm10": float, "no2": float,
                      "so2": float,  "co":   float, "o3":  float}

    Returns:
        (final_ispu, polutan_dominan, dict_subindeks)
        final_ispu = max sub-indeks
        polutan_dominan = key polutan dengan sub-indeks tertinggi
        dict_subindeks = {polutan: sub_index} untuk semua 6 polutan
    """
    subindeks = {
        pol: calculate_subindex(values.get(pol, 0.0), bps)
        for pol, bps in BREAKPOINTS.items()
    }
    final_ispu = max(subindeks.values())
    polutan_dominan = max(subindeks, key=subindeks.get)
    return final_ispu, polutan_dominan, subindeks


def get_ispu_category(ispu_value: float) -> str:
    """Mapping nilai ISPU ke kategori (3 kelas: Baik / Sedang / Tidak Sehat)."""
    for threshold, kategori in ISPU_CATEGORY_THRESHOLDS:
        if ispu_value <= threshold:
            return kategori
    return "Tidak Sehat"  # safety net


# ─── Wrappers untuk kompatibilitas dengan kode existing ───
def calculate_ispu_category(pm10, pm25, so2, co, o3, no2):
    """
    Wrapper signature lama. Internal-nya komposisi 3 fungsi modular di atas.
    Returns: (nilai_ispu_dibulatkan, kategori, polutan_dominan, dict_subindeks)
    """
    values = {"pm10": pm10, "pm25": pm25, "so2": so2, "co": co, "o3": o3, "no2": no2}
    final_ispu, polutan_dominan, subindeks = calculate_final_ispu(values)
    kategori = get_ispu_category(final_ispu)
    return round(final_ispu, 1), kategori, polutan_dominan, subindeks


def hitung_ispu(pm10, pm25, so2, co, o3, no2):
    """Wrapper backward-compatible — hanya mengembalikan (nilai, kategori)."""
    nilai, kategori, _, _ = calculate_ispu_category(pm10, pm25, so2, co, o3, no2)
    return nilai, kategori


# =================================================================
# SVG INLINE HELPERS (FIX #4, #5, #6)
# -----------------------------------------------------------------
# Mengganti emoji native (yang terlihat seperti emoji default sistem)
# dengan SVG inline kustom — konsisten lintas device & sesuai mockup.
# Logo sprout #0A6847 dan ilustrasi Jakarta juga dipindah ke SVG inline.
# =================================================================
def logo_jaku_svg(size=40):
    """
    Logo JakU - sprout sesuai mockup Figma.
    Tiga daun mekar (gelap-terang-tunas) + 2 tetesan biru kecil di bawah daun.
    """
    return f"""
    <svg width="{size}" height="{size}" viewBox="0 0 64 64"
         xmlns="http://www.w3.org/2000/svg" style="flex-shrink:0;">
      <!-- Daun kiri (gelap) -->
      <path d="M30 36 C16 36 8 24 12 10 C26 12 34 24 30 36 Z"
            fill="#0A6847"/>
      <!-- Daun kanan (sedang) -->
      <path d="M34 32 C48 32 56 20 52 6 C38 8 30 20 34 32 Z"
            fill="#16A34A"/>
      <!-- Tunas tengah (lancip ke atas, hijau muda) -->
      <path d="M32 30 C30 22 32 14 32 8 C32 14 34 22 32 30 Z"
            fill="#22C55E"/>
      <!-- Batang -->
      <path d="M32 48 L32 32" stroke="#0A6847" stroke-width="2.5"
            stroke-linecap="round" fill="none"/>
      <!-- Tetesan biru kiri & kanan (aksen air) -->
      <circle cx="26" cy="52" r="2.5" fill="#3B82F6"/>
      <circle cx="38" cy="52" r="2.5" fill="#3B82F6"/>
      <ellipse cx="32" cy="56" rx="3" ry="2" fill="#2563EB" opacity="0.85"/>
    </svg>
    """.strip()


def ispu_emoji_svg(kategori, size=72):
    """
    Emoji status udara dalam SVG inline (flat, clean, konsisten).
    Mengganti emoji native (😐 dll) yang terlihat random per OS.
    """
    cfg = {
        "Baik": {
            "fill": "#16A34A",
            "mouth": '<path d="M30 60 Q50 75 70 60" stroke="white" stroke-width="5" stroke-linecap="round" fill="none"/>',
            "eyes": '<circle cx="36" cy="42" r="4" fill="white"/><circle cx="64" cy="42" r="4" fill="white"/>',
        },
        "Sedang": {
            "fill": "#3B82F6",
            "mouth": '<line x1="35" y1="62" x2="65" y2="62" stroke="white" stroke-width="5" stroke-linecap="round"/>',
            "eyes": '<circle cx="36" cy="42" r="4" fill="white"/><circle cx="64" cy="42" r="4" fill="white"/>',
        },
        "Tidak Sehat": {
            "fill": "#F59E0B",
            "mouth": '<path d="M30 68 Q50 56 70 68" stroke="white" stroke-width="5" stroke-linecap="round" fill="none"/>',
            "eyes": '<line x1="30" y1="40" x2="42" y2="44" stroke="white" stroke-width="4" stroke-linecap="round"/><line x1="70" y1="40" x2="58" y2="44" stroke="white" stroke-width="4" stroke-linecap="round"/>',
        },
        "Sangat Tidak Sehat": {
            "fill": "#EF4444",
            "mouth": '<path d="M30 70 Q50 55 70 70" stroke="white" stroke-width="5" stroke-linecap="round" fill="none"/>',
            "eyes": '<path d="M30 38 L42 48 M42 38 L30 48" stroke="white" stroke-width="4" stroke-linecap="round"/><path d="M58 38 L70 48 M70 38 L58 48" stroke="white" stroke-width="4" stroke-linecap="round"/>',
        },
        "Berbahaya": {
            "fill": "#7C3AED",
            "mouth": '<path d="M30 70 Q50 55 70 70" stroke="white" stroke-width="5" stroke-linecap="round" fill="none"/>',
            "eyes": '<circle cx="36" cy="44" r="6" fill="white"/><circle cx="64" cy="44" r="6" fill="white"/><circle cx="36" cy="44" r="2" fill="#7C3AED"/><circle cx="64" cy="44" r="2" fill="#7C3AED"/>',
        },
    }
    c = cfg.get(kategori, cfg["Sedang"])
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 100 100" '
        f'xmlns="http://www.w3.org/2000/svg" style="flex-shrink:0;">'
        f'<circle cx="50" cy="50" r="46" fill="{c["fill"]}"/>'
        f'{c["eyes"]}{c["mouth"]}'
        f'</svg>'
    )


def jakarta_skyline_svg(width=180):
    """
    Ilustrasi flat Jakarta skyline (Monas + gedung).
    Mengikuti mockup: gradient lembut, gedung outline tipis biru-abu,
    Monas tegak dengan ujung emas, pohon-pohon hijau di foreground.
    """
    return f"""
    <svg width="{width}" viewBox="0 0 200 130"
         xmlns="http://www.w3.org/2000/svg"
         style="display:block; opacity:0.95;">
      <defs>
        <linearGradient id="skyGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="#EFF6FF"/>
          <stop offset="55%" stop-color="#F0FDF4"/>
          <stop offset="100%" stop-color="#FFFFFF"/>
        </linearGradient>
      </defs>
      <!-- Background gradient -->
      <rect width="200" height="120" fill="url(#skyGrad)" rx="6"/>
      <!-- Gedung-gedung latar (outline tipis, fill sangat lembut) -->
      <rect x="10" y="78" width="18" height="38" fill="#DBEAFE"
            stroke="#94A3B8" stroke-width="0.6" opacity="0.55" rx="1"/>
      <rect x="30" y="62" width="14" height="54" fill="#E0E7FF"
            stroke="#94A3B8" stroke-width="0.6" opacity="0.55" rx="1"/>
      <rect x="46" y="72" width="20" height="44" fill="#DBEAFE"
            stroke="#94A3B8" stroke-width="0.6" opacity="0.55" rx="1"/>
      <rect x="68" y="58" width="16" height="58" fill="#E0E7FF"
            stroke="#94A3B8" stroke-width="0.6" opacity="0.55" rx="1"/>
      <!-- Jendela2 simbolik untuk gedung kiri -->
      <line x1="34" y1="72" x2="42" y2="72" stroke="#94A3B8"
            stroke-width="0.4" opacity="0.6"/>
      <line x1="34" y1="82" x2="42" y2="82" stroke="#94A3B8"
            stroke-width="0.4" opacity="0.6"/>
      <line x1="34" y1="92" x2="42" y2="92" stroke="#94A3B8"
            stroke-width="0.4" opacity="0.6"/>
      <!-- Monas (tugu tengah, paling tinggi) -->
      <rect x="98" y="40" width="4" height="76" fill="#E5E7EB"
            stroke="#64748B" stroke-width="0.5"/>
      <!-- Ujung emas Monas (puncak api) -->
      <polygon points="96,40 104,40 100,28" fill="#FBBF24"
               stroke="#D97706" stroke-width="0.4"/>
      <!-- Base Monas (alas) -->
      <rect x="92" y="106" width="16" height="10" fill="#E5E7EB"
            stroke="#64748B" stroke-width="0.5"/>
      <!-- Gedung-gedung kanan -->
      <rect x="116" y="68" width="16" height="48" fill="#E0E7FF"
            stroke="#94A3B8" stroke-width="0.6" opacity="0.55" rx="1"/>
      <rect x="134" y="75" width="20" height="41" fill="#DBEAFE"
            stroke="#94A3B8" stroke-width="0.6" opacity="0.55" rx="1"/>
      <rect x="156" y="60" width="14" height="56" fill="#E0E7FF"
            stroke="#94A3B8" stroke-width="0.6" opacity="0.55" rx="1"/>
      <rect x="172" y="72" width="18" height="44" fill="#DBEAFE"
            stroke="#94A3B8" stroke-width="0.6" opacity="0.55" rx="1"/>
      <!-- Jendela2 simbolik gedung kanan -->
      <line x1="138" y1="85" x2="150" y2="85" stroke="#94A3B8"
            stroke-width="0.4" opacity="0.6"/>
      <line x1="138" y1="95" x2="150" y2="95" stroke="#94A3B8"
            stroke-width="0.4" opacity="0.6"/>
      <!-- Pohon-pohon foreground (hijau bulat) -->
      <circle cx="14" cy="116" r="8" fill="#16A34A" opacity="0.9"/>
      <circle cx="74" cy="118" r="6" fill="#16A34A" opacity="0.9"/>
      <circle cx="124" cy="118" r="7" fill="#16A34A" opacity="0.9"/>
      <circle cx="186" cy="116" r="8" fill="#16A34A" opacity="0.9"/>
      <!-- Detail pohon (tone berbeda untuk depth) -->
      <circle cx="20" cy="114" r="5" fill="#22C55E" opacity="0.85"/>
      <circle cx="180" cy="114" r="5" fill="#22C55E" opacity="0.85"/>
    </svg>
    """.strip()


def render_legend_safe(kategori_info):
    """
    FIX #1 & #2 — Legend peta yang reliable.

    Sebelumnya: triple-quote + "".join + indentasi membuat Streamlit/markdown
    salah mendeteksi code block, sehingga hanya baris pertama yang terender.

    Sekarang: bangun SATU string HTML utuh tanpa newline & tanpa indentasi
    awal-baris. SATU panggilan st.markdown.
    """
    rows = ""
    for nama, info in kategori_info.items():
        # Inline-only HTML, NO leading whitespace di awal tag baru
        rows += (
            '<div style="display:flex;align-items:center;gap:8px;'
            'margin:7px 0;font-size:13px;color:#334155;">'
            f'<span style="width:11px;height:11px;border-radius:50%;'
            f'background:{info["warna"]};display:inline-block;flex-shrink:0;"></span>'
            f'<span><strong style="color:#0F172A;font-weight:600;">{nama}</strong> '
            f'({info["rentang"]})</span>'
            '</div>'
        )
    html = (
        '<div style="padding-top:4px;">'
        '<div style="font-weight:700;font-size:14px;color:#0F172A;'
        'margin-bottom:10px;">Keterangan:</div>'
        + rows +
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def prediksi_ispu_xgboost(pm10, pm25, so2, co, o3, no2, model_choice="xgboost"):
    """
    Prediksi kategori ISPU - replikasi PERSIS fungsi prediksi_ispu() notebook
    (cell [70]). Mendukung 3 model:
        - 'xgboost'        -> model_xgboost.pkl, input mentah
        - 'random_forest'  -> model_random_forest.pkl, input mentah
        - 'svm'            -> model_svm.pkl, input WAJIB di-scale dulu

    Urutan fitur & cara prediksi identik notebook, jadi hasil == notebook
    (selama file .pkl-nya juga dari notebook / dataset yang sama).

    Mengembalikan dict: kategori, nilai_ispu (estimasi untuk display),
    confidence, model_used, fallback.
    """
    art = load_model()
    if not art["ok"]:
        # Fallback bobot polutan jika model gagal dimuat
        nilai = pm25 * 0.30 + pm10 * 0.20 + no2 * 0.15 + so2 * 0.15 + co * 0.10 + o3 * 0.10
        return {
            "kategori": kategori_dari_ispu(nilai),
            "nilai_ispu": int(round(nilai)),
            "confidence": None,
            "model_used": "Formula (fallback)",
            "fallback": True,
        }

    # Susun input PERSIS urutan notebook (cell [70])
    input_df = pd.DataFrame([{
        "pm_sepuluh":        pm10,
        "pm_duakomalima":    pm25,
        "sulfur_dioksida":   so2,
        "karbon_monoksida":  co,
        "ozon":              o3,
        "nitrogen_dioksida": no2,
    }])[art["fitur"]]

    # Pilih model + cara prediksi (sama persis logika notebook)
    confidence = None
    if model_choice == "random_forest":
        model = art["rf"]
        pred_idx = model.predict(input_df)[0]
        model_used = "Random Forest"
        try:
            confidence = float(np.max(model.predict_proba(input_df)[0]))
        except Exception:
            pass
    elif model_choice == "svm":
        model = art["svm"]
        # SVM WAJIB di-scale dulu (cell [70] notebook)
        input_scaled = art["scaler"].transform(input_df)
        pred_idx = model.predict(input_scaled)[0]
        model_used = "SVM"
        # SVC default tanpa probability=True -> tidak ada predict_proba
        try:
            confidence = float(np.max(model.predict_proba(input_scaled)[0]))
        except Exception:
            confidence = None
    else:  # default xgboost
        model = art["xgb"]
        pred_idx = model.predict(input_df)[0]
        model_used = "XGBoost"
        try:
            confidence = float(np.max(model.predict_proba(input_df)[0]))
        except Exception:
            pass

    # Decode label (BAIK / SEDANG / TIDAK SEHAT) -> sama dengan notebook
    kategori_raw = art["le"].inverse_transform([pred_idx])[0]
    kat_map = {"BAIK": "Baik", "SEDANG": "Sedang", "TIDAK SEHAT": "Tidak Sehat"}
    kategori = kat_map.get(kategori_raw, "Sedang")

    # Estimasi nilai ISPU numerik (HANYA untuk display angka besar di UI;
    # kategori tetap mengikuti output model, bukan angka ini)
    nilai = pm25 * 0.30 + pm10 * 0.20 + no2 * 0.15 + so2 * 0.15 + co * 0.10 + o3 * 0.10
    if kategori == "Baik":          nilai = min(nilai, 50)
    elif kategori == "Sedang":      nilai = max(51, min(nilai, 100))
    elif kategori == "Tidak Sehat": nilai = max(101, min(nilai, 200))

    return {
        "kategori": kategori,
        "nilai_ispu": int(round(nilai)),
        "confidence": confidence,
        "model_used": model_used,
        "fallback": False,
    }


# ================================================================
# ENGINE DATA HARIAN BERBASIS CSV ASLI (Data_ISPU.csv) + XGBOOST
# ----------------------------------------------------------------
# Sumber data: file ISPU di folder project dari 5 stasiun SPKU
# DKI Jakarta (DKI1-DKI5), Jan 2024 - Nov 2025. XGBoost (classifier)
# mengklasifikasikan kategori dari 6 polutan tiap (wilayah, tanggal);
# nilai ISPU yang ditampilkan = kolom `max` (ISPU asli BMKG).
# Kep. Seribu TIDAK ada di CSV -> selalu "tidak ada data".
# Stasiun -> Wilayah:
#   DKI1 (Bundaran HI)   -> Jakarta Pusat
#   DKI2 (Kelapa Gading) -> Jakarta Utara
#   DKI3 (Jagakarsa)     -> Jakarta Selatan
#   DKI4 (Lubang Buaya)  -> Jakarta Timur
#   DKI5 (Kebon Jeruk)   -> Jakarta Barat
# ================================================================
WILAYAH_DKI = ["Jakarta Pusat", "Jakarta Utara", "Jakarta Barat",
               "Jakarta Selatan", "Jakarta Timur"]

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


FITUR_CSV = _FITUR_ISPU

_PREFIX_WILAYAH = (
    ("DKI1", "Jakarta Pusat"), ("DKI2", "Jakarta Utara"),
    ("DKI3", "Jakarta Selatan"), ("DKI4", "Jakarta Timur"),
    ("DKI5", "Jakarta Barat"),
)


def _stasiun_ke_wilayah(s):
    s = str(s)
    for prefix, wil in _PREFIX_WILAYAH:
        if s.startswith(prefix):
            return wil
    return None


@st.cache_data(show_spinner=False)
def load_ispu_harian():
    """Baca file data ISPU di folder project (Data_ISPU.csv ATAU .xls unduhan
    open data DKI), petakan stasiun->wilayah, bangun kolom tanggal, dan
    imputasi median untuk polutan yang kosong (mengikuti pipeline notebook)."""
    df = baca_ispu(BASE_DIR, DATA_DIR).copy()
    if df.empty:
        return pd.DataFrame()
    df["wilayah"] = df["stasiun"].map(_stasiun_ke_wilayah)
    df = df[df["wilayah"].notna()].copy()
    df["tahun"] = (df["periode_data"] // 100).astype(int)
    df["tgl"] = pd.to_datetime(
        dict(year=df["tahun"], month=df["bulan"], day=df["tanggal"]),
        errors="coerce",
    )
    df = df[df["tgl"].notna()].copy()
    df["tgl_date"] = df["tgl"].dt.date
    # Imputasi median per kolom polutan (sama seperti notebook CRISP-DM)
    for c in FITUR_CSV:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].fillna(df[c].median())
    df["max"] = pd.to_numeric(df["max"], errors="coerce")
    # Bila ada >1 baris per (wilayah, tanggal) — ambil yang pertama
    df = df.drop_duplicates(subset=["wilayah", "tgl_date"], keep="first")
    return df


@st.cache_data(show_spinner=False)
def rentang_data():
    """Rentang tanggal date picker — periode 202401 sampai 202511."""
    lo_cap, hi_cap = date(2024, 1, 1), date(2025, 11, 30)
    df = load_ispu_harian()
    if df.empty:
        return (lo_cap, hi_cap)
    lo = max(min(df["tgl_date"]), lo_cap)
    hi = min(max(df["tgl_date"]), hi_cap)
    if lo > hi:                      # CSV di luar rentang → pakai default
        return (lo_cap, hi_cap)
    return (lo, hi)


@st.cache_data(show_spinner=False)
def predict_wilayah_tanggal(wilayah, d_iso, model_choice="xgboost"):
    """Ambil polutan asli (wilayah, tanggal) dari CSV -> XGBoost -> kategori.
    Mengembalikan None bila wilayah/tanggal tidak ada di CSV (mis. Kep. Seribu)."""
    df = load_ispu_harian()
    if df.empty:
        return None
    d = date.fromisoformat(d_iso)
    sub = df[(df["wilayah"] == wilayah) & (df["tgl_date"] == d)]
    if sub.empty:
        return None
    r = sub.iloc[0]
    res = prediksi_ispu_xgboost(
        r["pm_sepuluh"], r["pm_duakomalima"], r["sulfur_dioksida"],
        r["karbon_monoksida"], r["ozon"], r["nitrogen_dioksida"], model_choice,
    )
    # Nilai ISPU = kolom `max` (ISPU asli); fallback ke estimasi model bila kosong
    if pd.notna(r["max"]) and r["max"] > 0:
        ispu = int(round(r["max"]))
    else:
        ispu = int(res["nilai_ispu"])
    return {
        "kategori": res["kategori"], "ispu": ispu,
        "pm25": round(float(r["pm_duakomalima"]), 1),
        "pm10": round(float(r["pm_sepuluh"]), 1),
        "no2": round(float(r["nitrogen_dioksida"]), 1),
        "so2": round(float(r["sulfur_dioksida"]), 1),
        "co": round(float(r["karbon_monoksida"]), 1),
        "o3": round(float(r["ozon"]), 1),
    }


@st.cache_data(show_spinner=False)
def predict_dki_tanggal(d_iso, model_choice="xgboost"):
    """Rata-rata polutan 5 wilayah DKI pada tanggal tsb -> XGBoost -> kategori.
    None bila tidak ada satu pun wilayah berdata pada tanggal itu."""
    rows = [predict_wilayah_tanggal(w, d_iso, model_choice) for w in WILAYAH_DKI]
    rows = [r for r in rows if r is not None]
    if not rows:
        return None
    avg = {k: round(float(np.mean([r[k] for r in rows])), 1)
           for k in ["pm25", "pm10", "no2", "so2", "co", "o3"]}
    res = prediksi_ispu_xgboost(avg["pm10"], avg["pm25"], avg["so2"],
                                avg["co"], avg["o3"], avg["no2"], model_choice)
    out = {"kategori": res["kategori"],
           "ispu": int(round(float(np.mean([r["ispu"] for r in rows]))))}
    out.update(avg)
    return out


def rentang_tanggal(sel, mulai_offset, jumlah):
    """List tanggal: dari (sel+mulai_offset) sebanyak `jumlah` hari,
    dibatasi rentang tanggal yang tersedia di CSV."""
    lo, hi = rentang_data()
    hasil = []
    for i in range(jumlah):
        d = sel + timedelta(days=mulai_offset + i)
        if lo <= d <= hi:
            hasil.append(d)
    return hasil


def get_selected_date():
    """Tanggal terpilih global (default 15 Juni 2024, diklamp ke rentang CSV)."""
    lo, hi = rentang_data()
    if "sel_tanggal" not in st.session_state:
        default = date(2024, 6, 15)
        st.session_state["sel_tanggal"] = min(max(default, lo), hi)
    return st.session_state["sel_tanggal"]


# ── Format tanggal Bahasa Indonesia ──
BULAN_ID = {
    1: "Januari", 2: "Februari", 3: "Maret", 4: "April", 5: "Mei", 6: "Juni",
    7: "Juli", 8: "Agustus", 9: "September", 10: "Oktober", 11: "November",
    12: "Desember",
}
BULAN_ID_SINGKAT = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "Mei", 6: "Jun",
    7: "Jul", 8: "Agu", 9: "Sep", 10: "Okt", 11: "Nov", 12: "Des",
}


def tgl_id(d, singkat=False):
    """Format tanggal Bahasa Indonesia, mis. '15 Juni 2024' / '15 Jun 2024'."""
    bulan = (BULAN_ID_SINGKAT if singkat else BULAN_ID)[d.month]
    return f"{d.day:02d} {bulan} {d.year}"


def tgl_id_pendek(d):
    """Tanggal + bulan singkat tanpa tahun, mis. '09 Jun' (label chart)."""
    return f"{d.day:02d} {BULAN_ID_SINGKAT[d.month]}"


def render_date_picker():
    """Date picker kalender (rentang sesuai CSV) di pojok kanan header.
    Memakai SATU key global sehingga sinkron di semua halaman."""
    lo, hi = rentang_data()
    get_selected_date()  # pastikan ter-inisialisasi & diklamp
    st.markdown(
        "<div style='text-align:right; font-size:15px; font-weight:700; "
        "color:#4A6CF7; margin-bottom:6px; letter-spacing:0.01em;'>"
        "📅 Pilih Tanggal</div>",
        unsafe_allow_html=True,
    )
    return st.date_input(
        "Pilih Tanggal", key="sel_tanggal",
        min_value=lo, max_value=hi,
        format="DD/MM/YYYY", label_visibility="collapsed",
    )


def render_popup_polutan():
    """
    Popup "Informasi Polutan" - dipakai di Dashboard, Detail Wilayah,
    dan Simulasi Prediksi. Konten mengikuti gambar referensi POPUP.png.
    """
    @st.dialog("Informasi Polutan", width="small")
    def _popup():
        # Subjudul ringkas
        # Grid 2 kolom x 3 baris dibangun sebagai SATU blok HTML (bukan
        # st.columns) supaya compact, spacing presisi, dan tidak ada wrapper
        # kolom Streamlit yang menambah jarak.
        cards = ""
        for nama, info in INFO_POLUTAN.items():
            cards += (
                "<div style='border:1px solid #E5E7EB;border-radius:16px;"
                "background:#FFFFFF;padding:18px;'>"
                "<div style='color:#2563EB;font-weight:700;font-size:18px;"
                f"margin-bottom:6px;'>{nama}</div>"
                "<div style='color:#475569;font-size:13px;line-height:1.5;'>"
                f"{info['deskripsi']}</div>"
                "</div>"
            )
        st.markdown(
            "<div style='color:#64748B;font-size:14px;margin:-4px 0 16px;'>"
            "Penjelasan singkat tiap polutan udara yang dipantau JakU.</div>"
            "<div style='display:grid;grid-template-columns:1fr 1fr;"
            f"gap:16px;'>{cards}</div>",
            unsafe_allow_html=True,
        )

    _popup()


# ================================================================
# SIDEBAR
# ================================================================
def render_sidebar():
    with st.sidebar:
        # Logo lockup (ikon + wordmark + subtitle) sudah jadi satu di logo_jaku.svg.
        # Ukuran dijaga agar tidak selebar sidebar, diberi padding atas + jarak bawah
        # supaya tidak menempel/menumpuk dengan menu navigasi di bawahnya.
        LOGO_WIDTH_PX = 184
        logo_svg = ASSETS_DIR / "logo_jaku.svg"
        if logo_svg.exists():
            raw_svg = logo_svg.read_text(encoding="utf-8")
            raw_svg = re.sub(
                r'<svg\b',
                f'<svg style="width:{LOGO_WIDTH_PX}px; max-width:88%; '
                f'height:auto; display:block; margin:0 auto;"',
                raw_svg, count=1,
            )
            st.markdown(
                f'<div style="padding:8px 16px 0 16px; text-align:center;">{raw_svg}</div>'
                '<div style="height:70px;"></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div style="display:flex; align-items:center; justify-content:center;
                            gap:10px; padding:20px 0 3px 0;">
                    {logo_jaku_svg(size=42)}
                    <span style="font-size:30px; font-weight:800; letter-spacing:-0.02em;
                                 line-height:1;">
                        <span style="color:#0052A4;">Jak</span><span style="color:#19AE5D;">U</span>
                    </span>
                </div>
                <div class='sidebar-subtitle'>Pantau Udara, Jaga Jakarta</div>
                <div style="height:70px;"></div>
                """,
                unsafe_allow_html=True,
            )


        # Menu utama
        # ── Navigasi custom (st.button) supaya bisa pakai ikon SVG sendiri.
        #    option_menu hanya mendukung ikon Bootstrap & dirender di iframe,
        #    jadi tak bisa memakai SVG kustom. Ikon dipasang via CSS mask agar
        #    warnanya mengikuti status (abu-abu normal, biru saat aktif).
        NAV_ITEMS = [
            ("Dashboard", "label_dashboard.svg", "navdashboard"),
            ("Edukasi & Insight", "label_edukasi.svg", "navedukasi"),
            ("Detail Wilayah", "detail_wilayah.svg", "navdetail"),
            ("Simulasi Prediksi ISPU", "label_prediksi.svg", "navsimulasi"),
        ]
        if "nav_page" not in st.session_state:
            st.session_state["nav_page"] = "Dashboard"

        # CSS statis: gaya nav-link + ikon SVG per item (mask).
        css = "<style>"
        css += (
            # Jarak antar item menu (spasi ganda ala Word). Override
            # element-container margin-bottom:0 khusus item nav.
            'section[data-testid="stSidebar"] [class*="st-key-nav"]{'
            'margin-bottom:16px !important;}'
            'section[data-testid="stSidebar"] [class*="st-key-nav"] button{'
            'justify-content:flex-start !important;text-align:left;width:100%;'
            'background:#FFFFFF !important;border:none !important;box-shadow:none !important;'
            'color:#475569 !important;font-size:15px;font-weight:500;'
            'padding:11px 16px;border-radius:10px !important;transition:none !important;}'
            'section[data-testid="stSidebar"] [class*="st-key-nav"] button:hover{'
            'background:#F1F5F9 !important;color:#475569 !important;}'
            'section[data-testid="stSidebar"] [class*="st-key-nav"] button::before{'
            'content:"";display:inline-block;width:18px;height:18px;margin-right:12px;'
            'vertical-align:middle;flex-shrink:0;background-color:#4B5563;'
            '-webkit-mask-repeat:no-repeat;mask-repeat:no-repeat;'
            '-webkit-mask-position:center;mask-position:center;'
            '-webkit-mask-size:contain;mask-size:contain;}'
        )
        for _label, _fname, _key in NAV_ITEMS:
            _b64 = rekom_img_b64(_fname)
            if _b64:
                css += (
                    f'section[data-testid="stSidebar"] .st-key-{_key} button::before{{'
                    f'-webkit-mask-image:url("data:image/svg+xml;base64,{_b64}");'
                    f'mask-image:url("data:image/svg+xml;base64,{_b64}");}}'
                )
        css += "</style>"
        st.markdown(css, unsafe_allow_html=True)

        # Tombol-tombol navigasi
        for _label, _fname, _key in NAV_ITEMS:
            if st.button(_label, key=_key, use_container_width=True):
                st.session_state["nav_page"] = _label

        selected = st.session_state["nav_page"]

        # CSS state aktif (dirender SETELAH tombol → langsung sinkron dgn klik)
        _active_key = {l: k for l, _f, k in NAV_ITEMS}[selected]
        st.markdown(
            "<style>"
            f'section[data-testid="stSidebar"] .st-key-{_active_key} button,'
            f'section[data-testid="stSidebar"] .st-key-{_active_key} button:hover{{'
            "background:#DBEAFE !important;color:#2563EB !important;font-weight:600 !important;}"
            f'section[data-testid="stSidebar"] .st-key-{_active_key} button::before{{'
            "background-color:#2563EB !important;}"
            "</style>",
            unsafe_allow_html=True,
        )

        # Spacer untuk dorong footer ke bawah
        st.markdown("<div style='flex:1; min-height:96px;'></div>", unsafe_allow_html=True)

        # Footer sidebar
        st.markdown(
            """
            <div class='sidebar-footer'>
                <div class='sidebar-footer-title'>Data tidak realtime</div>
                <div class='sidebar-footer-desc'>
                    Hasil dihitung dari data sampel harian 2024 dan model XGBoost. Pilih tanggal di kanan atas untuk melihat prediksi.
                </div>
                <div class='sidebar-footer-ts-label'>Periode data</div>
                <div class='sidebar-footer-ts'>1 Jan – 31 Des 2024</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        return selected


# ================================================================
# ================================================================
# HALAMAN 1: DASHBOARD (REWRITE TOTAL — FIX #1, #2, #3, #5, #6, #7)
# ================================================================
def page_dashboard(data):
    """
    Perubahan dari versi lama:
    • Setiap "kartu" sekarang dibungkus st.container(border=True), BUKAN
      pasangan st.markdown("<div class='card'>") + </div>. Sebelumnya
      pattern itu menghasilkan div kosong (FIX #3) karena tiap st.markdown
      dibungkus DOM container terpisah oleh Streamlit.
    • Emoji status → ispu_emoji_svg(kategori) (FIX #5).
    • Ilustrasi kota → jakarta_skyline_svg() (FIX #6).
    • Legend peta → render_legend_safe() (FIX #1 + #2).
    • Tombol "Lihat Selengkapnya" dipindah ke bawah peta+legend dalam
      kartu yang SAMA (FIX #7).
    • Zoom peta 10 → 11 supaya fokus ke DKI Jakarta (FIX #7).
    """
    # ──────────────────────────── HEADER ────────────────────────────
    head1, head2 = st.columns([3, 1.1])
    with head1:
        st.markdown(
            "<div class='page-title'>Halo, Selamat Datang di JakU!</div>"
            "<div class='page-subtitle'>Berikut ringkasan kualitas udara di "
            "Provinsi DKI Jakarta</div>",
            unsafe_allow_html=True,
        )
    with head2:
        sel_tgl = render_date_picker()

    # ──────────────── ROW 1: HERO ISPU + PETA WILAYAH ────────────────
    # Tinggi tetap kartu per baris supaya dua kartu sebaris SEJAJAR bawahnya.
    # Cara CSS equal-height tidak andal lintas versi Streamlit, jadi dipakai
    # tinggi tetap. >>> Kalau ada kartu yang muncul scrollbar di dalamnya,
    # NAIKKAN angkanya; kalau terlalu banyak ruang kosong, TURUNKAN. <<<
    DASH_ROW1_H = 620   # baris 1: "Kualitas Udara hari ini" & "per Wilayah"
    DASH_ROW2_H = 440   # baris 2: "Prediksi ISPU" & "Tren ISPU"
    col_left, col_right = st.columns([1.18, 1], gap="medium")

    # ─── KIRI: Hero ISPU ───
    with col_left:
        with st.container(border=True, height=DASH_ROW1_H):   # tinggi tetap → sejajar
            dki_today = predict_dki_tanggal(sel_tgl.isoformat())
            if dki_today is None:
                dki_today = {"kategori": "Sedang", "ispu": 0, "pm25": "-",
                             "pm10": "-", "no2": "-", "so2": "-",
                             "co": "-", "o3": "-"}
            ispu_avg = dki_today["ispu"]
            kat = dki_today["kategori"]
            info = KATEGORI_INFO[kat]

            tgl_label = tgl_id(sel_tgl)
            st.markdown(
                "<div class='card-title'>Kualitas Udara di Jakarta "
                f"({tgl_label})</div>",
                unsafe_allow_html=True,
            )

            # Hero layout: pakai Streamlit columns [2, 1] untuk presisi.
            # Sebelumnya pakai single markdown dengan flex 3-child → ilustrasi
            # tidak konsisten posisinya.
            hero_main, hero_illust = st.columns([2.4, 1], gap="small")

            with hero_main:
                # SATU markdown: angka 78 (kiri) + emoji SVG/status/desc (kanan)
                # dengan flex inline, predictable height.
                st.markdown(
                    "<div style='display:flex; align-items:flex-start; "
                    "gap:24px; margin-top:4px;'>"
                    # Kolom kiri: angka ISPU + label
                    "<div style='flex-shrink:0;'>"
                    f"<div style='font-size:80px; font-weight:800; "
                    f"line-height:0.95; letter-spacing:-0.05em; "
                    f"color:{info['warna']};'>{ispu_avg}</div>"
                    "<div style='font-size:15px; font-weight:600; "
                    "color:#64748B; margin-top:5px;'>ISPU</div>"
                    "</div>"
                    # Kolom kanan: emoji SVG + status + deskripsi
                    "<div style='flex:1; padding-top:5px;'>"
                    f"<div style='margin-bottom:9px;'>"
                    f"{ispu_emoji_svg(kat, size=52)}</div>"
                    f"<div style='font-size:22px; font-weight:700; "
                    f"color:{info['warna']}; margin-bottom:6px;'>"
                    f"Udara {kat}</div>"
                    "<div style='font-size:14px; color:#475569; "
                    "line-height:1.55;'>"
                    f"{info['deskripsi']}</div>"
                    "</div>"
                    "</div>",
                    unsafe_allow_html=True,
                )

            with hero_illust:
                # Ilustrasi Jakarta dari PNG (assets/ilustrasi_jakarta.png).
                # Fallback ke SVG bawaan bila file tidak ditemukan.
                ilust_b64 = rekom_img_b64("ilustrasi_jakarta.png")
                if ilust_b64:
                    st.markdown(
                        "<div style='text-align:center; padding-top:2px;'>"
                        "<div style='background:#FFFFFF; border:1px solid #EEF2F7; "
                        "border-radius:14px; padding:8px; display:inline-block; "
                        "box-shadow:0 1px 3px rgba(15,23,42,0.04);'>"
                        f"<img src='data:image/png;base64,{ilust_b64}' "
                        "style='width:150px; height:auto; border-radius:8px; display:block;'/>"
                        "</div>"
                        "<div style='font-size:13px; color:#64748B; font-weight:500; "
                        "margin-top:5px;'>DKI Jakarta</div>"
                        "</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        "<div style='text-align:center; padding-top:6px;'>"
                        f"{jakarta_skyline_svg(width=180)}"
                        "<div style='font-size:13px; color:#64748B; "
                        "font-weight:500; margin-top:3px;'>DKI Jakarta</div>"
                        "</div>",
                        unsafe_allow_html=True,
                    )

            # ─── Polutan dominan (kiri) + tombol "Lihat penjelasan polutan"
            #     (kanan mentok, sejajar satu baris) ───
            st.markdown(
                "<div style='margin-top:12px;'></div>",
                unsafe_allow_html=True,
            )

            try:
                pdc1, pdc2 = st.columns([2, 1.15], vertical_alignment="center")
            except TypeError:
                # Fallback untuk Streamlit < 1.36 yang tidak punya vertical_alignment
                pdc1, pdc2 = st.columns([2, 1.15])

            with pdc1:
                st.markdown(
                    "<div style='display:flex; align-items:center; "
                    "gap:8px; font-size:15px; color:#0F172A;'>"
                    + LEAF_ICON_SVG +
                    "<span><strong>Polutan dominan:</strong>&nbsp; "
                    f"PM2.5 ({dki_today['pm25']} µg/m³)</span>"
                    "</div>",
                    unsafe_allow_html=True,
                )
            with pdc2:
                # use_container_width → tombol mengisi kolom kanan → posisinya
                # mentok ke kanan kartu, sejajar dengan teks polutan dominan.
                if st.button("ⓘ  Lihat penjelasan polutan",
                             key="btn_info_dashboard", use_container_width=True):
                    render_popup_polutan()

            # 6 polutan compact — SATU markdown call (nilai per tanggal terpilih)
            st.markdown(
                "<div class='pollutant-grid'>"
                "<div class='pollutant-cell'><div class='pollutant-name'>PM2.5</div>"
                f"<div class='pollutant-value'>{dki_today['pm25']}</div>"
                "<div class='pollutant-unit'>µg/m³</div></div>"
                "<div class='pollutant-cell'><div class='pollutant-name'>PM10</div>"
                f"<div class='pollutant-value'>{dki_today['pm10']}</div>"
                "<div class='pollutant-unit'>µg/m³</div></div>"
                "<div class='pollutant-cell'><div class='pollutant-name'>NO₂</div>"
                f"<div class='pollutant-value'>{dki_today['no2']}</div>"
                "<div class='pollutant-unit'>µg/m³</div></div>"
                "<div class='pollutant-cell'><div class='pollutant-name'>SO₂</div>"
                f"<div class='pollutant-value'>{dki_today['so2']}</div>"
                "<div class='pollutant-unit'>µg/m³</div></div>"
                "<div class='pollutant-cell'><div class='pollutant-name'>CO</div>"
                f"<div class='pollutant-value'>{dki_today['co']}</div>"
                "<div class='pollutant-unit'>mg/m³</div></div>"
                "<div class='pollutant-cell'><div class='pollutant-name'>O₃</div>"
                f"<div class='pollutant-value'>{dki_today['o3']}</div>"
                "<div class='pollutant-unit'>µg/m³</div></div>"
                "</div>",
                unsafe_allow_html=True,
            )

            # ─── Rekomendasi Aktivitas (gambar PNG dari assets sesuai
            #     kategori rata-rata; ISPU 78 → "Sedang" → rekom_sedang.png) ───
            rekom_b64 = rekom_img_b64(REKOM_IMG.get(kat, ""))
            if rekom_b64:
                st.markdown(
                    f"<img src='data:image/png;base64,{rekom_b64}' alt='Rekomendasi Aktivitas' "
                    f"style='width:100%; max-width:620px; height:auto; display:block; margin:18px auto 0;'/>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div style="background:{info['warna_bg']}; border:1.5px solid {info['warna']};
                                border-radius:16px; padding:18px 20px; margin-top:18px;">
                        <div style="font-size:16px; font-weight:700; color:{info['warna']}; margin-bottom:8px;">
                            Rekomendasi Aktivitas
                        </div>
                        <div style="font-size:13.5px; color:#1E293B; line-height:1.55;">{info['rekomendasi']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    # ─── KANAN: Peta wilayah + daftar status + legend + tombol ───
    with col_right:
        with st.container(border=True, height=DASH_ROW1_H):   # tinggi tetap → sejajar
            st.markdown(
                "<div class='card-title'>Kualitas Udara per Wilayah di Jakarta</div>",
                unsafe_allow_html=True,
            )

            # Snapshot per wilayah untuk tanggal terpilih (lat/lon dari data
            # statis; ispu/kategori dari prediksi XGBoost atas data CSV asli).
            # Wilayah tanpa data di CSV (mis. Kep. Seribu) -> ada=False.
            wil_today = data["wilayah"][["wilayah", "lat", "lon"]].copy()
            _pred = {w: predict_wilayah_tanggal(w, sel_tgl.isoformat())
                     for w in wil_today["wilayah"]}
            wil_today["ada"] = wil_today["wilayah"].map(lambda w: _pred[w] is not None)
            wil_today["ispu"] = wil_today["wilayah"].map(
                lambda w: _pred[w]["ispu"] if _pred[w] else None)
            wil_today["kategori"] = wil_today["wilayah"].map(
                lambda w: _pred[w]["kategori"] if _pred[w] else None)

            # Peta (kiri ~65%) + daftar status wilayah & legend (kanan ~35%)
            # MAP_H dipakai bersama: tinggi peta DAN tinggi kotak kolom kanan
            # dibuat sama persis supaya kedua kolom sejajar vertikal & tidak
            # ada whitespace di bawah (figma: tinggi peta ~420-450px).
            MAP_H = 450
            mc1, mc2 = st.columns([1.85, 1], gap="medium")
            with mc1:
                # Peta dengan tile berwarna (CartoDB Voyager) seperti desain.
                m = folium.Map(
                    location=[-6.17, 106.83],
                    zoom_start=11,
                    tiles=None,
                    zoom_control=True,
                    scrollWheelZoom=True,
                    dragging=True,
                    min_zoom=9,
                    max_zoom=16,
                )
                folium.TileLayer(
                    tiles="https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png",
                    attr="&copy; OpenStreetMap contributors &copy; CARTO",
                    name="Voyager",
                    control=False,
                ).add_to(m)

                # Tombol maximize/minimize (layar penuh) di pojok kanan atas.
                Fullscreen(
                    position="topright",
                    title="Perbesar peta",
                    title_cancel="Perkecil peta",
                    force_separate_button=True,
                ).add_to(m)

                # Koordinat tampilan: Kep. Seribu (lokasi asli jauh di utara,
                # ~-5.75) digeser ke area teluk utara Jakarta agar lingkarannya
                # tetap tampil dalam peta DKI seperti desain. Data asli tak diubah.
                display_coords = {"Kep. Seribu": (-6.03, 106.80)}

                bounds = []
                for _, row in wil_today.iterrows():
                    if not row["ada"]:
                        continue  # wilayah tanpa data (mis. Kep. Seribu) → tanpa marker
                    lat, lon = display_coords.get(
                        row["wilayah"], (row["lat"], row["lon"])
                    )
                    kat_w = row["kategori"]
                    warna = KATEGORI_INFO.get(
                        kat_w, KATEGORI_INFO["Sedang"]
                    )["warna"]
                    bounds.append([lat, lon])
                    # Lingkaran berwarna kategori
                    folium.CircleMarker(
                        location=[lat, lon],
                        radius=18,
                        color="white",
                        weight=2.5,
                        fill=True,
                        fillColor=warna,
                        fillOpacity=0.95,
                        tooltip=f"{row['wilayah']}: {row['ispu']} ({kat_w})",
                    ).add_to(m)
                    # Label skor ISPU di tengah lingkaran
                    folium.map.Marker(
                        [lat, lon],
                        icon=folium.DivIcon(
                            icon_size=(36, 36),
                            icon_anchor=(18, 18),
                            html=(
                                "<div style='font-size:12px; font-weight:800; "
                                "color:white; text-align:center; "
                                f"line-height:36px;'>{row['ispu']}</div>"
                            ),
                        ),
                    ).add_to(m)
                if bounds:
                    m.fit_bounds(bounds, padding=(30, 30))
                # PENTING: st_folium + use_container_width membuat tinggi tile
                # leaflet mengikuti RASIO ASPEK lebar kolom, BUKAN parameter
                # height -> menyisakan ruang putih di bawah frame. Solusinya:
                # bungkus map ke branca.Figure dengan tinggi tetap (px) lalu
                # render via components.html. Karena html/body figure
                # ber-height:100%, tile leaflet mengisi PENUH tinggi frame.
                fig = Figure(width="100%", height=f"{MAP_H}px")
                fig.add_child(m)
                components.html(fig.render(), height=MAP_H, scrolling=False)

            with mc2:
                # Kolom kanan dibuat SATU kotak flex setinggi peta (MAP_H).
                # Daftar wilayah di atas (didorong sedikit ke bawah via
                # padding-top), legend "Keterangan" di-push ke bawah dengan
                # margin-top:auto -> mengisi penuh & menghilangkan whitespace
                # bawah, tepi bawah legend sejajar dengan tepi bawah peta.

                # Baris daftar wilayah: nama abu gelap, status bold berwarna
                # sesuai kategori.
                rows_html = ""
                for _, row in wil_today.iterrows():
                    if row["ada"]:
                        c = KATEGORI_INFO.get(
                            row["kategori"], KATEGORI_INFO["Sedang"]
                        )["warna"]
                        status_html = (
                            f"<span style='color:{c};font-weight:700;'>"
                            f"{row['kategori']}</span>"
                        )
                    else:
                        status_html = (
                            "<span style='color:#94A3B8;font-weight:600;"
                            "font-style:italic;'>Tidak ada data</span>"
                        )
                    rows_html += (
                        "<div style='font-size:15px;color:#475569;"
                        "line-height:1.4;margin-bottom:10px;'>"
                        f"{row['wilayah']}: {status_html}</div>"
                    )

                # Legend kategori (vertikal, bulatan warna kiri, nama bold).
                leg_rows = ""
                for nama, info in KATEGORI_INFO.items():
                    leg_rows += (
                        "<div style='display:flex;align-items:center;gap:10px;"
                        "margin:7px 0;font-size:15px;color:#334155;'>"
                        "<span style='width:13px;height:13px;border-radius:50%;"
                        f"background:{info['warna']};display:inline-block;"
                        "flex-shrink:0;'></span>"
                        "<span><strong style='color:#0F172A;font-weight:700;'>"
                        f"{nama}</strong> ({info['rentang']})</span>"
                        "</div>"
                    )

                mc2_html = (
                    f"<div style='height:{MAP_H}px;display:flex;"
                    "flex-direction:column;padding-top:24px;"
                    "box-sizing:border-box;'>"
                    f"<div>{rows_html}</div>"
                    "<div style='margin-top:auto;'>"
                    "<div style='font-weight:700;font-size:16px;color:#0F172A;"
                    "margin-bottom:14px;'>Keterangan:</div>"
                    f"{leg_rows}"
                    "</div>"
                    "</div>"
                )
                st.markdown(mc2_html, unsafe_allow_html=True)

            # Tombol "Lihat Selengkapnya" di bawah peta (outline pill, kiri,
            # auto-width agar kompak seperti desain Figma).
            st.markdown("<div style='margin-top:10px;'></div>",
                        unsafe_allow_html=True)
            if st.button("Lihat Selengkapnya  →",
                         key="btn_selengkapnya",
                         use_container_width=False):
                st.session_state["jump_to_detail"] = True
                st.rerun()

    # ──────────────── ROW 2: PREDIKSI + TREN ────────────────
    st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
    pcol1, pcol2 = st.columns([1, 1.4], gap="medium")

    # ─── Prediksi 7 hari mendatang ───
    with pcol1:
        with st.container(border=True, height=DASH_ROW2_H):   # tinggi tetap → sejajar
            st.markdown(
                "<div class='card-title'>Prediksi ISPU di Jakarta "
                "(7 Hari Mendatang)</div>",
                unsafe_allow_html=True,
            )
            rows_html = ""
            for d in rentang_tanggal(sel_tgl, 1, 7):
                p = predict_dki_tanggal(d.isoformat())
                if p is None:
                    continue
                kat2 = p["kategori"]
                warna = KATEGORI_INFO.get(kat2, KATEGORI_INFO["Sedang"])["warna"]
                tanggal = tgl_id(d, singkat=True)
                rows_html += (
                    "<div class='pred-row'>"
                    f"<div class='pred-date'>{tanggal}</div>"
                    "<div>"
                    f"<span class='pred-pill' style='background:{warna};'>"
                    f"{p['ispu']}</span>"
                    "</div>"
                    f"<div class='pred-cat' style='color:{warna};'>{kat2}</div>"
                    f"<div class='pred-pm'>PM2.5 ({p['pm25']} µg/m³)</div>"
                    "</div>"
                )
            st.markdown(rows_html, unsafe_allow_html=True)

    # ─── Tren 7 hari terakhir (chart) ───
    with pcol2:
        with st.container(border=True, height=DASH_ROW2_H):   # tinggi tetap → sejajar
            st.markdown(
                "<div class='card-title'>Tren ISPU di Jakarta (7 Hari Terakhir)</div>",
                unsafe_allow_html=True,
            )

            _tgl_tren = rentang_tanggal(sel_tgl, -6, 7)  # 7 hari s.d. tanggal terpilih
            _pts = [(d, predict_dki_tanggal(d.isoformat())) for d in _tgl_tren]
            _pts = [(d, p) for d, p in _pts if p is not None]
            df_tren = pd.DataFrame({
                "tanggal": [pd.Timestamp(d) for d, _ in _pts],
                "ispu": [p["ispu"] for _, p in _pts],
            })
            df_tren["label_x"] = df_tren["tanggal"].apply(tgl_id_pendek)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_tren["label_x"], y=df_tren["ispu"],
                mode="lines+markers+text",
                text=df_tren["ispu"],
                textposition="top center",
                textfont=dict(size=11, color="#0F172A", weight=600),
                line=dict(color="#2563EB", width=3,
                          shape="spline", smoothing=1.0),
                marker=dict(size=9, color="#2563EB",
                            line=dict(color="white", width=2)),
                fill="tozeroy",
                fillcolor="rgba(37, 99, 235, 0.08)",
                hovertemplate="<b>%{x}</b><br>ISPU: %{y}<extra></extra>",
                showlegend=False,
            ))
            # Garis & label threshold kategori (3 kategori). Posisi label =
            # penanda zona kategori agar mudah dibaca, bukan batas presisi.
            for nilai, label, warna in [
                (50, "Baik", "#16A34A"),
                (100, "Sedang", "#2563EB"),
                (150, "Tidak Sehat", "#E5B93D"),
            ]:
                fig.add_hline(y=nilai, line_dash="dot",
                              line_color="#E2E8F0", line_width=1)
                fig.add_annotation(
                    x=1.0, xref="paper", y=nilai,
                    text=label, showarrow=False,
                    xanchor="left", yanchor="middle",
                    font=dict(size=10, color=warna, weight=600),
                    xshift=8,
                )
            fig.update_layout(
                # FIX — margin lebih besar di kiri/kanan/atas supaya label "62"
                # (di awal) dan "78" (di akhir) tidak terpotong; t=45 supaya
                # angka di atas marker tidak nyentuh batas card.
                height=340,
                margin=dict(l=40, r=140, t=50, b=30),
                paper_bgcolor="white", plot_bgcolor="white",
                xaxis=dict(
                    showgrid=False, showline=False,
                    tickfont=dict(size=11, color="#64748B"),
                    # Padding kiri-kanan: extend domain agar marker awal/akhir
                    # punya breathing room untuk label
                    range=[-0.4, 6.4],
                ),
                yaxis=dict(
                    range=[0, 200], gridcolor="#F1F5F9", showline=False,
                    tickfont=dict(size=11, color="#94A3B8"),
                    tickvals=[0, 50, 100, 150, 200, 250, 300],
                ),
            )
            st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar": False})

    # ──────────────── INFO BOX ML ────────────────
    st.markdown(
        "<div class='info-box'>"
        "<div class='info-box-icon'>ⓘ</div>"
        "<div class='info-box-text'>"
        "Prediksi ini dibuat menggunakan model machine learning "
        "<strong>XGBoost</strong> berdasarkan data historis ISPU pada tahun 2024."
        "</div></div>",
        unsafe_allow_html=True,
    )




# ================================================================
# HALAMAN 2: DETAIL WILAYAH
# ================================================================
# Wilayah yang tab-nya tetap tampil, tetapi datanya belum tersedia.
# Untuk wilayah ini Detail Wilayah merender "empty state" (lihat
# render_empty_state_wilayah) sesuai mockup Figma, alih-alih kartu detail.
WILAYAH_TANPA_DATA = {"Kep. Seribu"}


def render_empty_state_wilayah(wilayah):
    """
    Empty state untuk wilayah yang data ISPU-nya belum tersedia (mis. Kep. Seribu).
    Mengikuti mockup Figma: satu kartu berisi ikon lingkaran-silang biru,
    judul 'Data Kualitas Udara Belum Tersedia', dan keterangan singkat.
    """
    circle_x_svg = (
        '<svg width="84" height="84" viewBox="0 0 84 84" fill="none" '
        'xmlns="http://www.w3.org/2000/svg" style="display:block;">'
        '<circle cx="42" cy="42" r="33" stroke="#2563EB" stroke-width="3.5" fill="none"/>'
        '<path d="M31 31 L53 53 M53 31 L31 53" stroke="#2563EB" '
        'stroke-width="3.5" stroke-linecap="round"/>'
        '</svg>'
    )
    with st.container(border=True):
        st.markdown(
            f"<div class='card-title'>Kualitas Udara {wilayah}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div style="display:flex; flex-direction:column; align-items:center;
                        justify-content:center; text-align:center;
                        padding:60px 20px 70px;">
                <div style="margin-bottom:24px;">{circle_x_svg}</div>
                <div style="font-size:18px; font-weight:700; color:#0F172A; margin-bottom:9px;">
                    Data Kualitas Udara Belum Tersedia
                </div>
                <div style="font-size:13.5px; color:#94A3B8; max-width:540px; line-height:1.55;">
                    Data ISPU untuk wilayahnya saat ini belum tersedia atau belum diperbarui.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def page_detail_wilayah(data):
    head1, head2 = st.columns([3, 1.1])
    with head1:
        st.markdown(
            "<div class='page-title'>Detail Wilayah</div>"
            "<div class='page-subtitle'>Pilih wilayah untuk melihat informasi kualitas udara lebih detail.</div>",
            unsafe_allow_html=True,
        )
    with head2:
        sel_tgl = render_date_picker()

    # Tabs wilayah
    wilayah_list = data["wilayah"]["wilayah"].tolist()

    # Ikon penunjuk per-wilayah (SVG Figma) ditanam langsung sebagai base64,
    # lalu disuntik ke tiap pill tab berdasarkan urutannya. Tidak bergantung
    # pada file di folder assets/ sehingga selalu tampil.
    _ICON_B64 = {
        "Jakarta Pusat":   "PHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPGNpcmNsZSBjeD0iOCIgY3k9IjgiIHI9IjgiIGZpbGw9IiMxOUFFNUQiLz4KPC9zdmc+Cg==",
        "Jakarta Utara":   "PHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTQiIHZpZXdCb3g9IjAgMCAxNiAxNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTEzLjY1NyAtMC4wMDAxNTc0OTZDMTQuNzc1OCAxLjExODY3IDE1LjUzNzcgMi41NDQxNCAxNS44NDY0IDQuMDk2QzE2LjE1NTEgNS42NDc4NiAxNS45OTY2IDcuMjU2NDEgMTUuMzkxMSA4LjcxODIzQzE0Ljc4NTYgMTAuMTggMTMuNzYwMyAxMS40Mjk1IDEyLjQ0NDcgMTIuMzA4NUMxMS4xMjkxIDEzLjE4NzYgOS41ODIzOSAxMy42NTY4IDguMDAwMTUgMTMuNjU2OEM2LjQxNzkxIDEzLjY1NjggNC44NzEyIDEzLjE4NzYgMy41NTU2MiAxMi4zMDg1QzIuMjQwMDQgMTEuNDI5NSAxLjIxNDY2IDEwLjE4IDAuNjA5MTY2IDguNzE4MjNDMC4wMDM2Njk3NyA3LjI1NjQxIC0wLjE1NDc1NiA1LjY0Nzg2IDAuMTUzOTI0IDQuMDk2QzAuNDYyNjAzIDIuNTQ0MTQgMS4yMjQ1MiAxLjExODY3IDIuMzQzMzQgLTAuMDAwMTU3NDk2TDguMDAwMTUgNS42NTY3NEwxMy42NTcgLTAuMDAwMTU3NDk2WiIgZmlsbD0iIzE5QUU1RCIvPgo8L3N2Zz4K",
        "Jakarta Barat":   "PHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTggMC4wMDAxMTk3MzVDOS41ODIyNSAwLjAwMDE0MzIyNyAxMS4xMjkgMC40NjkzNTggMTIuNDQ0NiAxLjM0ODQzQzEzLjc2MDIgMi4yMjc1IDE0Ljc4NTUgMy40NzY5NCAxNS4zOTEgNC45Mzg3NkMxNS45OTY1IDYuNDAwNTggMTYuMTU1IDguMDA5MTEgMTUuODQ2MyA5LjU2MDk2QzE1LjUzNzYgMTEuMTEyOCAxNC43NzU3IDEyLjUzODMgMTMuNjU2OSAxMy42NTcxQzEyLjUzOCAxNC43NzU5IDExLjExMjYgMTUuNTM3OCA5LjU2MDcyIDE1Ljg0NjRDOC4wMDg4NyAxNi4xNTUxIDYuNDAwMzQgMTUuOTk2NiA0LjkzODUzIDE1LjM5MTFDMy40NzY3MiAxNC43ODU2IDIuMjI3MjkgMTMuNzYwMiAxLjM0ODI0IDEyLjQ0NDZDMC40NjkxOTIgMTEuMTI5IDUuNzY2MjdlLTA4IDkuNTgyMjUgOS41Mzk5ZS0wOCA4TDggOC4wMDAxMkw4IDAuMDAwMTE5NzM1WiIgZmlsbD0iIzE5QUU1RCIvPgo8L3N2Zz4K",
        "Jakarta Selatan": "PHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTQiIHZpZXdCb3g9IjAgMCAxNiAxNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTIuMzQzIDEzLjY1NjhDMS4yMjQyIDEyLjUzNzkgMC40NjIyODUgMTEuMTEyNSAwLjE1MzYxOCA5LjU2MDU5Qy0wLjE1NTA1MSA4LjAwODczIDAuMDAzMzg3OTEgNi40MDAxOCAwLjYwODg5NSA0LjkzODM3QzEuMjE0NCAzLjQ3NjU1IDIuMjM5NzggMi4yMjcxMiAzLjU1NTM3IDEuMzQ4MDhDNC44NzA5NiAwLjQ2OTAyOCA2LjQxNzY3IC0wLjAwMDE1NTgxNiA3Ljk5OTkxIC0wLjAwMDE0Mzc3N0M5LjU4MjE1IC0wLjAwMDEzMTgyMyAxMS4xMjg5IDAuNDY5MDc1IDEyLjQ0NDQgMS4zNDgxNEMxMy43NiAyLjIyNzIxIDE0Ljc4NTQgMy40NzY2NiAxNS4zOTA5IDQuOTM4NDhDMTUuOTk2MyA2LjQwMDMgMTYuMTU0OCA4LjAwODg1IDE1Ljg0NjEgOS41NjA3MUMxNS41Mzc0IDExLjExMjYgMTQuNzc1NCAxMi41MzggMTMuNjU2NiAxMy42NTY5TDcuOTk5ODUgNy45OTk5MkwyLjM0MyAxMy42NTY4WiIgZmlsbD0iIzE5QUU1RCIvPgo8L3N2Zz4K",
        "Jakarta Timur":   "PHN2ZyB3aWR0aD0iMTQiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNCAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTEzLjY1NjggMTMuNjU3QzEyLjUzNzkgMTQuNzc1OCAxMS4xMTI1IDE1LjUzNzcgOS41NjA1OSAxNS44NDY0QzguMDA4NzMgMTYuMTU1MSA2LjQwMDE5IDE1Ljk5NjYgNC45MzgzNyAxNS4zOTExQzMuNDc2NTYgMTQuNzg1NiAyLjIyNzEzIDEzLjc2MDIgMS4zNDgwOCAxMi40NDQ2QzAuNDY5MDI5IDExLjEyOSAtMC4wMDAxNTQ3MjkgOS41ODIzMyAtMC4wMDAxNDI4MjQgOC4wMDAwOUMtMC4wMDAxMzEwMDIgNi40MTc4NSAwLjQ2OTA3NiA0Ljg3MTE1IDEuMzQ4MTQgMy41NTU1N0MyLjIyNzIxIDIuMjM5OTkgMy40NzY2NiAxLjIxNDYzIDQuOTM4NDggMC42MDkxNDRDNi40MDAzMSAwLjAwMzY1NzkxIDguMDA4ODUgLTAuMTU0NzU2IDkuNTYwNzEgMC4xNTM5MzVDMTEuMTEyNiAwLjQ2MjYyNiAxMi41MzggMS4yMjQ1NiAxMy42NTY5IDIuMzQzMzhMNy45OTk5MiA4LjAwMDE1TDEzLjY1NjggMTMuNjU3WiIgZmlsbD0iIzE5QUU1RCIvPgo8L3N2Zz4K",
        "Kep. Seribu":     "PHN2ZyB3aWR0aD0iMTQiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNCAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTAuMDAwNDU1MjQgMi4zNDNDMS4xMTkyOSAxLjIyNDIgMi41NDQ3NyAwLjQ2MjI4NiA0LjA5NjYzIDAuMTUzNjE4QzUuNjQ4NDkgLTAuMTU1MDUxIDcuMjU3MDQgMC4wMDMzODc2NCA4LjcxODg1IDAuNjA4ODk0QzEwLjE4MDcgMS4yMTQ0IDExLjQzMDEgMi4yMzk3OCAxMi4zMDkxIDMuNTU1MzdDMTMuMTg4MiA0Ljg3MDk2IDEzLjY1NzQgNi40MTc2NyAxMy42NTc0IDcuOTk5OTFDMTMuNjU3NCA5LjU4MjE1IDEzLjE4ODEgMTEuMTI4OSAxMi4zMDkxIDEyLjQ0NDRDMTEuNDMgMTMuNzYgMTAuMTgwNiAxNC43ODU0IDguNzE4NzUgMTUuMzkwOUM3LjI1NjkyIDE1Ljk5NjMgNS42NDgzOCAxNi4xNTQ4IDQuMDk2NTIgMTUuODQ2MUMyLjU0NDY2IDE1LjUzNzQgMS4xMTkxOSAxNC43NzU0IDAuMDAwMzcyMjcgMTMuNjU2Nkw1LjY1NzMxIDcuOTk5ODVMMC4wMDA0NTUyNCAyLjM0M1oiIGZpbGw9IiMxOUFFNUQiLz4KPC9zdmc+Cg==",
    }
    icon_css = "<style>"
    for i, w in enumerate(wilayah_list, start=1):
        b64 = _ICON_B64.get(w, "")
        if b64:
            icon_css += (
                f'.stTabs [data-baseweb="tab"]:nth-of-type({i})::before,'
                f'.stTabs [data-baseweb="tab"]:nth-child({i})::before,'
                f'.stTabs [data-testid="stTab"]:nth-of-type({i})::before,'
                f'.stTabs [data-testid="stTab"]:nth-child({i})::before{{'
                f'background-image:url("data:image/svg+xml;base64,{b64}");}}'
            )
    icon_css += "</style>"
    st.markdown(icon_css, unsafe_allow_html=True)

    tabs = st.tabs(wilayah_list)

    for tab, wilayah in zip(tabs, wilayah_list):
        with tab:
            # Wilayah tanpa data → tampilkan empty state (sesuai mockup Figma)
            if wilayah in WILAYAH_TANPA_DATA:
                render_empty_state_wilayah(wilayah)
                continue

            row = predict_wilayah_tanggal(wilayah, sel_tgl.isoformat())
            if row is None:
                # Tidak ada data untuk wilayah ini pada tanggal terpilih
                render_empty_state_wilayah(wilayah)
                continue
            kat = row["kategori"]
            info = KATEGORI_INFO[kat]

            # Kualitas udara + Rekomendasi Aktivitas dalam SATU kartu (sesuai Figma):
            # kiri = hero ISPU + polutan dominan + grid polutan; kanan = kotak
            # Rekomendasi Aktivitas berupa gambar PNG dari assets sesuai kategori.
            with st.container(border=True):
                st.markdown(
                    f"<div class='card-title'>Kualitas Udara {wilayah} "
                    f"({tgl_id(sel_tgl)})</div>",
                    unsafe_allow_html=True,
                )

                left_col, right_col = st.columns([1, 1], gap="large")

                # ── Kiri: hero + polutan dominan + grid polutan ──
                with left_col:
                    st.markdown(
                        f"""
                        <div style="display:flex; align-items:center; gap:20px;
                                    margin-top:4px; flex-wrap:wrap;">
                            <div style="flex-shrink:0;">
                                <div class='ispu-number' style='color:{info["warna"]};'>{row["ispu"]}</div>
                                <div class='ispu-label' style='text-align:center;'>ISPU</div>
                            </div>
                            <div class='ispu-emoji' style='margin-bottom:0;'>{ispu_emoji_svg(kat, size=56)}</div>
                            <div style="flex:1; min-width:160px;">
                                <div class='ispu-status' style='color:{info["warna"]};'>Udara {kat}</div>
                                <div class='ispu-desc'>{info["deskripsi"]}</div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    pdc1, pdc2 = st.columns([1, 1])
                    with pdc1:
                        st.markdown(
                            f"""
                            <div style='display:flex; align-items:center; gap:8px;
                                        padding-top:16px; font-size:14px; color:#0F172A;'>
                                {LEAF_ICON_SVG}
                                <span><strong>Polutan dominan:</strong> PM2.5 ({row["pm25"]} µg/m³)</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    with pdc2:
                        st.markdown("<div style='padding-top:10px;'></div>", unsafe_allow_html=True)
                        if st.button("ⓘ Lihat penjelasan polutan", key=f"btn_info_{wilayah}", use_container_width=True):
                            render_popup_polutan()

                    st.markdown(
                        f"""
                        <div class='pollutant-grid' style='margin-bottom:16px;'>
                          <div class='pollutant-cell'><div class='pollutant-name'>PM2.5</div><div class='pollutant-value'>{row["pm25"]}</div><div class='pollutant-unit'>µg/m³</div></div>
                          <div class='pollutant-cell'><div class='pollutant-name'>PM10</div><div class='pollutant-value'>{row["pm10"]}</div><div class='pollutant-unit'>µg/m³</div></div>
                          <div class='pollutant-cell'><div class='pollutant-name'>NO₂</div><div class='pollutant-value'>{row["no2"]}</div><div class='pollutant-unit'>µg/m³</div></div>
                          <div class='pollutant-cell'><div class='pollutant-name'>SO₂</div><div class='pollutant-value'>{row["so2"]}</div><div class='pollutant-unit'>µg/m³</div></div>
                          <div class='pollutant-cell'><div class='pollutant-name'>CO</div><div class='pollutant-value'>{row["co"]}</div><div class='pollutant-unit'>mg/m³</div></div>
                          <div class='pollutant-cell'><div class='pollutant-name'>O₃</div><div class='pollutant-value'>{row["o3"]}</div><div class='pollutant-unit'>µg/m³</div></div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # ── Kanan: Rekomendasi Aktivitas (PNG dari assets sesuai kategori) ──
                with right_col:
                    rekom_b64 = rekom_img_b64(REKOM_IMG.get(kat, ""))
                    if rekom_b64:
                        st.markdown(
                            f"<img src='data:image/png;base64,{rekom_b64}' alt='Rekomendasi Aktivitas' "
                            f"style='width:100%; max-width:620px; height:auto; display:block; margin-top:8px;'/>",
                            unsafe_allow_html=True,
                        )
                    else:
                        # Fallback teks bila gambar tidak ditemukan
                        st.markdown(
                            f"""
                            <div style="background:{info['warna_bg']}; border:1.5px solid {info['warna']};
                                        border-radius:16px; padding:18px 20px; margin-top:8px;">
                                <div style="font-size:16px; font-weight:700; color:{info['warna']}; margin-bottom:8px;">
                                    Rekomendasi Aktivitas
                                </div>
                                <div style="font-size:13.5px; color:#1E293B; line-height:1.55;">{info['rekomendasi']}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            st.markdown("<div style='margin-top:19px;'></div>", unsafe_allow_html=True)

            # Prediksi + Tren — pola SAMA dengan Dashboard: kedua card diberi
            # tinggi tetap yang SAMA supaya sejajar & bottom-aligned.
            DETAIL_ROW_H = 440
            pc1, pc2 = st.columns([1, 1.4], gap="medium")

            # Prediksi 7 hari
            with pc1:
                with st.container(border=True, height=DETAIL_ROW_H):
                    st.markdown(
                        f"<div class='card-title'>Prediksi ISPU di {wilayah} (7 Hari Mendatang)</div>",
                        unsafe_allow_html=True,
                    )
                    rows_html = ""
                    for d in rentang_tanggal(sel_tgl, 1, 7):
                        p = predict_wilayah_tanggal(wilayah, d.isoformat())
                        if p is None:
                            continue
                        kat2 = p["kategori"]
                        warna = KATEGORI_INFO.get(kat2, KATEGORI_INFO["Sedang"])["warna"]
                        tanggal = tgl_id(d, singkat=True)
                        # Dibangun TANPA newline/indentasi -> markdown tidak salah
                        # mengira ini code block (penyebab HTML tampil mentah).
                        rows_html += (
                            "<div class='pred-row'>"
                            f"<div class='pred-date'>{tanggal}</div>"
                            "<div>"
                            f"<span class='pred-pill' style='background:{warna};'>"
                            f"{p['ispu']}</span></div>"
                            f"<div class='pred-cat' style='color:{warna};'>{kat2}</div>"
                            f"<div class='pred-pm'>PM2.5 ({p['pm25']} µg/m³)</div>"
                            "</div>"
                        )
                    # List dibungkus flex-column space-between -> baris terdistribusi
                    # memenuhi tinggi card (tanpa whitespace besar di bawah).
                    st.markdown(
                        f"<div style='height:{DETAIL_ROW_H - 78}px;display:flex;"
                        f"flex-direction:column;justify-content:space-between;'>"
                        f"{rows_html}</div>",
                        unsafe_allow_html=True,
                    )

            # Tren 7 hari (data dummy diolah per wilayah)
            with pc2:
                with st.container(border=True, height=DETAIL_ROW_H):
                    st.markdown(
                        f"<div class='card-title'>Tren ISPU di {wilayah} (7 Hari Terakhir)</div>",
                        unsafe_allow_html=True,
                    )

                    _tgl_tren = rentang_tanggal(sel_tgl, -6, 7)
                    _pts = [(d, predict_wilayah_tanggal(wilayah, d.isoformat()))
                            for d in _tgl_tren]
                    _pts = [(d, p) for d, p in _pts if p is not None]
                    df_tren = pd.DataFrame({
                        "tanggal": [pd.Timestamp(d) for d, _ in _pts],
                        "ispu_w": [p["ispu"] for _, p in _pts],
                    })
                    df_tren["label_x"] = df_tren["tanggal"].apply(tgl_id_pendek)

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df_tren["label_x"], y=df_tren["ispu_w"],
                        mode="lines+markers+text",
                        text=df_tren["ispu_w"],
                        textposition="top center",
                        textfont=dict(size=11, color="#0F172A", weight=600),
                        line=dict(color="#2563EB", width=3, shape="spline", smoothing=1.0),
                        marker=dict(size=9, color="#2563EB", line=dict(color="white", width=2)),
                        fill="tozeroy",
                        fillcolor="rgba(37, 99, 235, 0.08)",
                        hovertemplate="<b>%{x}</b><br>ISPU: %{y}<extra></extra>",
                        showlegend=False,
                    ))
                    # Garis + label threshold kategori — identik dengan Dashboard.
                    for nilai, label, warna in [
                        (50, "Baik", "#16A34A"),
                        (100, "Sedang", "#2563EB"),
                        (150, "Tidak Sehat", "#E5B93D"),
                    ]:
                        fig.add_hline(y=nilai, line_dash="dot",
                                      line_color="#E2E8F0", line_width=1)
                        fig.add_annotation(
                            x=1.0, xref="paper", y=nilai,
                            text=label, showarrow=False,
                            xanchor="left", yanchor="middle",
                            font=dict(size=10, color=warna, weight=600),
                            xshift=8,
                        )
                    fig.update_layout(
                        height=340,
                        margin=dict(l=40, r=140, t=50, b=30),
                        paper_bgcolor="white", plot_bgcolor="white",
                        xaxis=dict(
                            showgrid=False, showline=False,
                            tickfont=dict(size=11, color="#64748B"),
                            range=[-0.4, 6.4],
                        ),
                        yaxis=dict(
                            range=[0, 200], gridcolor="#F1F5F9", showline=False,
                            tickfont=dict(size=11, color="#94A3B8"),
                            tickvals=[0, 50, 100, 150, 200, 250, 300],
                        ),
                    )
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            # Info box ML
            st.markdown(
                """
                <div class='info-box'>
                    <div class='info-box-icon'>ⓘ</div>
                    <div class='info-box-text'>
                        Prediksi ini dibuat menggunakan model machine learning <strong>XGBoost</strong>
                        berdasarkan data historis ISPU pada tahun 2024.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ================================================================
# HALAMAN 3: SIMULASI PREDIKSI ISPU
# ================================================================
# ── Konfigurasi terpusat ───────────────────────────────────────
# DEFAULT_VALUES untuk state awal & target tombol Reset.
# Semua nol → ISPU final = 0, kategori = "Baik" (kondisi netral).
# Preset di bawah dirancang agar masing-masing jatuh tepat di
# kategori yang dimaksud sesuai breakpoint PerMenLHK No. 14/2020.
SIM_DEFAULT_VALUES = {
    "pm25": 0.0, "pm10": 0.0, "no2": 0.0,
    "so2":  0.0, "co":  0.0, "o3":  0.0,
}

# Preset: 3 skenario, satu per kategori ISPU.
# Setiap preset dirancang agar polutan dominan jatuh di pita target,
# sehingga ISPU final berada dalam rentang yang user inginkan.
SIM_PRESETS = {
    "Baik":               {"pm25": 10.0,  "pm10": 20.0,  "no2": 10.0,  "so2": 10.0,  "co": 1.0,  "o3": 20.0},
    "Sedang":             {"pm25": 35.0,  "pm10": 60.0,  "no2": 20.0,  "so2": 25.0,  "co": 2.0,  "o3": 45.0},
    "Tidak Sehat":        {"pm25": 90.0,  "pm10": 140.0, "no2": 60.0,  "so2": 70.0,  "co": 8.0,  "o3": 120.0},
}

# Konfigurasi slider per polutan (min/max/step + metadata UI).
# Max range dibiarkan lebar agar slider tetap bisa dieksplorasi bebas.
SIM_SLIDER_CONFIG = {
    "pm25": {"label": "PM2.5", "info_key": "PM2.5", "min": 0.0, "max": 500.0, "step": 0.5, "unit": "µg/m³", "decimals": 2, "slider_key": "sl_pm25"},
    "pm10": {"label": "PM10",  "info_key": "PM10",  "min": 0.0, "max": 500.0, "step": 0.5, "unit": "µg/m³", "decimals": 2, "slider_key": "sl_pm10"},
    "no2":  {"label": "NO₂",   "info_key": "NO₂",   "min": 0.0, "max": 500.0, "step": 0.5, "unit": "µg/m³", "decimals": 2, "slider_key": "sl_no2"},
    "so2":  {"label": "SO₂",   "info_key": "SO₂",   "min": 0.0, "max": 500.0, "step": 0.5, "unit": "µg/m³", "decimals": 2, "slider_key": "sl_so2"},
    "co":   {"label": "CO",    "info_key": "CO",    "min": 0.0, "max": 50.0,  "step": 0.1, "unit": "mg/m³", "decimals": 2, "slider_key": "sl_co"},
    "o3":   {"label": "O₃",    "info_key": "O₃",    "min": 0.0, "max": 500.0, "step": 0.5, "unit": "µg/m³", "decimals": 2, "slider_key": "sl_o3"},
}

POLUTAN_DISPLAY_NAME = {
    "pm25": "PM2.5", "pm10": "PM10", "no2": "NO₂",
    "so2":  "SO₂",   "co":   "CO",   "o3":  "O₃",
}


def _sim_init_state():
    """Init session state untuk simulasi — idempoten, aman dipanggil tiap rerun."""
    for pol, cfg in SIM_SLIDER_CONFIG.items():
        if cfg["slider_key"] not in st.session_state:
            st.session_state[cfg["slider_key"]] = float(SIM_DEFAULT_VALUES[pol])
    if "sim_active_preset" not in st.session_state:
        st.session_state["sim_active_preset"] = None
    if "sim_model_choice" not in st.session_state:
        st.session_state["sim_model_choice"] = "xgboost"


def apply_preset(name: str):
    """
    Callback `on_click` untuk tombol preset.
    Modifikasi session_state SEBELUM widget di-instantiate pada rerun berikutnya,
    sehingga slider otomatis nge-snap ke nilai preset tanpa st.rerun() manual.
    """
    if name not in SIM_PRESETS:
        return
    preset = SIM_PRESETS[name]
    for pol, val in preset.items():
        st.session_state[SIM_SLIDER_CONFIG[pol]["slider_key"]] = float(val)
    st.session_state["sim_active_preset"] = name


def reset_simulation():
    """
    Callback `on_click` untuk tombol Reset.
    Kembalikan semua slider ke default + hapus penanda preset aktif.
    """
    for pol, val in SIM_DEFAULT_VALUES.items():
        st.session_state[SIM_SLIDER_CONFIG[pol]["slider_key"]] = float(val)
    st.session_state["sim_active_preset"] = None


def _detect_active_preset(current_vals: dict):
    """
    Sinkronisasi 2-arah: deteksi preset aktif dari nilai slider.
    Jika user menggeser slider manual sehingga keluar dari preset,
    badge highlight otomatis hilang.
    """
    for name, preset in SIM_PRESETS.items():
        if all(abs(current_vals[k] - preset[k]) < 0.01 for k in preset):
            return name
    return None


def _polutan_slider_block(pol_key: str):
    """
    Render satu slider polutan dalam kartu ber-border (mengikuti mockup Figma):
    dot warna polutan + nama (PM2.5 diberi label "(Dominan)"), deskripsi singkat,
    lalu slider lebar penuh dengan label min/max bawaan Streamlit (0 … max).
    Mengembalikan nilai terbaru.
    """
    cfg = SIM_SLIDER_CONFIG[pol_key]
    info = INFO_POLUTAN[cfg["info_key"]]
    cur_val = float(st.session_state[cfg["slider_key"]])

    nama = cfg["label"] + (" (Dominan)" if pol_key == "pm25" else "")

    # Tiap polutan = satu kartu ber-border. Header (dot + nama) + deskripsi,
    # lalu slider Streamlit (label 0 … max muncul otomatis di bawah track).
    with st.container(border=True):
        st.markdown(
            f"""
            <div style="display:flex; align-items:center; gap:9px; margin-bottom:6px;">
                <span style="width:14px; height:14px; border-radius:50%;
                             background:{info["warna"]}; display:inline-block; flex-shrink:0;"></span>
                <span style="font-size:16px; font-weight:700; color:#0F172A;">{nama}</span>
            </div>
            <div style="font-size:13.5px; color:#64748B; line-height:1.45; margin-bottom:4px;">
                {info["deskripsi_pendek"]}
            </div>
            """,
            unsafe_allow_html=True,
        )
        val = st.slider(
            cfg["label"], cfg["min"], cfg["max"],
            value=cur_val, step=cfg["step"],
            key=cfg["slider_key"],
            label_visibility="collapsed",
        )
    return val


def page_simulasi(data):
    st.markdown(
        "<div class='page-title'>Simulasi Prediksi ISPU</div>"
        "<div class='page-subtitle'>Simulasikan kualitas udara berdasarkan konsentrasi polutan.</div>",
        unsafe_allow_html=True,
    )

    # Banner panduan
    st.markdown(
        """
        <div class='step-bar'>
            <div class='step-title'>ⓘ Cara Menggunakan Simulasi</div>
            <div class='step-item'>
                <div class='step-num'>1</div>
                <div class='step-text'>Pilih preset skenario atau geser slider untuk mengatur konsentrasi polutan.</div>
            </div>
            <div class='step-item'>
                <div class='step-num'>2</div>
                <div class='step-text'>Hasil ISPU dan kategori akan ter-update secara real-time di samping kanan.</div>
            </div>
            <div class='step-item'>
                <div class='step-num'>3</div>
                <div class='step-text'>Tekan "Reset" untuk mengembalikan semua slider ke kondisi awal.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Init state (idempoten) ──
    _sim_init_state()

    # Layout 2 kolom: kiri (Komposisi Polutan) ~59%, kanan (Hasil Prediksi)
    # ~41%. Gap dikecilkan dari "large" -> "medium" supaya kedua card lebih
    # rapat (tidak ada jarak kosong besar di tengah) namun tetap ada pemisah.
    col_left, col_right = st.columns([1.45, 1], gap="medium")

    # Mapping nama preset → suffix CSS class supaya warna pill sesuai kategori
    preset_css_suffix = {
        "Baik":               "baik",
        "Sedang":             "sedang",
        "Tidak Sehat":        "tdksehat",
    }

    # ─────────── KIRI: Card "Komposisi Polutan" ───────────
    with col_left:
        with st.container(border=True):

            # ── Header: judul (kiri) + tombol "Lihat Penjelasan Polutan" (kanan) ──
            head_l, head_r = st.columns([1, 1], gap="small")
            with head_l:
                st.markdown(
                    "<div class='sim-card-title' style='padding-top:8px;'>Komposisi Polutan</div>",
                    unsafe_allow_html=True,
                )
            with head_r:
                if st.button(
                    "ⓘ Lihat Penjelasan Polutan",
                    key="btn_info_simulasi", use_container_width=True,
                ):
                    render_popup_polutan()

            # Deskripsi card
            st.markdown(
                "<div class='sim-card-desc' style='margin-top:6px; margin-bottom:4px;'>"
                "Sesuaikan slider di bawah untuk mensimulasikan kondisi polutan dan "
                "memprediksi Indeks Standar Pencemar Udara (ISPU)."
                "</div>",
                unsafe_allow_html=True,
            )

            # ── Preset Skenario Udara ──
            st.markdown(
                "<div class='sim-section-label' style='margin-top:14px;'>Preset Skenario Udara</div>",
                unsafe_allow_html=True,
            )

            # Label tombol mengikuti mockup Figma. Kunci internal preset (key dict)
            # tetap memakai kategori ISPU resmi agar apply_preset & kalkulasi konsisten.
            preset_labels = {
                "Baik":               "Baik",
                "Sedang":             "Sedang",
                "Tidak Sehat":        "Tidak Baik",
            }
            pc = st.columns(3, gap="small")
            active_preset = st.session_state.get("sim_active_preset")
            for col, (name, label) in zip(pc, preset_labels.items()):
                with col:
                    suffix = preset_css_suffix[name]
                    is_active = " active" if name == active_preset else ""
                    # Marker sibling (tak terlihat) → dibaca CSS :has() untuk
                    # mewarnai tombol preset sesuai kategori + tandai aktif.
                    st.markdown(
                        f"<div class='pmkr pmkr-{suffix}{is_active}'></div>",
                        unsafe_allow_html=True,
                    )
                    st.button(
                        label, key=f"preset_{suffix}",
                        use_container_width=True,
                        help=f"Terapkan skenario kualitas udara {name}.",
                        on_click=apply_preset, args=(name,),
                    )

            # Model klasifikasi tetap XGBoost (disembunyikan dari UI sesuai desain).
            st.session_state["sim_model_choice"] = "xgboost"

            # ── Sliders 6 polutan dalam 2 kolom × 3 baris (tiap polutan = kartu) ──
            st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)
            sc1, sc2 = st.columns(2, gap="medium")
            vals = {}
            with sc1:
                vals["pm25"] = _polutan_slider_block("pm25")
                vals["no2"]  = _polutan_slider_block("no2")
                vals["co"]   = _polutan_slider_block("co")
            with sc2:
                vals["pm10"] = _polutan_slider_block("pm10")
                vals["so2"]  = _polutan_slider_block("so2")
                vals["o3"]   = _polutan_slider_block("o3")

            # Sinkron 2-arah preset↔slider
            detected = _detect_active_preset(vals)
            if detected != st.session_state.get("sim_active_preset"):
                st.session_state["sim_active_preset"] = detected

            # ── Footer: tombol Reset di kanan ──
            st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)
            _ft_spacer, ft_reset = st.columns([3, 1])
            with ft_reset:
                st.button(
                    "Reset", key="btn_reset",
                    type="secondary", use_container_width=True,
                    on_click=reset_simulation,
                    help="Kembalikan semua slider ke 0 & hapus preset aktif.",
                )


    # ─────────── KANAN: Card "Hasil Prediksi ISPU" ───────────
    with col_right:
        # Hitung ISPU realtime tiap rerun
        nilai_ispu, kategori, polutan_dominan, subindeks = calculate_ispu_category(
            pm10=vals["pm10"], pm25=vals["pm25"], so2=vals["so2"],
            co=vals["co"],   o3=vals["o3"],     no2=vals["no2"],
        )

        try:
            ml = prediksi_ispu_xgboost(
                pm10=vals["pm10"], pm25=vals["pm25"], so2=vals["so2"],
                co=vals["co"],   o3=vals["o3"],     no2=vals["no2"],
                model_choice=st.session_state.get("sim_model_choice", "xgboost"),
            )
            ml_kategori   = ml.get("kategori")
            ml_model_used = ml.get("model_used", "XGBoost")
            ml_confidence = ml.get("confidence")
        except Exception:
            ml_kategori = ml_model_used = ml_confidence = None

        info = KATEGORI_INFO[kategori]
        is_neutral = (nilai_ispu == 0)
        status_text = "Belum Ada Simulasi" if is_neutral else f"Udara {kategori}"
        deskripsi_text = (
            "Geser slider atau pilih preset untuk memulai simulasi."
            if is_neutral else info["deskripsi"]
        )
        rekom_text = (
            "Belum ada rekomendasi — silakan atur nilai polutan terlebih dahulu."
            if is_neutral else info["rekomendasi"]
        )
        # Variabel tampilan hero & kotak rekomendasi (netral = abu-abu).
        if is_neutral:
            hero_warna = "#94A3B8"
            hero_emoji_svg = (
                '<svg width="64" height="64" viewBox="0 0 100 100" '
                'xmlns="http://www.w3.org/2000/svg" style="flex-shrink:0;">'
                '<circle cx="50" cy="50" r="46" fill="#CBD5E1"/>'
                '<circle cx="36" cy="42" r="4" fill="white"/>'
                '<circle cx="64" cy="42" r="4" fill="white"/>'
                '<line x1="35" y1="60" x2="65" y2="60" stroke="white" '
                'stroke-width="5" stroke-linecap="round"/></svg>'
            )
            rekom_bg, rekom_border, rekom_color = "#F8FAFC", "#CBD5E1", "#64748B"
            rekom_emoji = "🧭"
        else:
            hero_warna = info["warna"]
            hero_emoji_svg = ispu_emoji_svg(kategori, size=64)
            rekom_bg, rekom_border, rekom_color = info["warna_bg"], info["warna"], info["warna"]
            rekom_emoji = info["rekom_emoji"]

        with st.container(border=True):

            # Judul card
            st.markdown(
                "<div class='card-title'>Hasil Prediksi ISPU</div>",
                unsafe_allow_html=True,
            )

            # Hero: angka ISPU (kiri) + emoji + status & deskripsi
            st.markdown(
                f"""
                <div style="display:flex; align-items:center; gap:22px;
                            margin-top:8px; margin-bottom:4px; flex-wrap:wrap;">
                    <div style="flex-shrink:0;">
                        <div class='ispu-number' style='color:{hero_warna};'>{nilai_ispu:.0f}</div>
                        <div class='ispu-label' style='text-align:center;'>ISPU</div>
                    </div>
                    <div class='ispu-emoji' style='margin-bottom:0;'>{hero_emoji_svg}</div>
                    <div style="flex:1; min-width:180px;">
                        <div class='ispu-status' style='color:{hero_warna};'>{status_text}</div>
                        <div class='ispu-desc'>{deskripsi_text}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Kotak Rekomendasi Aktivitas:
            #  • kondisi valid → gambar PNG hasil desain Figma (per kategori)
            #  • kondisi netral → kotak abu sederhana (belum ada desain khusus)
            rekom_b64 = "" if is_neutral else rekom_img_b64(REKOM_IMG.get(kategori, ""))
            if rekom_b64:
                st.markdown(
                    f"<img src='data:image/png;base64,{rekom_b64}' alt='Rekomendasi Aktivitas' "
                    f"style='width:100%; max-width:620px; height:auto; display:block; margin-top:18px;'/>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div style="background:{rekom_bg}; border:1.5px solid {rekom_border};
                                border-radius:14px; padding:18px 20px; margin-top:18px;">
                        <div style="font-size:18px; font-weight:700; color:{rekom_color};
                                    margin-bottom:10px;">
                            Rekomendasi Aktivitas
                        </div>
                        <div style="display:flex; gap:16px; align-items:flex-start;">
                            <div style="font-size:30px; line-height:1.1; flex-shrink:0;">{rekom_emoji}</div>
                            <div style="font-size:14.5px; color:#1E293B; line-height:1.6;">{rekom_text}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Ruang napas di bawah Rekomendasi Aktivitas supaya tidak menempel
            # ke tepi bawah card (sesuai desain Figma).
            st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

            # Kartu Hasil Prediksi ISPU berakhir di kotak Rekomendasi Aktivitas
            # (mengikuti mockup Figma). Bagian Sub-Indeks per Polutan & pembanding
            # model ML dihilangkan agar tampilan sama persis dengan desain.



# ================================================================
# HALAMAN 4: EDUKASI & INSIGHT
# ================================================================
def page_edukasi(data):
    st.markdown(
        "<div class='page-title'>Edukasi & Insight</div>"
        "<div class='page-subtitle'>Pelajari kategori ISPU, dampak kesehatan, dan tips menjaga kualitas hidup saat polusi udara meningkat.</div>",
        unsafe_allow_html=True,
    )

    # ============================================================
    # Section 1: Mengenal ISPU + 5 kategori
    # FIX: pakai st.container(border=True) (FIX #3) agar judul + kartu
    #      kategori berada DI DALAM card, bukan floating di luar.
    # ============================================================
    with st.container(border=True):
        st.markdown(
            "<div class='card-title'>Mengenal ISPU (Indeks Standar Pencemar Udara)</div>"
            "<div style='font-size:14px; color:#475569; margin-bottom:19px; line-height:1.5;'>"
            "ISPU digunakan untuk menggambarkan kualitas udara ambien di sekitar kita."
            "</div>",
            unsafe_allow_html=True,
        )

        # 5 kartu kategori ISPU = SVG desain Figma (assets/ispu_kategori.svg).
        # Kartu ini SENGAJA menampilkan 5 pita ISPU resmi PerMenLHK 14/2020
        # sebagai materi edukasi, terpisah dari skema 3 kelas yang dipakai
        # model klasifikasi dan dashboard.
        kat_svg = svg_inline("ispu_kategori.svg")
        if kat_svg:
            st.markdown(kat_svg, unsafe_allow_html=True)
        st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:19px;'></div>", unsafe_allow_html=True)

    # ============================================================
    # Section 2: Dampak Kesehatan + Sumber Polusi (2 kolom, equal-height)
    # Kedua card diberi tinggi tetap SAMA (EDU_CARD_H) -> tinggi presisi
    # sejajar. Card kiri (Dampak) isinya sedikit, jadi diisi penuh dengan
    # grid 2x2 yang ROW-nya stretch (1fr 1fr) + konten tiap sel di-center,
    # sehingga ruang terisi merata tanpa whitespace besar di bawah.
    # ============================================================
    EDU_CARD_H = 460
    dc1, dc2 = st.columns([1.4, 1], gap="medium")

    # --- Dampak Kesehatan ---
    with dc1:
        with st.container(border=True, height=EDU_CARD_H):
            dampak_items = [
                ("🫁", "Saluran Pernapasan",
                 "Polusi udara dapat menyebabkan iritasi, batuk, sesak napas, "
                 "dan memperparah asma."),
                ("❤️", "Jantung &amp; Pembuluh Darah",
                 "Paparan polusi jangka panjang meningkatkan risiko penyakit "
                 "jantung dan tekanan darah tinggi."),
                ("👶", "Anak-anak",
                 "Anak-anak lebih rentan mengalami infeksi saluran pernapasan "
                 "dan gangguan perkembangan paru-paru."),
                ("🧓", "Lansia",
                 "Lansia lebih berisiko mengalami gangguan kesehatan akibat "
                 "polusi udara, terutama yang memiliki penyakit bawaan."),
            ]
            cells = ""
            for _emoji, _judul, _desc in dampak_items:
                cells += (
                    # sel grid (stretch) -> item di-center vertikal agar penuh
                    "<div style='display:flex; align-items:center;'>"
                    "<div style='display:flex; gap:14px; align-items:flex-start;'>"
                    "<div style='flex-shrink:0; width:48px; height:48px; "
                    "border-radius:999px; background:#F1F5F9; display:flex; "
                    "align-items:center; justify-content:center; font-size:22px; "
                    f"line-height:1;'>{_emoji}</div>"
                    "<div>"
                    "<div style='font-weight:700; color:#111827; font-size:16px; "
                    f"margin-bottom:5px;'>{_judul}</div>"
                    "<div style='font-size:14px; color:#475569; "
                    f"line-height:1.55;'>{_desc}</div>"
                    "</div>"
                    "</div>"
                    "</div>"
                )
            dampak_html = (
                # tinggi dalam = EDU_CARD_H - padding wrapper (20px atas+bawah)
                f"<div style='height:{EDU_CARD_H - 40}px; display:flex; "
                "flex-direction:column;'>"
                "<div style='font-weight:700; color:#111827; font-size:18px; "
                "margin-bottom:6px;'>Dampak Kualitas Udara terhadap "
                "Kesehatan</div>"
                "<div style='display:grid; grid-template-columns:1fr 1fr; "
                "grid-template-rows:1fr 1fr; gap:18px 30px; flex:1;'>"
                f"{cells}</div>"
                "</div>"
            )
            st.markdown(dampak_html, unsafe_allow_html=True)

    # --- Sumber Polusi (donut chart) ---
    with dc2:
        with st.container(border=True, height=EDU_CARD_H):
            st.markdown(
                "<div style='font-weight:700; color:#111827; font-size:18px; "
                "margin-bottom:6px;'>Sumber Polusi Udara di Jakarta</div>"
                "<div style='font-size:14px; color:#475569; margin-bottom:8px; "
                "line-height:1.5;'>Estimasi kontribusi tiap sektor.</div>",
                unsafe_allow_html=True,
            )

            sumber = {
                "Transportasi": (45, "#2563EB"),
                "Industri": (20, "#16A34A"),
                "Aktivitas Rumah Tangga": (15, "#F59E0B"),
                "Konstruksi": (10, "#EF4444"),
                "Lainnya": (10, "#7C3AED"),
            }

            fig = go.Figure(go.Pie(
                labels=list(sumber.keys()),
                values=[v[0] for v in sumber.values()],
                hole=0.6,
                marker=dict(colors=[v[1] for v in sumber.values()],
                            line=dict(color="white", width=3)),
                textinfo="none",
                hovertemplate="<b>%{label}</b><br>%{value}%<extra></extra>",
            ))
            fig.update_layout(
                height=220,
                margin=dict(l=0, r=0, t=0, b=0),
                showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )

            # Donut (kiri) + legend (kanan), sejajar — sesuai desain Figma.
            try:
                pc1, pc2 = st.columns([0.8, 1.2], vertical_alignment="center")
            except TypeError:
                # Fallback Streamlit < 1.36 (tanpa vertical_alignment)
                pc1, pc2 = st.columns([0.8, 1.2])

            with pc1:
                st.plotly_chart(fig, use_container_width=True,
                                config={"displayModeBar": False})

            with pc2:
                # Legend — SATU blok HTML agar tidak bocor keluar card
                legend_html = "<div>"
                for nama, (pct, warna) in sumber.items():
                    legend_html += (
                        f"<div class='donut-legend-row'>"
                        f"<div class='donut-legend-left'>"
                        f"<div class='donut-legend-dot' style='background:{warna};'></div>"
                        f"<span>{nama}</span></div>"
                        f"<div class='donut-legend-pct'>{pct}%</div>"
                        f"</div>"
                    )
                legend_html += "</div>"
                st.markdown(legend_html, unsafe_allow_html=True)

            st.markdown(
                "<div class='info-card'>"
                "<span class='info-icon'>ℹ️</span>"
                "<span>Data bersifat <b>ilustratif</b>, bukan kondisi realtime. "
                "Angka estimasi merujuk pada kajian umum sumber polusi udara perkotaan "
                "dan dapat berbeda dengan kondisi aktual Jakarta.</span>"
                "</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<div style='margin-top:19px;'></div>", unsafe_allow_html=True)

    # ============================================================
    # Section 3: Tips Kesehatan — dibangun ulang sebagai HTML NATIVE.
    # Sebelumnya: 1 SVG besar (jaga_kesehatan.svg) — SVG itu menggambar
    # outer-rect ber-stroke sendiri — DI DALAM st.container(border=True),
    # sehingga border bertumpuk ("kotak di dalam kotak").
    # Sekarang: SATU <div> container ber-border (outline tunggal) berisi
    # judul + 5 kartu (flex). Tiap kartu punya border tipisnya sendiri.
    # Tidak ada st.container / kolom Streamlit -> tidak ada border ganda.
    # ============================================================
    TIPS_ICONS = {
        "mask": (
            "<svg width='30' height='30' viewBox='0 0 24 24' fill='none' "
            "stroke='currentColor' stroke-width='1.8' stroke-linecap='round' "
            "stroke-linejoin='round'>"
            "<path d='M5 9h14v3a5 5 0 0 1-5 5h-4a5 5 0 0 1-5-5V9z'/>"
            "<path d='M5 11H2M19 11h3'/><path d='M9 11h6M9.5 14h5'/></svg>"
        ),
        "ban": (
            "<svg width='30' height='30' viewBox='0 0 24 24'>"
            "<circle cx='12' cy='12' r='11' fill='#2563EB'/>"
            "<path d='M8.5 8.5l7 7M15.5 8.5l-7 7' stroke='#fff' "
            "stroke-width='2' stroke-linecap='round'/></svg>"
        ),
        "wind": (
            "<svg width='30' height='30' viewBox='0 0 24 24' fill='none' "
            "stroke='currentColor' stroke-width='1.8' stroke-linecap='round' "
            "stroke-linejoin='round'>"
            "<path d='M17.7 7.7a2.5 2.5 0 1 1 1.8 4.3H2'/>"
            "<path d='M9.6 4.6A2 2 0 1 1 11 8H2'/>"
            "<path d='M12.6 19.4A2 2 0 1 0 14 16H2'/></svg>"
        ),
        "droplet": (
            "<svg width='30' height='30' viewBox='0 0 24 24' fill='none' "
            "stroke='currentColor' stroke-width='1.8' stroke-linecap='round' "
            "stroke-linejoin='round'>"
            "<path d='M12 22a7 7 0 0 0 7-7c0-2-1-3.9-3-5.5s-3.5-4-4-6.5"
            "c-.5 2.5-2 4.9-4 6.5C6 11.1 5 13 5 15a7 7 0 0 0 7 7z'/></svg>"
        ),
        "purifier": (
            "<svg width='30' height='30' viewBox='0 0 24 24' fill='none' "
            "stroke='currentColor' stroke-width='1.8' stroke-linecap='round' "
            "stroke-linejoin='round'>"
            "<rect x='4' y='4' width='9' height='16' rx='2'/>"
            "<path d='M7 8h3M7 11h3'/>"
            "<path d='M16 9a4 4 0 0 1 0 6'/><path d='M18.5 7a7 7 0 0 1 0 10'/>"
            "</svg>"
        ),
    }
    tips_data = [
        ("mask", "Gunakan Masker",
         "Gunakan masker berstandar untuk mengurangi paparan polusi udara."),
        ("ban", "Kurangi Aktivitas di Luar Ruangan",
         "Kurangi aktivitas fisik berat di luar ruangan, terutama saat sore "
         "hingga malam hari."),
        ("wind", "Jaga Kualitas Udara Dalam Ruangan",
         "Tutup jendela saat polusi tinggi dan pastikan ventilasi rumah tetap "
         "berfungsi baik."),
        ("droplet", "Perbanyak Minum Air",
         "Membantu menjaga kenyamanan saluran pernapasan saat kualitas udara "
         "memburuk."),
        ("purifier", "Gunakan Air Purifier",
         "Jika memungkinkan, gunakan alat penyaring udara di dalam ruangan "
         "untuk udara lebih bersih."),
    ]
    tips_cards = ""
    for _key, _judul, _desc in tips_data:
        tips_cards += (
            "<div style='flex:1 1 0;min-width:0;border:1px solid #E5E7EB;"
            "border-radius:14px;background:#fff;padding:24px;display:flex;"
            "flex-direction:column;'>"
            "<div style='color:#2563EB;margin-bottom:16px;line-height:0;'>"
            f"{TIPS_ICONS[_key]}</div>"
            "<div style='font-size:17px;font-weight:700;color:#0F172A;"
            f"line-height:1.3;margin-bottom:10px;'>{_judul}</div>"
            "<div style='font-size:14px;color:#64748B;line-height:1.6;'>"
            f"{_desc}</div>"
            "</div>"
        )
    tips_html = (
        "<div style='border:1px solid #E5E7EB;border-radius:16px;"
        "background:#fff;padding:30px;margin-top:4px;'>"
        "<div style='display:flex;align-items:center;gap:10px;font-size:22px;"
        "font-weight:700;color:#0F172A;margin-bottom:24px;'>"
        "<span style='font-size:22px;line-height:1;'>💡</span>"
        "Tips Menjaga Kesehatan Saat Kualitas Udara Tidak Sehat</div>"
        "<div style='display:flex;align-items:stretch;gap:20px;'>"
        f"{tips_cards}</div>"
        "</div>"
    )
    st.markdown(tips_html, unsafe_allow_html=True)


# ================================================================
# MAIN ROUTER
# ================================================================
def inject_layout_lock():
    """Kunci lebar area konten ke nilai px tetap & rata-tengah.

    Ini membuat layout konsisten dari zoom 50% s/d 100%: karena lebar
    konten tidak lagi mengikuti lebar viewport (yang berubah saat zoom),
    susunan kolom/kartu jadi identik di semua level zoom. Yang berubah
    hanya ukuran fisik di layar — itu memang perilaku zoom yang wajar.
    """
    st.markdown(
        f"""
        <style>
        /* Kunci lebar konten utama + rata-tengah.
           Pakai max-width: konten TIDAK pernah melebihi nilai ini & tidak
           ada scroll horizontal. Kalau di zoom 100% masih terlihat menyempit
           (viewport monitor lebih kecil dari nilai ini), pilih salah satu:
             1) turunkan CONTENT_MAX_WIDTH, atau
             2) ganti 'max-width' di bawah jadi 'width' (kunci keras /
                hard-lock; identik di semua zoom, tapi bisa muncul scroll
                horizontal kalau tidak muat di zoom 100%). */
        .stApp .block-container {{
            max-width: {CONTENT_MAX_WIDTH}px !important;
            margin-left: auto !important;
            margin-right: auto !important;
        }}
        /* Cegah kolom Streamlit reflow/stacking saat viewport melebar/menyempit
           — pertahankan rasio kolom apa adanya di semua zoom */
        .stApp [data-testid="stHorizontalBlock"] {{
            flex-wrap: nowrap !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    inject_css()
    inject_layout_lock()
    data = load_data()

    # Handle redirect dari tombol "Lihat Selengkapnya" di dashboard.
    # Set halaman SEBELUM sidebar dirender agar highlight nav ikut sinkron.
    if st.session_state.get("jump_to_detail"):
        st.session_state["jump_to_detail"] = False
        st.session_state["nav_page"] = "Detail Wilayah"

    page = render_sidebar()

    if page == "Dashboard":
        page_dashboard(data)
    elif page == "Detail Wilayah":
        page_detail_wilayah(data)
    elif page == "Simulasi Prediksi ISPU":
        page_simulasi(data)
    elif page == "Edukasi & Insight":
        page_edukasi(data)


if __name__ == "__main__":
    main()
