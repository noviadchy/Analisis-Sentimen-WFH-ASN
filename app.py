import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import re
import html as html_module
import os
from scipy.sparse import hstack, csr_matrix
from collections import Counter

import nltk
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
from nltk.tokenize import word_tokenize

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# PAGE CONFIG
st.set_page_config(
    page_title="Analisis Sentimen WFH ASN",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS (WHITE THEME)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif !important;
    background: #f8f9fb !important;
    color: #1a1a2e !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e5e7eb !important;
}
section[data-testid="stSidebar"] * { color: #374151 !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #111827 !important; }

/* Main container */
.main .block-container {
    padding: 1.75rem 2.5rem !important;
    max-width: 1100px !important;
}

/* Hero */
.hero {
    background: linear-gradient(120deg, #4f46e5 0%, #6366f1 60%, #818cf8 100%);
    border-radius: 14px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    color: #fff;
}
.hero h1 { font-size: 1.6rem; font-weight: 700; margin: 0 0 0.4rem; color: #fff; }
.hero p  { font-size: 0.95rem; color: rgba(255,255,255,0.9); margin: 0; line-height: 1.7; }

/* Stat cards */
.stats-row { display: flex; gap: 0.75rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
.stat-box {
    flex: 1;
    min-width: 120px;
    background: #fff;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 0.875rem 1rem;
}
.stat-box .lbl { font-size: 0.75rem; font-weight: 600; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 4px; }
.stat-box .val { font-size: 1.5rem; font-weight: 700; color: #111827; line-height: 1; }
.stat-box .sub { font-size: 0.78rem; color: #9ca3af; margin-top: 4px; }

/* Cards */
.card {
    background: #fff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}
.card-title {
    font-size: 0.75rem;
    font-weight: 700;
    color: #6366f1;
    letter-spacing: 1px;
    text-transform: uppercase;
    font-family: 'DM Mono', monospace;
    margin-bottom: 0.875rem;
}

/* Result banner */
.result-wrap {
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin: 0.75rem 0;
    border-left: 5px solid;
}
.res-pos { background: #f0fdf4; border-color: #22c55e; }
.res-neg { background: #fff1f2; border-color: #ef4444; }
.res-neu { background: #f8fafc; border-color: #94a3b8; }
.res-label { font-size: 1.6rem; font-weight: 700; margin: 0 0 6px; }
.res-meta  { font-size: 0.85rem; color: #6b7280; }

/* Confidence bars */
.cbar-wrap { margin: 8px 0; }
.cbar-head { display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 4px; }
.cbar-track { background: #f1f5f9; border-radius: 100px; height: 8px; overflow: hidden; }
.cbar-fill  { height: 100%; border-radius: 100px; }

/* Token pills */
.pill {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 100px;
    font-size: 0.78rem;
    font-weight: 500;
    margin: 3px 2px;
    font-family: 'DM Mono', monospace;
}
.pill-pos { background: #dcfce7; color: #166534; border: 1px solid #bbf7d0; }
.pill-neg { background: #fee2e2; color: #991b1b; border: 1px solid #fecaca; }
.pill-dom { background: #eff6ff; color: #1e40af; border: 1px solid #bfdbfe; }
.pill-neu { background: #f1f5f9; color: #475569; border: 1px solid #e2e8f0; }

/* Word score bar rows */
.wbar-row { display: flex; align-items: center; gap: 8px; margin: 5px 0; }
.wbar-name { font-size: 0.8rem; color: #374151; width: 110px; flex-shrink: 0;
             font-family: 'DM Mono', monospace; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.wbar-track { flex: 1; background: #f1f5f9; border-radius: 100px; height: 14px; overflow: hidden; }
.wbar-fill  { height: 100%; border-radius: 100px; }
.wbar-score { font-size: 0.75rem; color: #6b7280; width: 42px; text-align: right;
              font-family: 'DM Mono', monospace; flex-shrink: 0; }
.wbar-tag   { font-size: 0.72rem; font-weight: 600; width: 95px; text-align: right; flex-shrink: 0; }

/* Batch row */
.batch-item { display: flex; gap: 10px; align-items: flex-start; padding: 10px 0; border-bottom: 1px solid #f1f5f9; }
.batch-item:last-child { border-bottom: none; }
.batch-txt { flex: 1; font-size: 0.85rem; color: #374151; line-height: 1.55; }
.badge { display: inline-block; padding: 3px 10px; border-radius: 100px; font-size: 0.75rem; font-weight: 700; white-space: nowrap; }
.badge-pos { background: #dcfce7; color: #166534; }
.badge-neg { background: #fee2e2; color: #991b1b; }
.badge-neu { background: #f1f5f9; color: #475569; }

/* Dist bar */
.dist-bar-wrap { margin: 6px 0; }
.dist-bar-head { display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 4px; }
.dist-bar-track { background: #f1f5f9; border-radius: 100px; height: 9px; overflow: hidden; }
.dist-bar-fill  { height: 100%; border-radius: 100px; }

/* Buttons */
div[data-testid="stButton"] button {
    background: #fff !important;
    border: 1px solid #d1d5db !important;
    color: #374151 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.82rem !important;
    border-radius: 8px !important;
    transition: all 0.15s !important;
    text-align: left !important;
}
div[data-testid="stButton"] button:hover {
    background: #f5f3ff !important;
    border-color: #6366f1 !important;
    color: #4f46e5 !important;
}
div[data-testid="stButton"] button[kind="primary"] {
    background: #4f46e5 !important;
    border-color: #4f46e5 !important;
    color: #fff !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
}
div[data-testid="stButton"] button[kind="primary"]:hover {
    background: #4338ca !important;
    border-color: #4338ca !important;
    color: #fff !important;
}

/* Textarea */
textarea {
    background: #fff !important;
    border: 1px solid #d1d5db !important;
    color: #111827 !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.92rem !important;
}
textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.12) !important;
}

/* Tabs */
div[data-testid="stTabs"] button {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    color: #6b7280 !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #4f46e5 !important;
    border-bottom-color: #4f46e5 !important;
}

/* Metrics */
div[data-testid="metric-container"] {
    background: #fff !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 10px !important;
    padding: 0.75rem 1rem !important;
}
div[data-testid="metric-container"] label {
    color: #9ca3af !important; font-size: 0.8rem !important;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color: #111827 !important; font-size: 1.4rem !important; font-weight: 700 !important;
}

/* Expander */
details {
    background: #fff !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 10px !important;
    padding: 0 !important;
}
details summary {
    font-size: 0.88rem !important;
    color: #6b7280 !important;
    padding: 0.75rem 1rem !important;
}

/* Section headers */
h4 { font-size: 1.05rem !important; font-weight: 700 !important; color: #111827 !important; }

hr { border-color: #e5e7eb !important; }
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #f1f5f9; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }

@media (max-width: 768px) {
    .main .block-container { padding: 1rem 1rem !important; }
    .hero { padding: 1.25rem 1.25rem; border-radius: 10px; }
    .hero h1 { font-size: 1.25rem; }
    .hero p  { font-size: 0.875rem; }
    .stats-row { display: grid !important; grid-template-columns: 1fr 1fr; gap: 0.6rem; }
    .stat-box .val { font-size: 1.25rem; }
    .card { padding: 1rem 1rem; }
    .wbar-name { width: 80px; }
    .wbar-tag  { width: 75px; font-size: 0.68rem; }
    .res-label { font-size: 1.3rem; }
    .batch-item { flex-wrap: wrap; }
}
</style>
""", unsafe_allow_html=True)

# CONSTANTS
WARNA     = {"Positif": "#22c55e", "Negatif": "#ef4444", "Netral": "#94a3b8"}
RES_CLS   = {"Positif": "res-pos", "Negatif": "res-neg", "Netral": "res-neu"}
RES_COLOR = {"Positif": "#16a34a", "Negatif": "#dc2626", "Netral": "#64748b"}

CONTOH = [
    "WFH sangat efektif mengurangi kemacetan Jakarta dan hemat BBM!",
    "ASN WFH malah jadi ajang liburan, kerja cuma setengah hati.",
    "Pemerintah resmi menerapkan WFH setiap hari Jumat untuk ASN.",
    "Kebijakan WFH tidak signifikan mengurangi konsumsi BBM nasional.",
    "WFH bukan solusi nyata, transportasi umum yang harus diperbaiki!",
    "Mendukung kebijakan WFH ASN karena bisa meningkatkan produktivitas.",
]

KATA_POSITIF_DOMAIN = {
    'efisien','efektif','produktif','mendukung','setuju','bagus','manfaat','bermanfaat',
    'berhasil','sukses','tepat','bijak','dukung','apresiasi','inovatif','fleksibel',
    'fleksibilitas','hemat','ramah','transparan','akuntabel','disiplin','profesional',
    'optimal','optimalkan','meningkat','meningkatkan','maju','kemajuan','solusi',
    'solutif','responsif','adaptif','kreatif','kolaboratif'
}
KATA_NEGATIF_DOMAIN = {
    'malas','males','bolos','gagal','boros','korupsi','nganggur','percuma','manipulasi',
    'menyalahgunakan','disalahgunakan','kecewa','menolak','tolak','protes','kontra',
    'liburan','santai','nonton','rebahan','tidur','lambat','lamban','terhambat',
    'terbengkalai','mangkir','absen','buang','membuang','keluhan','mengeluh',
    'buruk','jelek','parah','amburadul','kacau'
}
KATA_NETRAL_DOMAIN = {
    'kebijakan','aturan','regulasi','peraturan','ketentuan','surat','edaran','instruksi',
    'pergub','perpres','permenpan','mengkaji','membahas','mengevaluasi','meninjau',
    'rapat','sidang','konferensi','diskusi','sosialisasi','laporan','data','statistik',
    'survei','hasil','temuan','menyatakan','mengatakan','menegaskan','menjelaskan',
    'mengumumkan','berlaku','diterapkan','diberlakukan','ditetapkan','diumumkan',
    'mulai','dimulai','rencananya','direncanakan','dijadwalkan',
    'wfh','wfo','wfa','hybrid','asn','pns','aparatur','pegawai','karyawan',
    'kantor','instansi','kementerian','dinas','badan','jumat','senin','jadwal',
    'minggu','sepekan','persen','poin','angka','jumlah','total',
    'jakarta','provinsi','kabupaten','kota','daerah','menteri','gubernur',
    'bupati','walikota','kepala'
}

KAMUS_SLANG = {
    "yg":"yang","dgn":"dengan","utk":"untuk","tdk":"tidak","gak":"tidak",
    "gk":"tidak","ga":"tidak","nggak":"tidak","ngga":"tidak","enggak":"tidak",
    "engga":"tidak","tak":"tidak","kagak":"tidak","krn":"karena","karna":"karena",
    "soalnya":"karena","sdh":"sudah","udah":"sudah","udh":"sudah","blm":"belum",
    "blum":"belum","jg":"juga","spt":"seperti","sprt":"seperti","kyk":"seperti",
    "kayak":"seperti","kaya":"seperti","dr":"dari","pd":"pada","dg":"dengan",
    "tp":"tapi","tpi":"tapi","ttg":"tentang","spy":"supaya","bgt":"banget",
    "bngt":"banget","aja":"saja","aj":"saja","doang":"saja","sy":"saya",
    "sya":"saya","gua":"saya","gue":"saya","gw":"saya","w":"saya",
    "lu":"kamu","lo":"kamu","elu":"kamu","bs":"bisa","bsa":"bisa",
    "hrs":"harus","nih":"ini","tuh":"itu","tu":"itu","ni":"ini",
    "emg":"memang","emang":"memang","kpd":"kepada","thdp":"terhadap",
    "dgr":"dengar","jwb":"jawab","lbh":"lebih","skrg":"sekarang",
    "skrang":"sekarang","stlh":"setelah","sblm":"sebelum","jd":"jadi",
    "jdi":"jadi","bkn":"bukan","krj":"kerja","msh":"masih","dpt":"dapat",
    "sm":"sama","klo":"kalau","klu":"kalau","kl":"kalau","klau":"kalau",
    "gmn":"bagaimana","gimana":"bagaimana","knp":"kenapa","sbg":"sebagai",
    "thd":"terhadap","jgn":"jangan","lgsg":"langsung","lngsng":"langsung",
    "wkwk":"","wkwkwk":"","haha":"","hehe":"","hihi":"","hmm":"",
    "ehh":"","duh":"","aduh":"","ih":"","sih":"","deh":"",
    "dong":"","loh":"","kok":"","lah":"","kan":"","yah":"","ya":"",
}

# SESSION STATE
for k, v in [
    ("input_teks", ""),        # storage value textarea tab 1
    ("hasil_analisis", None),
    ("batch_teks", ""),        # storage value textarea tab 2
    ("batch_rows", []),
]:
    if k not in st.session_state:
        st.session_state[k] = v

# LOAD MODEL
@st.cache_resource(show_spinner="Memuat model...")
def load_semua():
    priority = [
        "svm_linear_model.pkl","svm_rbf_model.pkl",
        "svm_polynomial_model.pkl","svm_sigmoid_model.pkl",
    ]
    model_path = next((p for p in priority if os.path.exists(p)), None)
    if model_path is None:
        raise FileNotFoundError("File svm_*_model.pkl tidak ditemukan.")
    model      = joblib.load(model_path)
    tfidf_word = joblib.load("tfidf_word_vectorizer.pkl")
    tfidf_char = joblib.load("tfidf_char_vectorizer.pkl")
    lexicon    = joblib.load("inset_lexicon.pkl")
    sw         = StopWordRemoverFactory()
    swids      = set(sw.get_stop_words())
    kata_neg   = {"tidak","bukan","jangan","belum","tanpa","tiada","tak"}
    stopwords_fin = swids - kata_neg
    stemmer    = StemmerFactory().create_stemmer()
    return model, tfidf_word, tfidf_char, lexicon, stopwords_fin, stemmer, model_path

try:
    model, tfidf_word, tfidf_char, lexicon, stopwords_fin, stemmer, model_path = load_semua()
    MODEL_OK   = True
    NAMA_MODEL = os.path.splitext(model_path)[0].replace("svm_","").replace("_model","").capitalize()
except Exception as e:
    MODEL_OK = False
    ERR_MSG  = str(e)
    NAMA_MODEL = "—"

# PREPROCESSING
def clean_text(text):
    text = html_module.unescape(str(text))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\d+", "", text)
    return re.sub(r"\s+", " ", text).strip()

def normalize_slang(text):
    words = text.split()
    return " ".join(w for w in (KAMUS_SLANG.get(x.lower(), x) for x in words) if w)

def preprocess_full(raw):
    cleaned   = clean_text(raw)
    slang     = normalize_slang(cleaned)
    lower     = slang.lower()
    tokens    = word_tokenize(lower)
    no_stop   = [w for w in tokens if w not in stopwords_fin]
    pre_stem  = " ".join(no_stop)
    stemmed   = [stemmer.stem(w) for w in no_stop]
    post_stem = " ".join(stemmed)
    return pre_stem, post_stem, no_stop, stemmed

def buat_fitur_num(pre_stem_text):
    words   = pre_stem_text.split()
    n_pos   = sum(1 for w in words if lexicon.get(w, 0) > 0)
    n_neg   = sum(1 for w in words if lexicon.get(w, 0) < 0)
    selisih = n_pos - n_neg
    total   = n_pos + n_neg
    rasio   = selisih / total if total > 0 else 0
    return csr_matrix(np.array([[n_pos, n_neg, selisih, rasio]]))

def _softmax_proba(scores, classes):
    arr = np.array(scores, dtype=float)
    arr -= arr.max()
    exp = np.exp(arr)
    soft = exp / exp.sum()
    return {c: float(p) for c, p in zip(classes, soft)}

def predict_sentimen(raw_text):
    pre_stem, post_stem, tokens_no_stop, _ = preprocess_full(raw_text)
    X_word = tfidf_word.transform([post_stem])
    X_char = tfidf_char.transform([post_stem])
    X_num  = buat_fitur_num(pre_stem)
    X      = hstack([X_word, X_char, X_num])

    label   = model.predict(X)[0]
    classes = list(model.classes_)

    if hasattr(model, "predict_proba") and model.probability:
        proba = {c: float(p) for c, p in zip(classes, model.predict_proba(X)[0])}
    else:
        df_sc  = model.decision_function(X)[0]
        scores = list(df_sc) if len(classes) > 2 else [-float(df_sc), float(df_sc)]
        proba  = _softmax_proba(scores, classes)

    skor_inset = sum(lexicon.get(w, 0) for w in pre_stem.split())

    kata_info = []
    for w in tokens_no_stop:
        inset_skor = lexicon.get(w, 0)
        if inset_skor > 0:
            kata_info.append({"kata": w, "skor": inset_skor, "sumber": "InSet", "polaritas": "Positif"})
        elif inset_skor < 0:
            kata_info.append({"kata": w, "skor": inset_skor, "sumber": "InSet", "polaritas": "Negatif"})
        elif w in KATA_POSITIF_DOMAIN:
            kata_info.append({"kata": w, "skor": None, "sumber": "Domain", "polaritas": "Positif"})
        elif w in KATA_NEGATIF_DOMAIN:
            kata_info.append({"kata": w, "skor": None, "sumber": "Domain", "polaritas": "Negatif"})
        elif w in KATA_NETRAL_DOMAIN:
            kata_info.append({"kata": w, "skor": None, "sumber": "Domain", "polaritas": "Netral"})
        else:
            kata_info.append({"kata": w, "skor": None, "sumber": "-", "polaritas": "-"})

    return {
        "label": label, "proba": proba, "skor_inset": skor_inset,
        "pre_stem": pre_stem, "post_stem": post_stem,
        "tokens": tokens_no_stop, "kata_info": kata_info,
    }

# HTML HELPERS
def conf_bar_html(proba):
    order = [("Positif","#22c55e"), ("Netral","#94a3b8"), ("Negatif","#ef4444")]
    parts = []
    for lbl, clr in order:
        pct = proba.get(lbl, 0) * 100
        parts.append(f"""
        <div class="cbar-wrap">
          <div class="cbar-head">
            <span style="color:#374151">{lbl}</span>
            <span style="color:{clr};font-weight:700;font-family:'DM Mono',monospace">{pct:.1f}%</span>
          </div>
          <div class="cbar-track">
            <div class="cbar-fill" style="width:{pct}%;background:{clr}"></div>
          </div>
        </div>""")
    return "\n".join(parts)

def kata_bars_html(kata_info):
    relevant = [k for k in kata_info if k["polaritas"] != "-"]
    if not relevant:
        return """
        <div style="padding:1.5rem;text-align:center;background:#f8f9fb;border-radius:8px;
             border:1px dashed #e5e7eb;color:#9ca3af;font-size:0.85rem">
          Tidak ada kata bermuatan sentimen yang terdeteksi.<br>
          <span style="font-size:0.78rem">Kemungkinan teks hanya berisi kata-kata netral atau teks terlalu pendek.</span>
        </div>"""

    inset_scores = [abs(k["skor"]) for k in relevant if k["skor"] is not None]
    max_skor = max(inset_scores) if inset_scores else 1

    color_map = {
        "Positif": "#22c55e",
        "Negatif": "#ef4444",
        "Netral":  "#6366f1",
    }

    parts = []
    for k in relevant:
        clr  = color_map.get(k["polaritas"], "#94a3b8")
        kata = k["kata"]

        if k["skor"] is not None:
            pct      = (abs(k["skor"]) / max_skor) * 75
            skor_txt = f"{k['skor']:+.2f}"
            tag_txt  = "InSet"
            tag_desc = k["polaritas"]
        else:
            if k["polaritas"] == "Netral":
                pct      = 22
                skor_txt = "—"
                tag_txt  = "Domain"
                tag_desc = k["polaritas"]     # Netral
            else:
                pct      = 38
                skor_txt = "—"
                tag_txt  = "Domain"
                tag_desc = k["polaritas"]     # Positif / Negatif

        parts.append(f"""
        <div class="wbar-row">
          <div class="wbar-name" title="{kata}">{kata}</div>
          <div class="wbar-track">
            <div class="wbar-fill" style="width:{pct}%;background:{clr};opacity:0.8"></div>
          </div>
          <div class="wbar-score">{skor_txt}</div>
          <div class="wbar-tag" style="color:{clr}">{tag_txt}</div>
        </div>
        <div style="font-size:0.68rem;color:#9ca3af;margin:-3px 0 4px 118px">{tag_desc if k["skor"] is None else ""}</div>""")

    return "\n".join(parts)

def dist_bars_html(counts, total):
    """Distribusi bar untuk multi-teks — tidak bergantung pada variabel h."""
    parts = []
    for lbl, clr in [("Positif","#22c55e"),("Netral","#94a3b8"),("Negatif","#ef4444")]:
        n   = counts.get(lbl, 0)
        pct = n / total * 100 if total > 0 else 0
        parts.append(f"""
        <div class="dist-bar-wrap">
          <div class="dist-bar-head">
            <span style="color:#374151">{lbl}</span>
            <span style="color:{clr};font-weight:700;font-family:'DM Mono',monospace">{n} ({pct:.1f}%)</span>
          </div>
          <div class="dist-bar-track">
            <div class="dist-bar-fill" style="width:{pct}%;background:{clr}"></div>
          </div>
        </div>""")
    return "\n".join(parts)

def token_pills_html(kata_info):
    pill_map = {
        ("InSet",  "Positif"): "pill-pos",
        ("InSet",  "Negatif"): "pill-neg",
        ("Domain", "Positif"): "pill-pos",
        ("Domain", "Negatif"): "pill-neg",
        ("Domain", "Netral"):  "pill-dom",
        ("-",      "-"):       "pill-neu",
    }
    parts = []
    for k in kata_info:
        cls = pill_map.get((k["sumber"], k["polaritas"]), "pill-neu")
        parts.append(f'<span class="pill {cls}">{k["kata"]}</span>')
    return "".join(parts)

# SIDEBAR
with st.sidebar:
    st.markdown("""
    <div style="padding:0.75rem 0 0.5rem; text-align:center;">
      <h2 style="font-size:1.05rem;font-weight:700;margin:0;color:#111827">Analisis Sentimen Kebijakan WFH ASN</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown('<p style="font-size:0.75rem;font-weight:700;color:#9ca3af;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:8px">Kelas Sentimen</p>', unsafe_allow_html=True)
    for kelas, clr, desk in [
        ("Positif","#22c55e","Mendukung kebijakan WFH ASN"),
        ("Netral","#94a3b8","Informasi faktual / objektif"),
        ("Negatif","#ef4444","Menolak / mengkritik kebijakan WFH ASN"),
    ]:
        st.markdown(f"""
        <div style="display:flex;align-items:flex-start;gap:8px;margin-bottom:8px;
             padding:8px 10px;background:#f8f9fb;border-radius:8px;border:1px solid #e5e7eb">
          <div style="width:9px;height:9px;border-radius:50%;background:{clr};margin-top:4px;flex-shrink:0"></div>
          <div>
            <p style="margin:0;font-size:0.85rem;font-weight:600;color:#111827">{kelas}</p>
            <p style="margin:0;font-size:0.78rem;color:#9ca3af">{desk}</p>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.75rem;font-weight:700;color:#9ca3af;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:8px">Distribusi Label Dataset</p>', unsafe_allow_html=True)
    for lbl, n, pct, clr in [
        ("Positif",1404,55.2,"#22c55e"),
        ("Negatif",755,29.7,"#ef4444"),
        ("Netral",384,15.1,"#6366f1"),
    ]:
        st.markdown(f"""
        <div class="dist-bar-wrap">
          <div class="dist-bar-head">
            <span style="color:#374151">{lbl}</span>
            <span style="color:{clr};font-weight:700;font-family:'DM Mono',monospace">{n} ({pct}%)</span>
          </div>
          <div class="dist-bar-track"><div class="dist-bar-fill" style="width:{pct}%;background:{clr}"></div></div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.75rem;font-weight:700;color:#9ca3af;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:8px">Warna Kata</p>', unsafe_allow_html=True)
    for warna_hex, label_txt, keterangan in [
        ("#22c55e","Hijau","Kata positif (InSet / Domain)"),
        ("#ef4444","Merah","Kata negatif (InSet / Domain)"),
        ("#6366f1","Biru","Kata netral (InSet / Domain)"),
        ("#94a3b8","Abu","Kata tidak berkategori"),
    ]:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">
          <div style="width:28px;height:14px;border-radius:4px;background:{warna_hex};border:1px solid #e5e7eb;flex-shrink:0"></div>
          <div style="font-size:0.8rem;color:#374151"><strong>{label_txt}</strong> — {keterangan}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.75rem;color:#d1d5db;text-align:center">Novia Dwi Cahyanti (51423123)</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.75rem;color:#d1d5db;text-align:center">Universitas Gunadarma - 2026</p>', unsafe_allow_html=True)

# MAIN
st.markdown("""
<div class="hero">
  <h1>Analisis Sentimen Kebijakan WFH ASN</h1>
  <p>Berdasarkan Surat Edaran Menteri PANRB Nomor 3 Tahun 2026, pemerintah menetapkan kebijakan Work From Home bagi Aparatur Sipil Negara yang mulai berlaku pada 1 April 2026. Aplikasi website berbasis Streamlit ini digunakan untuk menganalisis sentimen opini publik terhadap kebijakan tersebut menggunakan algoritma Support Vector Machine (SVM) dengan data dari Twitter/X.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="stats-row">
  <div class="stat-box"><div class="lbl" style="text-align:center;">Dataset</div><div class="val" style="text-align:center;">2,543</div><div class="sub" style="text-align:center;">tweet terproses</div></div>
  <div class="stat-box"><div class="lbl" style="text-align:center;">Akurasi</div><div class="val" style="text-align:center;">91.4%</div><div class="sub" style="text-align:center;">split 80:20</div></div>
  <div class="stat-box"><div class="lbl" style="text-align:center;">Periode</div><div class="val" style="text-align:center;">26 Hari</div><div class="sub" style="text-align:center;">31 Mar – 26 Apr 2026</div></div>
  <div class="stat-box"><div class="lbl" style="text-align:center;">F1 Macro</div><div class="val" style="text-align:center;">86.9%</div><div class="sub" style="text-align:center;">3 kelas sentimen</div></div>
</div>
""", unsafe_allow_html=True)

if not MODEL_OK:
    st.error(f"Model tidak dapat dimuat: {ERR_MSG}")
    st.stop()

# TABS
tab1, tab2 = st.tabs(["  Analisis Teks  ", "  Analisis Multi-Teks  "])

# TAB 1 — ANALISIS TEKS
with tab1:
    col_input, col_hasil = st.columns([1, 1], gap="large")

    with col_input:
        st.markdown("#### Masukkan Teks")

        # Contoh teks 
        st.markdown('<p style="font-size:0.82rem;color:#9ca3af;margin-bottom:6px">Coba contoh teks (klik untuk isi):</p>', unsafe_allow_html=True)
        cols_c = st.columns(2)
        for i, c in enumerate(CONTOH):
            lbl = c[:36] + "…" if len(c) > 36 else c
            with cols_c[i % 2]:
                if st.button(lbl, key=f"ex_{i}", use_container_width=True):
                    st.session_state["input_teks"] = c
                    st.session_state["hasil_analisis"] = None
                    st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        # Widget key berbeda dari storage key agar tidak konflik

        input_text = st.text_area(
            "Teks:",
            value=st.session_state["input_teks"],
            height=140,
            placeholder="Tulis komentar tentang kebijakan WFH ASN di sini...",
            label_visibility="collapsed",
        )
        cb1, cb2 = st.columns([3, 1])
        with cb1:
            btn_analisis = st.button("Analisis Sentimen", type="primary",
                                     use_container_width=True, key="btn_go")
        with cb2:
            # Hapus mengosongkan storage key sehingga textarea ikut kosong
            
            if st.button("Hapus", use_container_width=True, key="btn_clr"):
                st.session_state["input_teks"] = ""
                st.session_state["hasil_analisis"] = None
                st.rerun()

        if btn_analisis:
            teks = input_text.strip()
            if not teks:
                st.warning("Teks tidak boleh kosong.")
            else:
                with st.spinner("Menganalisis..."):
                    st.session_state["hasil_analisis"] = predict_sentimen(teks)

        # Token pills — muncul di bawah input setelah ada hasil
        if st.session_state["hasil_analisis"]:
            h = st.session_state["hasil_analisis"]
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Token setelah preprocessing:**")
            st.markdown(token_pills_html(h["kata_info"]), unsafe_allow_html=True)
            st.markdown("""
            <p style="font-size:0.78rem;color:#9ca3af;margin-top:6px">
            🟢 Positif &nbsp;·&nbsp; 🔴 Negatif &nbsp;·&nbsp; 🔵 Netral &nbsp;·&nbsp; ⬜ Tidak berkategori
            </p>""", unsafe_allow_html=True)

    with col_hasil:
        st.markdown("#### Hasil Analisis")

        if not st.session_state["hasil_analisis"]:
            st.markdown("""
            <div style="background:#f8f9fb;border:2px dashed #e5e7eb;border-radius:12px;
                 padding:3.5rem 2rem;text-align:center;color:#9ca3af;margin-top:3.5rem">
              <p style="font-size:0.92rem;margin:8px 0 0">Masukkan teks dan klik <strong>Analisis Sentimen</strong></p>
              <p style="font-size:0.82rem;margin:6px 0 0;color:#cbd5e1">Hasil akan muncul di sini</p>
            </div>""", unsafe_allow_html=True)
        else:
            h     = st.session_state["hasil_analisis"]
            label = h["label"]
            proba = h["proba"]
            clr   = RES_COLOR[label]

            st.markdown(f"""
            <div class="result-wrap {RES_CLS[label]}">
              <p class="res-label" style="color:{clr}">{label}</p>
              <p class="res-meta">
                Skor InSet: <strong style="color:{clr}">{h['skor_inset']:+.2f}</strong>
                &nbsp;·&nbsp; Confidence: <strong style="color:{clr}">{proba.get(label,0)*100:.1f}%</strong>
                &nbsp;·&nbsp; Token: <strong>{len(h['tokens'])}</strong>
              </p>
            </div>""", unsafe_allow_html=True)

            # Confidence bars
            st.markdown('<div class="card"><p class="card-title">Rasio Skor per Kelas</p>', unsafe_allow_html=True)
            st.markdown(conf_bar_html(proba), unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Kontribusi kata
            st.markdown('<div class="card"><p class="card-title">Kontribusi Kata terhadap Kamus Sentimen</p>', unsafe_allow_html=True)
            st.markdown(kata_bars_html(h["kata_info"]), unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            with st.expander("Detail Preprocessing"):
                p1, p2 = st.columns(2)
                with p1:
                    st.markdown("**Pre-stem (InSet scoring):**")
                    st.code(h["pre_stem"] or "—", language=None)
                with p2:
                    st.markdown("**Post-stem (TF-IDF):**")
                    st.code(h["post_stem"] or "—", language=None)
                rows_tbl = [
                    {
                        "Kata": k["kata"],
                        "Sumber": k["sumber"],
                        "Skor InSet": f"{k['skor']:+.2f}" if k["skor"] is not None else "—",
                        "Polaritas": k["polaritas"],
                    }
                    for k in h["kata_info"] if k["polaritas"] != "-"
                ]
                if rows_tbl:
                    st.dataframe(pd.DataFrame(rows_tbl), use_container_width=True, hide_index=True)

# TAB 2 — MULTI-TEKS
with tab2:
    col_in2, col_out2 = st.columns([1, 1], gap="large")

    with col_in2:
        st.markdown("#### Masukkan Multi-Teks")
        st.markdown('<p style="font-size:0.82rem;color:#6b7280;margin-bottom:8px">Satu komentar per baris (maksimal 50 baris)</p>', unsafe_allow_html=True)

        # Tab2: widget key wgt_batch berbeda dari storage key batch_teks
        batch_val = st.text_area(
            "",
            value=st.session_state["batch_teks"],
            height=300,
            placeholder="WFH bikin jalanan Jakarta lebih lengang!\nASN WFH sama aja bolos kerja.\nPemerintah terapkan WFH tiap Jumat.",
            label_visibility="collapsed",
        )

        bb1, bb2 = st.columns([3, 1])
        with bb1:
            btn_batch = st.button("Analisis Semua Sentimen", type="primary",
                                   use_container_width=True, key="btn_batch_go")
        with bb2:
            # Hapus juga mengosongkan storage key tab2
            if st.button("Hapus", use_container_width=True, key="btn_batch_clr"):
                st.session_state["batch_teks"] = ""
                st.session_state["batch_rows"] = []
                st.rerun()

        if btn_batch:
            lines = [l.strip() for l in batch_val.strip().split("\n") if l.strip()]
            if not lines:
                st.warning("Masukkan minimal satu baris teks.")
            elif len(lines) > 50:
                st.warning("Maksimal 50 baris sekaligus.")
            else:
                with st.spinner(f"Memproses {len(lines)} komentar..."):
                    batch_rows = []
                    for line in lines:
                        r = predict_sentimen(line)
                        batch_rows.append({
                            "teks": line,
                            "label": r["label"],
                            "pct_pos": r["proba"].get("Positif", 0)*100,
                            "pct_neu": r["proba"].get("Netral",  0)*100,
                            "pct_neg": r["proba"].get("Negatif", 0)*100,
                            "skor": r["skor_inset"],
                        })
                st.session_state["batch_rows"] = batch_rows
                st.session_state["batch_teks"] = batch_val

    with col_out2:
        st.markdown("#### Hasil Analisis")

        if not st.session_state["batch_rows"]:
            st.markdown("""
            <div style="background:#f8f9fb;border:2px dashed #e5e7eb;border-radius:12px;
                 padding:3.5rem 2rem;text-align:center;color:#9ca3af;margin-top:3.5rem">
              <p style="font-size:0.92rem;margin:8px 0 0">Masukkan teks dan klik <strong>Analisis Semua Sentimen</strong></p>
              <p style="font-size:0.82rem;margin:6px 0 0;color:#cbd5e1">Hasil akan muncul di sini</p>
            </div>""", unsafe_allow_html=True)
        else:
            rows   = st.session_state["batch_rows"]
            counts = Counter(r["label"] for r in rows)
            total  = len(rows)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total", total)
            m2.metric("Positif", counts.get("Positif", 0))
            m3.metric("Netral",  counts.get("Netral",  0))
            m4.metric("Negatif", counts.get("Negatif", 0))

            # FIX: Distribusi bar menggunakan fungsi dist_bars_html(counts, total)
            # — tidak bergantung pada variabel `h` yang tidak terdefinisi di sini.
            st.markdown('<div class="card"><p class="card-title">Distribusi Sentimen</p>', unsafe_allow_html=True)
            st.markdown(dist_bars_html(counts, total), unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Daftar hasil per komentar
            badge_cls = {"Positif":"badge-pos","Netral":"badge-neu","Negatif":"badge-neg"}
            st.markdown('<div class="card"><p class="card-title">Hasil Per Komentar</p>', unsafe_allow_html=True)
            for r in rows:
                conf = max(r["pct_pos"], r["pct_neu"], r["pct_neg"])
                st.markdown(f"""
                <div class="batch-item">
                  <div class="batch-txt">{r["teks"][:130]}{"…" if len(r["teks"])>130 else ""}</div>
                  <div style="text-align:right;flex-shrink:0;min-width:90px">
                    <span class="badge {badge_cls[r['label']]}">{r['label']}</span>
                    <div style="font-size:0.75rem;color:#9ca3af;margin-top:3px;font-family:'DM Mono',monospace">{conf:.1f}%</div>
                  </div>
                </div>""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            df_dl = pd.DataFrame([{
                "Komentar": r["teks"], "Sentimen": r["label"],
                "% Positif": f"{r['pct_pos']:.1f}", "% Netral": f"{r['pct_neu']:.1f}",
                "% Negatif": f"{r['pct_neg']:.1f}", "Skor InSet": f"{r['skor']:+.2f}",
            } for r in rows])
            st.download_button(
                "Download CSV",
                data=df_dl.to_csv(index=False).encode("utf-8-sig"),
                file_name="hasil_sentimen_multiteks.csv",
                mime="text/csv",
            )

# FOOTER
st.markdown("""
<div style="margin-top:2.5rem;padding-top:1rem;border-top:1px solid #e5e7eb;
     display:flex;justify-content:space-between;flex-wrap:wrap;gap:0.5rem">
  <p style="font-size:0.78rem;color:#9ca3af;margin:0;font-family:'DM Mono',monospace">
    Analisis Sentimen Kebijakan WFH ASN Tahun 2026
  </p>
  <p style="font-size:0.78rem;color:#9ca3af;margin:0">Universitas Gunadarma</p>
</div>
""", unsafe_allow_html=True)
