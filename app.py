import io
import re
import warnings
import traceback

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import nltk
import numpy as np
import pandas as pd
import streamlit as st
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
MAX_BYTES = 50 * 1024 * 1024
MAX_ROWS  = 100_000

# ─────────────────────────────────────────────
# CSS  — dark editorial dashboard
# ─────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=Outfit:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

/* ── Reset & base ─────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    font-family: "Times New Roman", Times, serif;
    font-size: 15px;
}

/* ── App shell ───────────────────────────── */
.stApp {
    background: #0c0e14;
    color: #c9d1e0;
    min-height: 100vh;
}

/* ── Hide Streamlit chrome ───────────────── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 0 2.5rem 3rem !important;
    max-width: 1200px !important;
}

/* ── Sidebar ─────────────────────────────── */
section[data-testid="stSidebar"] { display: none; }

/* ── Hero bar ────────────────────────────── */
.hero {
    position: relative;
    padding: 3rem 0 2.2rem;
    margin-bottom: 0.5rem;
    border-bottom: 1px solid #1e2330;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: 0; left: -10%; right: -10%; height: 2px;
    background: linear-gradient(90deg,
        transparent 0%,
        #e8a020 20%,
        #f0c060 50%,
        #e8a020 80%,
        transparent 100%
    );
    animation: shimmer 4s ease-in-out infinite;
}
@keyframes shimmer {
    0%, 100% { opacity: 0.5; }
    50%       { opacity: 1;   }
}

.hero-eyebrow {
    font-family: "Times New Roman", Times, serif;
    font-size: 0.68rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #e8a020;
    margin-bottom: 0.7rem;
}
.hero-title {
    font-family: "Times New Roman", Times, serif;
    font-size: 3rem;
    font-weight: 800;
    line-height: 1;
    color: #f0f4ff;
    letter-spacing: -0.04em;
    margin-bottom: 0.7rem;
}
.hero-sub {
    font-size: 0.92rem;
    color: #5a6480;
    max-width: 520px;
    line-height: 1.65;
    font-weight: 300;
}

/* ── Section label ───────────────────────── */
.label {
    font-family: "Times New Roman", Times, serif;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #3d4560;
    margin: 2rem 0 0.55rem;
}

/* ── Info strip (upload rules) ───────────── */
.info-strip {
    display: flex;
    gap: 0;
    border: 1px solid #1e2330;
    border-radius: 10px;
    overflow: hidden;
    margin-bottom: 1.4rem;
    background: #0f111a;
}
.info-cell {
    flex: 1;
    padding: 0.85rem 1.1rem;
    border-right: 1px solid #1e2330;
}
.info-cell:last-child { border-right: none; }
.info-cell .ic-label {
    font-family: "Times New Roman", Times, serif;
    font-size: 0.6rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #3d4560;
    margin-bottom: 0.3rem;
}
.info-cell .ic-value {
    font-size: 0.85rem;
    font-weight: 500;
    color: #8b95b0;
}

/* ── File uploader ───────────────────────── */
[data-testid="stFileUploadDropzone"] {
    background: #0f111a !important;
    border: 1.5px dashed #252a3a !important;
    border-radius: 12px !important;
    transition: border-color 0.25s, background 0.25s;
}
[data-testid="stFileUploadDropzone"]:hover {
    border-color: #e8a020 !important;
    background: #12141f !important;
}
[data-testid="stFileUploadDropzone"] p {
    color: #3d4560 !important;
    font-size: 0.85rem !important;
}
[data-testid="stFileUploadDropzone"] svg { stroke: #3d4560 !important; }

/* ── Select box ──────────────────────────── */
[data-baseweb="select"] > div {
    background: #0f111a !important;
    border: 1px solid #1e2330 !important;
    border-radius: 8px !important;
    color: #c9d1e0 !important;
    font-family: "Times New Roman", Times, serif;
    font-size: 0.88rem !important;
    transition: border-color 0.2s;
}
[data-baseweb="select"] > div:hover,
[data-baseweb="select"] > div:focus-within {
    border-color: #e8a020 !important;
}
[data-baseweb="select"] svg { fill: #3d4560 !important; }

/* Select dropdown menu */
[data-baseweb="popover"] [role="listbox"] {
    background: #0f111a !important;
    border: 1px solid #1e2330 !important;
    border-radius: 8px !important;
}
[data-baseweb="popover"] [role="option"] {
    color: #8b95b0 !important;
    font-size: 0.88rem !important;
    background: transparent !important;
}
[data-baseweb="popover"] [role="option"]:hover {
    background: #161926 !important;
    color: #e8a020 !important;
}

/* ── Primary button ──────────────────────── */
.stButton > button {
    background: #e8a020 !important;
    color: #0c0e14 !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: "Times New Roman", Times, serif;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 0.6rem 2rem !important;
    letter-spacing: 0.02em;
    transition: background 0.2s, transform 0.15s;
    cursor: pointer;
}
.stButton > button:hover {
    background: #f0b030 !important;
    transform: translateY(-1px);
}
.stButton > button:active { transform: translateY(0); }

/* ── Download button ─────────────────────── */
.stDownloadButton > button {
    background: transparent !important;
    color: #e8a020 !important;
    border: 1.5px solid #e8a020 !important;
    border-radius: 8px !important;
    font-family: "Times New Roman", Times, serif;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    padding: 0.55rem 1.6rem !important;
    transition: background 0.2s, color 0.2s;
}
.stDownloadButton > button:hover {
    background: #e8a020 !important;
    color: #0c0e14 !important;
}

/* ── Progress bar ────────────────────────── */
[data-testid="stProgress"] > div > div > div {
    background: linear-gradient(90deg, #e8a020, #f0c060) !important;
    border-radius: 999px !important;
}
[data-testid="stProgress"] > div > div {
    background: #1e2330 !important;
    border-radius: 999px !important;
    height: 4px !important;
}

/* ── Alerts ──────────────────────────────── */
[data-testid="stAlert"] {
    background: #0f111a !important;
    border: 1px solid #1e2330 !important;
    border-radius: 10px !important;
    color: #8b95b0 !important;
    font-size: 0.85rem !important;
}
.stSuccess > div {
    background: #0c1810 !important;
    border-color: #1a4025 !important;
    color: #4ade80 !important;
}
.stError > div {
    background: #180c0c !important;
    border-color: #401a1a !important;
    color: #f87171 !important;
}

/* ── Metric cards ────────────────────────── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 0.75rem;
    margin: 0.5rem 0 1.5rem;
}
.mc {
    background: #0f111a;
    border: 1px solid #1e2330;
    border-radius: 12px;
    padding: 1.1rem 1rem 1rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.mc:hover { border-color: #2a3050; }
.mc::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
}
.mc.pos::after { background: #22c55e; }
.mc.neu::after { background: #e8a020; }
.mc.neg::after { background: #ef4444; }
.mc.tot::after { background: #6366f1; }
.mc.avg::after { background: #38bdf8; }

.mc-label {
    font-family: "Times New Roman", Times, serif;
    font-size: 0.6rem;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    color: #3d4560;
    margin-bottom: 0.55rem;
}
.mc-value {
    font-family: "Times New Roman", Times, serif;
    font-size: 2rem;
    font-weight: 700;
    line-height: 1;
    color: #f0f4ff;
    margin-bottom: 0.3rem;
}
.mc-sub {
    font-size: 0.72rem;
    color: #3d4560;
    font-weight: 400;
}
.mc.pos .mc-value { color: #22c55e; }
.mc.neg .mc-value { color: #ef4444; }
.mc.neu .mc-value { color: #e8a020; }
.mc.avg .mc-value { color: #38bdf8; }

/* ── DataFrame ───────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid #1e2330 !important;
    border-radius: 10px !important;
    overflow: hidden;
}
[data-testid="stDataFrame"] iframe { background: #0f111a !important; }

/* ── Tabs ────────────────────────────────── */
[data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #1e2330 !important;
    gap: 0 !important;
}
[data-baseweb="tab"] {
    background: transparent !important;
    color: #3d4560 !important;
    font-family: "Times New Roman", Times, serif;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    padding: 0.6rem 1.2rem !important;
    border-bottom: 2px solid transparent !important;
    transition: color 0.2s, border-color 0.2s;
}
[data-baseweb="tab"]:hover { color: #8b95b0 !important; }
[aria-selected="true"][data-baseweb="tab"] {
    color: #e8a020 !important;
    border-bottom-color: #e8a020 !important;
}
[data-baseweb="tab-panel"] { padding: 1.2rem 0 0 !important; }

/* ── Spinner ─────────────────────────────── */
.stSpinner > div > div {
    border-color: #e8a020 transparent transparent transparent !important;
}

/* ── Scrollbar ───────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #0c0e14; }
::-webkit-scrollbar-thumb { background: #1e2330; border-radius: 999px; }
::-webkit-scrollbar-thumb:hover { background: #2a3050; }

/* ── Preview table wrapper ───────────────── */
.preview-wrap {
    background: #0f111a;
    border: 1px solid #1e2330;
    border-radius: 10px;
    padding: 0.1rem;
    overflow: hidden;
}

/* ── Divider ─────────────────────────────── */
.divider {
    height: 1px;
    background: #1e2330;
    margin: 2rem 0;
}
</style>
"""

# ─────────────────────────────────────────────
# PLOT THEME  (matches dark UI)
# ─────────────────────────────────────────────
BG      = "#0c0e14"
CARD    = "#0f111a"
BORDER  = "#1e2330"
INK     = "#8b95b0"
INK2    = "#3d4560"
GOLD    = "#e8a020"
GREEN   = "#22c55e"
RED     = "#ef4444"
AMBER   = "#f59e0b"

SENT_COLORS = {"Positive": GREEN, "Neutral": AMBER, "Negative": RED}

def _apply_theme(fig, axes):
    fig.patch.set_facecolor(BG)
    for ax in (axes if hasattr(axes, "__iter__") else [axes]):
        ax.set_facecolor(CARD)
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER)
        ax.tick_params(colors=INK, labelsize=8.5)
        ax.xaxis.label.set_color(INK)
        ax.yaxis.label.set_color(INK)
        ax.title.set_color("#c9d1e0")
        ax.grid(color=BORDER, linewidth=0.6, linestyle="-", zorder=0)
        ax.set_axisbelow(True)


def chart_distribution(df) -> plt.Figure:
    counts = (
        df["Sentiment"]
        .value_counts()
        .reindex(["Positive", "Neutral", "Negative"], fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    colors = [SENT_COLORS[k] for k in counts.index]
    bars = ax.bar(counts.index, counts.values,
                  color=colors, width=0.38,
                  zorder=3, edgecolor="none",
                  linewidth=0)
    # Glow effect — faint wider bar behind
    for b, c in zip(bars, colors):
        ax.bar(b.get_x() + b.get_width() / 2,
               b.get_height(),
               width=0.55, color=c, alpha=0.12,
               zorder=2, edgecolor="none")
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2,
                b.get_height() + counts.max() * 0.03,
                f"{v:,}",
                ha="center", va="bottom",
                color="#f0f4ff", fontsize=10.5,
                fontweight="600",
                fontfamily="Outfit")
    ax.set_ylabel("Count", labelpad=8, fontsize=8.5)
    ax.set_title("Sentiment Distribution",
                 fontsize=11, fontweight="600", pad=14,
                 fontfamily="Outfit")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    _apply_theme(fig, ax)
    fig.tight_layout(pad=1.4)
    return fig


def chart_histogram(df) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    n, bins, patches = ax.hist(df["ensemble_score"], bins=45,
                               color="#6366f1", edgecolor="none",
                               alpha=0.9, zorder=3)
    # Colour bars by bucket
    for patch, left in zip(patches, bins[:-1]):
        if left >= 0.05:
            patch.set_facecolor(GREEN)
            patch.set_alpha(0.85)
        elif left <= -0.05:
            patch.set_facecolor(RED)
            patch.set_alpha(0.85)
        else:
            patch.set_facecolor(AMBER)
            patch.set_alpha(0.85)
    ax.axvline( 0.05, color="#f0f4ff", lw=1, linestyle="--", alpha=0.3)
    ax.axvline(-0.05, color="#f0f4ff", lw=1, linestyle="--", alpha=0.3)
    ax.set_xlabel("Ensemble Score", labelpad=8, fontsize=8.5)
    ax.set_ylabel("Frequency",      labelpad=8, fontsize=8.5)
    ax.set_title("Score Distribution",
                 fontsize=11, fontweight="600", pad=14,
                 fontfamily="Outfit")
    _apply_theme(fig, ax)
    fig.tight_layout(pad=1.4)
    return fig


def chart_confidence(df) -> plt.Figure:
    ct = (
        df.groupby(["Sentiment", "Confidence"])
        .size()
        .unstack(fill_value=0)
        .reindex(["Positive", "Neutral", "Negative"])
    )
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    conf_colors = {"High": "#f0f4ff", "Medium": "#5a6480", "Low": "#2a3050"}
    bottom = np.zeros(len(ct))
    for level in ["High", "Medium", "Low"]:
        if level in ct.columns:
            vals = ct[level].fillna(0).values
            ax.bar(ct.index, vals, bottom=bottom,
                   label=level,
                   color=[conf_colors[level]] * len(ct),
                   width=0.38, zorder=3, edgecolor="none")
            bottom += vals
    ax.set_ylabel("Count", labelpad=8, fontsize=8.5)
    ax.set_title("Confidence Breakdown",
                 fontsize=11, fontweight="600", pad=14,
                 fontfamily="Outfit")
    leg = ax.legend(framealpha=0, labelcolor=INK, fontsize=8,
                    title="Confidence", title_fontsize=8,
                    loc="upper right")
    leg.get_title().set_color(INK2)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    _apply_theme(fig, ax)
    fig.tight_layout(pad=1.4)
    return fig


def chart_scatter(df) -> plt.Figure:
    sample = df.sample(min(800, len(df)), random_state=42)
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for sent, color in SENT_COLORS.items():
        mask = sample["Sentiment"] == sent
        ax.scatter(
            sample.loc[mask, "ensemble_score"],
            sample.loc[mask, "subjectivity"],
            c=color, s=9, alpha=0.55,
            edgecolors="none", label=sent, zorder=3,
        )
    ax.set_xlabel("Ensemble Score",  labelpad=8, fontsize=8.5)
    ax.set_ylabel("Subjectivity",    labelpad=8, fontsize=8.5)
    ax.set_title("Score vs Subjectivity",
                 fontsize=11, fontweight="600", pad=14,
                 fontfamily="Outfit")
    leg = ax.legend(framealpha=0, labelcolor=INK, fontsize=8)
    _apply_theme(fig, ax)
    fig.tight_layout(pad=1.4)
    return fig


def chart_wordcloud(df, sentiment: str, cmap: str):
    text = " ".join(
        df.loc[df["Sentiment"] == sentiment, "_clean_"].dropna().tolist()
    )
    if not text.strip():
        return None
    wc = WordCloud(
        width=640, height=300,
        background_color="#0f111a",
        colormap=cmap,
        max_words=110,
        collocations=False,
        stopwords=STOPWORDS,
        prefer_horizontal=0.8,
    ).generate(text)
    fig, ax = plt.subplots(figsize=(5.5, 2.9))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    color_map = {"Positive": GREEN, "Negative": RED, "Neutral": AMBER}
    ax.set_title(
        f"{sentiment} · Word Frequency",
        fontsize=10.5, fontweight="600", pad=10,
        color=color_map.get(sentiment, "#c9d1e0"),
        fontfamily="Outfit",
    )
    fig.patch.set_facecolor(BG)
    fig.tight_layout(pad=0.4)
    return fig


# ─────────────────────────────────────────────
# NLP
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_sia():
    for pkg in ("vader_lexicon", "stopwords", "punkt", "punkt_tab"):
        nltk.download(pkg, quiet=True)
    return SentimentIntensityAnalyzer()


_HTML  = re.compile(r"<[^>]+>")
_URL   = re.compile(r"https?://\S+|www\.\S+")
_SPACE = re.compile(r"\s{2,}")

def clean(raw) -> str:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return ""
    t = str(raw).strip()
    if t.lower() in {"nan", "none", "null", "n/a", "na", ""}:
        return ""
    t = _HTML.sub(" ", t)
    t = _URL.sub(" ", t)
    return _SPACE.sub(" ", t).strip()


def score_row(text: str, sia) -> dict:
    v        = sia.polarity_scores(text)
    tb       = TextBlob(text).sentiment
    ensemble = round(0.55 * v["compound"] + 0.45 * tb.polarity, 4)
    sentiment = (
        "Positive" if ensemble >=  0.05 else
        "Negative" if ensemble <= -0.05 else
        "Neutral"
    )
    confidence = (
        "High"   if abs(ensemble) >= 0.60 else
        "Medium" if abs(ensemble) >= 0.30 else
        "Low"
    )
    return {
        "vader_score":    round(v["compound"],      4),
        "textblob_score": round(tb.polarity,        4),
        "subjectivity":   round(tb.subjectivity,    4),
        "ensemble_score": ensemble,
        "Sentiment":      sentiment,
        "Confidence":     confidence,
    }


def analyse(df, col, sia, bar, status):
    n, rows = len(df), []
    for i, raw in enumerate(df[col]):
        t = clean(raw)
        rows.append(
            score_row(t, sia) if t else
            {"vader_score": 0.0, "textblob_score": 0.0,
             "subjectivity": 0.0, "ensemble_score": 0.0,
             "Sentiment": "Neutral", "Confidence": "Low"}
        )
        if (i + 1) % 500 == 0 or (i + 1) == n:
            bar.progress((i + 1) / n)
            status.markdown(
                f"<span style='font-family: Times New Roman, Times, serif;font-size:0.75rem;"
                f"color:#3d4560;'>{i + 1:,} / {n:,} rows processed</span>",
                unsafe_allow_html=True,
            )
    result = pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    result["_clean_"] = [clean(r) for r in df[col]]
    return result


# ─────────────────────────────────────────────
# FILE I/O
# ─────────────────────────────────────────────

def read_file(f):
    if f.size > MAX_BYTES:
        return None, f"File is {f.size / 1e6:.1f} MB — exceeds the 50 MB limit."
    name = f.name.lower()
    buf  = io.BytesIO(f.read())
    try:
        if name.endswith(".csv"):
            for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
                try:
                    buf.seek(0)
                    df = pd.read_csv(buf, encoding=enc, low_memory=False)
                    break
                except (UnicodeDecodeError, pd.errors.ParserError):
                    continue
            else:
                return None, "Could not decode CSV. Save as UTF-8 and retry."
        elif name.endswith((".xlsx", ".xls")):
            engine = "openpyxl" if name.endswith(".xlsx") else "xlrd"
            df = pd.read_excel(buf, engine=engine)
        else:
            return None, "Unsupported format. Upload CSV, XLSX, or XLS."
    except Exception as e:
        return None, f"Read error: {e}"
    df = df.dropna(how="all").reset_index(drop=True)
    if df.empty:
        return None, "File is empty."
    if len(df) > MAX_ROWS:
        return None, f"File has {len(df):,} rows — maximum is {MAX_ROWS:,}."
    return df, ""


# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Sentiment Analyzer",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(CSS, unsafe_allow_html=True)

if "results_df" not in st.session_state:
    st.session_state.results_df = None

sia = load_sia()

# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────

st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">Natural Language Processing</div>
    <div class="hero-title">Sentiment Analyzer</div>
    <div class="hero-sub">
        Upload a dataset, pick your text column, and get a full sentiment
        breakdown — scores, charts, and a downloadable report.
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# UPLOAD RULES
# ─────────────────────────────────────────────

st.markdown("<div class='label'>Requirements</div>", unsafe_allow_html=True)
st.markdown("""
<div class="info-strip">
    <div class="info-cell">
        <div class="ic-label">Accepted formats</div>
        <div class="ic-value">CSV &nbsp;/&nbsp; XLSX &nbsp;/&nbsp; XLS</div>
    </div>
    <div class="info-cell">
        <div class="ic-label">Max file size</div>
        <div class="ic-value">50 MB</div>
    </div>
    <div class="info-cell">
        <div class="ic-label">Max rows</div>
        <div class="ic-value">1,00,000</div>
    </div>
    <div class="info-cell">
        <div class="ic-label">Required column</div>
        <div class="ic-value">Any text column</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# UPLOAD
# ─────────────────────────────────────────────

st.markdown("<div class='label'>Upload</div>", unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Drop your file here or click to browse",
    type=["csv", "xlsx", "xls"],
    label_visibility="collapsed",
)

if uploaded:
    df_raw, err = read_file(uploaded)
    if err:
        st.error(err)
        st.stop()

    st.success(
        f"Loaded **{len(df_raw):,} rows** across **{len(df_raw.columns)} columns** "
        f"from `{uploaded.name}`"
    )

    # Auto-detect text column
    hints = {"text", "review", "comment", "description", "body",
             "content", "message", "feedback", "tweet", "post", "summary"}
    default = next(
        (c for c in df_raw.columns if any(h in c.lower() for h in hints)),
        df_raw.columns[0],
    )

    st.markdown("<div class='label'>Column selection</div>", unsafe_allow_html=True)
    col_a, col_b = st.columns([1, 2])

    with col_a:
        text_col = st.selectbox(
            "Choose the text column to analyse",
            options=df_raw.columns.tolist(),
            index=df_raw.columns.tolist().index(default),
            label_visibility="collapsed",
        )
        st.markdown(
            f"<span style='font-family: Times New Roman, Times, serif;font-size:0.72rem;"
            f"color:#3d4560;'>{len(df_raw):,} rows will be scored</span>",
            unsafe_allow_html=True,
        )

    with col_b:
        st.markdown("<div class='preview-wrap'>", unsafe_allow_html=True)
        st.dataframe(
            df_raw[[text_col]].head(5).rename(columns={text_col: "Sample text"}),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Run Analysis"):
        bar    = st.progress(0.0)
        status = st.empty()
        try:
            st.session_state.results_df = analyse(df_raw, text_col, sia, bar, status)
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            st.code(traceback.format_exc())
            st.stop()
        bar.empty()
        status.empty()

# ─────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────

if st.session_state.results_df is not None:
    df    = st.session_state.results_df
    total = len(df)
    n_pos = (df["Sentiment"] == "Positive").sum()
    n_neu = (df["Sentiment"] == "Neutral").sum()
    n_neg = (df["Sentiment"] == "Negative").sum()
    avg   = df["ensemble_score"].mean()

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='label'>Summary</div>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metric-grid">
        <div class="mc tot">
            <div class="mc-label">Total rows</div>
            <div class="mc-value">{total:,}</div>
            <div class="mc-sub">records analysed</div>
        </div>
        <div class="mc pos">
            <div class="mc-label">Positive</div>
            <div class="mc-value">{n_pos:,}</div>
            <div class="mc-sub">{n_pos/total:.1%} of total</div>
        </div>
        <div class="mc neu">
            <div class="mc-label">Neutral</div>
            <div class="mc-value">{n_neu:,}</div>
            <div class="mc-sub">{n_neu/total:.1%} of total</div>
        </div>
        <div class="mc neg">
            <div class="mc-label">Negative</div>
            <div class="mc-value">{n_neg:,}</div>
            <div class="mc-sub">{n_neg/total:.1%} of total</div>
        </div>
        <div class="mc avg">
            <div class="mc-label">Avg Score</div>
            <div class="mc-value">{avg:+.3f}</div>
            <div class="mc-sub">ensemble mean</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Data table ──────────────────────────
    st.markdown("<div class='label'>Results preview &mdash; first 25 rows</div>", unsafe_allow_html=True)
    show_cols = [c for c in df.columns if c != "_clean_"]
    st.dataframe(df[show_cols].head(25), use_container_width=True, hide_index=True)

    # ── Charts ──────────────────────────────
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='label'>Charts</div>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Distribution", "Score breakdown", "Word clouds"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(chart_distribution(df), use_container_width=True)
        with c2:
            st.pyplot(chart_scatter(df), use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(chart_histogram(df), use_container_width=True)
        with c2:
            st.pyplot(chart_confidence(df), use_container_width=True)

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            fig = chart_wordcloud(df, "Positive", "YlGn")
            if fig:
                st.pyplot(fig, use_container_width=True)
            else:
                st.markdown(
                    "<span style='color:#3d4560;font-size:0.82rem;'>"
                    "No positive texts to display.</span>",
                    unsafe_allow_html=True,
                )
        with c2:
            fig = chart_wordcloud(df, "Negative", "OrRd")
            if fig:
                st.pyplot(fig, use_container_width=True)
            else:
                st.markdown(
                    "<span style='color:#3d4560;font-size:0.82rem;'>"
                    "No negative texts to display.</span>",
                    unsafe_allow_html=True,
                )

    # ── Download ────────────────────────────
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='label'>Export</div>", unsafe_allow_html=True)

    export = df[[c for c in df.columns if c != "_clean_"]]

    dl_col, _ = st.columns([1, 3])
    with dl_col:
        st.download_button(
            label="Download results as CSV",
            data=export.to_csv(index=False).encode("utf-8"),
            file_name="sentiment_results.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown(
        "<span style='font-family: Times New Roman, Times, serif;font-size:0.7rem;color:#2a3050;'>"
        f"Model: VADER 55% + TextBlob 45% ensemble &nbsp;|&nbsp; "
        f"{total:,} rows scored"
        "</span>",
        unsafe_allow_html=True,
    )
