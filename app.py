import io
import re
import warnings
import traceback

import matplotlib.pyplot as plt
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

PALETTE = {
    "Positive": "#22c55e",
    "Neutral":  "#f59e0b",
    "Negative": "#ef4444",
}

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

.stApp { background: #f8f9fb; color: #1a1a2e; }

section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e5e7eb;
}

.page-title {
    font-size: 1.75rem;
    font-weight: 600;
    color: #1a1a2e;
    letter-spacing: -0.03em;
    margin-bottom: 0.15rem;
}
.page-sub {
    color: #6b7280;
    font-size: 0.88rem;
    margin-bottom: 1.4rem;
}

.rule-box {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-left: 3px solid #6366f1;
    border-radius: 8px;
    padding: 0.85rem 1.1rem;
    font-size: 0.82rem;
    color: #6b7280;
    margin-bottom: 1.2rem;
    line-height: 1.75;
}
.rule-box b { color: #374151; }

.section-label {
    font-size: 0.68rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #9ca3af;
    margin: 1.5rem 0 0.45rem;
}

.metric-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 0.85rem 1rem;
    text-align: center;
}
.metric-card .m-label {
    font-size: 0.67rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #9ca3af;
    margin-bottom: 0.28rem;
}
.metric-card .m-value {
    font-size: 1.65rem;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
    line-height: 1;
}
.metric-card .m-pct {
    font-size: 0.73rem;
    color: #9ca3af;
    margin-top: 0.18rem;
}

.stDownloadButton > button {
    background: #1a1a2e;
    color: #ffffff;
    border: none;
    border-radius: 7px;
    font-weight: 500;
    font-size: 0.85rem;
    padding: 0.5rem 1.4rem;
}
.stDownloadButton > button:hover { background: #2d2d4e; }

.stButton > button {
    border-radius: 7px;
    font-weight: 500;
    font-size: 0.85rem;
}

div[data-testid="stDataFrame"] { border-radius: 8px; }
</style>
"""

# ─────────────────────────────────────────────
# NLP SETUP
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_sia() -> SentimentIntensityAnalyzer:
    for pkg in ("vader_lexicon", "stopwords", "punkt", "punkt_tab"):
        nltk.download(pkg, quiet=True)
    return SentimentIntensityAnalyzer()


# ─────────────────────────────────────────────
# TEXT CLEANING
# ─────────────────────────────────────────────

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
    t = _SPACE.sub(" ", t).strip()
    return t


# ─────────────────────────────────────────────
# SCORING  (VADER 55% + TextBlob 45%)
# ─────────────────────────────────────────────

def score_row(text: str, sia: SentimentIntensityAnalyzer) -> dict:
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
        "vader_score":    round(v["compound"], 4),
        "textblob_score": round(tb.polarity,  4),
        "subjectivity":   round(tb.subjectivity, 4),
        "ensemble_score": ensemble,
        "Sentiment":      sentiment,
        "Confidence":     confidence,
    }


def analyse(df: pd.DataFrame, col: str, sia, bar, status) -> pd.DataFrame:
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
            status.text(f"Processed {i + 1:,} of {n:,} rows")

    result = pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    # Store clean text for word clouds only (not exported)
    result["_clean_"] = [clean(r) for r in df[col]]
    return result


# ─────────────────────────────────────────────
# FILE READING
# ─────────────────────────────────────────────

def read_file(f):
    if f.size > MAX_BYTES:
        return None, f"File is {f.size / 1e6:.1f} MB — the 50 MB limit is exceeded."

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
                return None, "CSV could not be decoded. Please save the file as UTF-8."
        elif name.endswith((".xlsx", ".xls")):
            engine = "openpyxl" if name.endswith(".xlsx") else "xlrd"
            df = pd.read_excel(buf, engine=engine)
        else:
            return None, "Unsupported format. Upload a CSV, XLSX, or XLS file."
    except Exception as e:
        return None, f"Read error: {e}"

    df = df.dropna(how="all").reset_index(drop=True)
    if df.empty:
        return None, "The uploaded file is empty."
    if len(df) > MAX_ROWS:
        return None, (
            f"The file has {len(df):,} rows. "
            f"The maximum allowed is {MAX_ROWS:,} rows."
        )
    return df, ""


# ─────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────

_BG   = "#f8f9fb"
_CARD = "#ffffff"
_GRID = "#f1f3f5"
_INK  = "#374151"

def _theme(fig, axes):
    fig.patch.set_facecolor(_BG)
    axlist = axes if hasattr(axes, "__iter__") else [axes]
    for ax in axlist:
        ax.set_facecolor(_CARD)
        ax.tick_params(colors=_INK, labelsize=9)
        ax.xaxis.label.set_color(_INK)
        ax.yaxis.label.set_color(_INK)
        ax.title.set_color(_INK)
        for sp in ax.spines.values():
            sp.set_edgecolor("#e5e7eb")
        ax.grid(color=_GRID, linewidth=0.8, linestyle="-", zorder=0)
        ax.set_axisbelow(True)


def fig_distribution(df) -> plt.Figure:
    counts = (
        df["Sentiment"]
        .value_counts()
        .reindex(["Positive", "Neutral", "Negative"], fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(5, 3.4))
    bars = ax.bar(
        counts.index, counts.values,
        color=[PALETTE[k] for k in counts.index],
        width=0.42, zorder=3, edgecolor="none",
    )
    for b, v in zip(bars, counts.values):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + counts.max() * 0.025,
            f"{v:,}", ha="center", va="bottom",
            color=_INK, fontsize=10, fontweight="600",
        )
    ax.set_ylabel("Count", labelpad=6, fontsize=9)
    ax.set_title("Sentiment Distribution", fontsize=11, fontweight="600", pad=10)
    _theme(fig, ax)
    fig.tight_layout()
    return fig


def fig_histogram(df) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.hist(df["ensemble_score"], bins=40,
            color="#6366f1", edgecolor="none", alpha=0.85, zorder=3)
    ax.axvline( 0.05, color="#22c55e", lw=1.4, linestyle="--", label="Positive boundary")
    ax.axvline(-0.05, color="#ef4444", lw=1.4, linestyle="--", label="Negative boundary")
    ax.set_xlabel("Ensemble Score", labelpad=6, fontsize=9)
    ax.set_ylabel("Frequency",      labelpad=6, fontsize=9)
    ax.set_title("Score Distribution", fontsize=11, fontweight="600", pad=10)
    ax.legend(framealpha=0, labelcolor=_INK, fontsize=8)
    _theme(fig, ax)
    fig.tight_layout()
    return fig


def fig_wordcloud(df, sentiment: str, cmap: str):
    text = " ".join(
        df.loc[df["Sentiment"] == sentiment, "_clean_"].dropna().tolist()
    )
    if not text.strip():
        return None
    wc = WordCloud(
        width=560, height=260,
        background_color="white",
        colormap=cmap,
        max_words=100,
        collocations=False,
        stopwords=STOPWORDS,
    ).generate(text)
    fig, ax = plt.subplots(figsize=(5, 2.6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(
        f"{sentiment} Keywords",
        fontsize=11, fontweight="600", color=_INK, pad=8,
    )
    fig.patch.set_facecolor(_BG)
    fig.tight_layout(pad=0)
    return fig


# ─────────────────────────────────────────────
# PAGE INIT
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Sentiment Analyzer",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Persist results across Streamlit reruns (including download clicks)
if "results_df" not in st.session_state:
    st.session_state.results_df = None

sia = load_sia()

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

st.markdown("<div class='page-title'>Sentiment Analyzer</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='page-sub'>"
    "Upload a dataset, choose your text column, and get sentiment scores with charts instantly."
    "</div>",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# UPLOAD RULES BOX
# ─────────────────────────────────────────────

st.markdown(
    """
    <div class="rule-box">
        <b>Accepted formats</b>&nbsp;&nbsp;CSV &nbsp;&middot;&nbsp; XLSX &nbsp;&middot;&nbsp; XLS
        &nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;
        <b>Max file size</b>&nbsp;&nbsp;50 MB
        &nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;
        <b>Max rows</b>&nbsp;&nbsp;1,00,000
        <br>
        Your file must contain at least one column with text.
        Every other column is preserved in the downloaded report.
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# FILE UPLOAD
# ─────────────────────────────────────────────

uploaded = st.file_uploader(
    "Upload your dataset",
    type=["csv", "xlsx", "xls"],
    help="CSV, XLSX, or XLS — max 50 MB, max 1,00,000 rows",
)

if uploaded:
    df_raw, err = read_file(uploaded)
    if err:
        st.error(err)
        st.stop()

    st.success(
        f"Loaded {len(df_raw):,} rows and {len(df_raw.columns)} columns "
        f"from {uploaded.name}"
    )

    # Auto-detect most likely text column
    hints   = {"text", "review", "comment", "description", "body",
               "content", "message", "feedback", "tweet", "post", "summary"}
    default = next(
        (c for c in df_raw.columns if any(h in c.lower() for h in hints)),
        df_raw.columns[0],
    )

    col_pick, col_prev = st.columns([2, 3])
    with col_pick:
        st.markdown("<div class='section-label'>Select text column</div>", unsafe_allow_html=True)
        text_col = st.selectbox(
            "Text column",
            options=df_raw.columns.tolist(),
            index=df_raw.columns.tolist().index(default),
            label_visibility="collapsed",
        )
    with col_prev:
        st.markdown("<div class='section-label'>Column preview</div>", unsafe_allow_html=True)
        st.dataframe(
            df_raw[[text_col]].head(4).rename(columns={text_col: "Sample text"}),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Run Analysis", type="primary"):
        bar    = st.progress(0.0)
        status = st.empty()
        try:
            st.session_state.results_df = analyse(df_raw, text_col, sia, bar, status)
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            st.code(traceback.format_exc())
            st.stop()
        bar.progress(1.0)
        status.empty()

# ─────────────────────────────────────────────
# RESULTS  — rendered from session_state so
# a download click (which reruns the page) does
# NOT wipe the results.
# ─────────────────────────────────────────────

if st.session_state.results_df is not None:
    df = st.session_state.results_df

    total = len(df)
    n_pos = (df["Sentiment"] == "Positive").sum()
    n_neu = (df["Sentiment"] == "Neutral").sum()
    n_neg = (df["Sentiment"] == "Negative").sum()
    avg   = df["ensemble_score"].mean()

    # ── Summary metrics ──────────────────────
    st.markdown("<div class='section-label'>Summary</div>", unsafe_allow_html=True)

    def metric(label, value, sub, color):
        return (
            f"<div class='metric-card'>"
            f"<div class='m-label'>{label}</div>"
            f"<div class='m-value' style='color:{color};'>{value}</div>"
            f"<div class='m-pct'>{sub}</div>"
            f"</div>"
        )

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1: st.markdown(metric("Total rows", f"{total:,}",   "analysed",         "#6366f1"), unsafe_allow_html=True)
    with m2: st.markdown(metric("Positive",   f"{n_pos:,}",   f"{n_pos/total:.1%}", "#22c55e"), unsafe_allow_html=True)
    with m3: st.markdown(metric("Neutral",    f"{n_neu:,}",   f"{n_neu/total:.1%}", "#f59e0b"), unsafe_allow_html=True)
    with m4: st.markdown(metric("Negative",   f"{n_neg:,}",   f"{n_neg/total:.1%}", "#ef4444"), unsafe_allow_html=True)
    with m5: st.markdown(metric("Avg Score",  f"{avg:+.3f}",  "ensemble mean",    "#374151"), unsafe_allow_html=True)

    # ── Results table ────────────────────────
    st.markdown("<div class='section-label'>Results preview</div>", unsafe_allow_html=True)
    show = [c for c in df.columns if c != "_clean_"]
    st.dataframe(df[show].head(25), use_container_width=True, hide_index=True)

    # ── Charts ───────────────────────────────
    st.markdown("<div class='section-label'>Charts</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1: st.pyplot(fig_distribution(df), use_container_width=True)
    with c2: st.pyplot(fig_histogram(df),    use_container_width=True)

    w1, w2 = st.columns(2)
    with w1:
        fig = fig_wordcloud(df, "Positive", "Greens")
        if fig: st.pyplot(fig, use_container_width=True)
    with w2:
        fig = fig_wordcloud(df, "Negative", "Reds")
        if fig: st.pyplot(fig, use_container_width=True)

    # ── Download ─────────────────────────────
    st.markdown("<div class='section-label'>Download</div>", unsafe_allow_html=True)

    export = df[[c for c in df.columns if c != "_clean_"]]
    st.download_button(
        label="Download results as CSV",
        data=export.to_csv(index=False).encode("utf-8"),
        file_name="sentiment_results.csv",
        mime="text/csv",
    )
