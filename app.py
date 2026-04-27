"""
╔══════════════════════════════════════════════════════════════╗
║          SentimentIQ — Industry-Grade Sentiment Analyzer     ║
║          Multi-Model Ensemble | CSV & XLSX Support           ║
╚══════════════════════════════════════════════════════════════╝
"""

import io
import re
import time
import warnings
import traceback
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import nltk
import numpy as np
import openpyxl
import pandas as pd
import seaborn as sns
import streamlit as st
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024        # 50 MB hard limit
MAX_ROWS             = 100_000
BATCH_SIZE           = 512                    # rows per progress tick
VADER_WEIGHT         = 0.55
TEXTBLOB_WEIGHT      = 0.45

PALETTE = {
    "Positive": "#22c55e",
    "Neutral":  "#f59e0b",
    "Negative": "#ef4444",
}

CUSTOM_CSS = """
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Space+Grotesk:wght@700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Page background ── */
.stApp {
    background: #0f1117;
    color: #e2e8f0;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #1a1d27;
    border-right: 1px solid #2d3148;
}

/* ── Title block ── */
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(135deg, #818cf8 0%, #c084fc 50%, #fb7185 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 0.25rem;
}
.hero-sub {
    color: #94a3b8;
    font-size: 1rem;
    margin-bottom: 1.5rem;
}

/* ── Metric cards ── */
.metric-card {
    background: #1e2130;
    border: 1px solid #2d3148;
    border-radius: 12px;
    padding: 1.1rem 1.4rem;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #6366f1; }
.metric-card .label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #64748b;
    margin-bottom: 0.35rem;
}
.metric-card .value {
    font-size: 1.9rem;
    font-weight: 700;
    font-family: 'Space Grotesk', sans-serif;
}
.metric-card .sub { font-size: 0.78rem; color: #64748b; margin-top: 0.2rem; }

/* ── Section headers ── */
.section-header {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: #e2e8f0;
    border-left: 3px solid #6366f1;
    padding-left: 0.7rem;
    margin: 1.5rem 0 0.75rem;
}

/* ── File format box ── */
.format-box {
    background: #1e2130;
    border: 1px dashed #3b4268;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    font-size: 0.85rem;
    color: #94a3b8;
    margin-bottom: 1rem;
}
.format-box code {
    background: #2d3148;
    border-radius: 4px;
    padding: 0.1rem 0.4rem;
    color: #a5b4fc;
    font-size: 0.82rem;
}

/* ── Status pills ── */
.pill-positive { background:#14532d; color:#86efac; border-radius:999px; padding:2px 10px; font-size:0.78rem; font-weight:600; }
.pill-neutral  { background:#451a03; color:#fde68a; border-radius:999px; padding:2px 10px; font-size:0.78rem; font-weight:600; }
.pill-negative { background:#450a0a; color:#fca5a5; border-radius:999px; padding:2px 10px; font-size:0.78rem; font-weight:600; }

/* ── Dataframe ── */
.stDataFrame { border-radius: 8px; overflow: hidden; }

/* ── Buttons ── */
.stDownloadButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.5rem 1.5rem;
    transition: opacity 0.2s;
}
.stDownloadButton > button:hover { opacity: 0.85; }

/* ── Expander ── */
details { background: #1e2130; border-radius: 10px; border: 1px solid #2d3148 !important; }

/* ── Alert / info boxes ── */
.stAlert { border-radius: 8px; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: #6366f1 !important; }
</style>
"""

# ──────────────────────────────────────────────────────────────
# CACHED RESOURCE LOADERS
# ──────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _bootstrap_nlp() -> SentimentIntensityAnalyzer:
    """Download all required NLTK assets once and return a shared VADER instance."""
    for pkg in ("vader_lexicon", "stopwords", "punkt", "punkt_tab"):
        nltk.download(pkg, quiet=True)
    return SentimentIntensityAnalyzer()


@st.cache_resource(show_spinner=False)
def _load_transformer():
    """
    Lazily load a DistilBERT SST-2 pipeline (runs only when user enables it).
    Returns (pipeline, True) on success, (None, False) on failure.
    """
    try:
        from transformers import pipeline as hf_pipeline
        pipe = hf_pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            truncation=True,
            max_length=512,
            device=-1,          # CPU inference
        )
        return pipe, True
    except Exception:
        return None, False


# ──────────────────────────────────────────────────────────────
# TEXT PREPROCESSING
# ──────────────────────────────────────────────────────────────

_URL_RE    = re.compile(r"https?://\S+|www\.\S+")
_EMOJI_RE  = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+", flags=re.UNICODE
)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MULTI_SPACE = re.compile(r"\s{2,}")


def sanitize_text(raw: object) -> str:
    """Normalize raw input to a clean unicode string safe for NLP."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return ""
    text = str(raw).strip()
    if not text or text.lower() in {"nan", "none", "null", "n/a", "na"}:
        return ""
    # Remove HTML tags
    text = _HTML_TAG_RE.sub(" ", text)
    # Remove URLs
    text = _URL_RE.sub(" ", text)
    # Collapse whitespace (keep emojis – VADER handles them)
    text = _MULTI_SPACE.sub(" ", text).strip()
    return text


# ──────────────────────────────────────────────────────────────
# SENTIMENT SCORING
# ──────────────────────────────────────────────────────────────

def _score_vader(sia: SentimentIntensityAnalyzer, text: str) -> dict:
    scores = sia.polarity_scores(text)
    return {
        "vader_compound": scores["compound"],
        "vader_pos":      scores["pos"],
        "vader_neu":      scores["neu"],
        "vader_neg":      scores["neg"],
    }


def _score_textblob(text: str) -> dict:
    blob = TextBlob(text)
    polarity    = blob.sentiment.polarity       # –1 → +1
    subjectivity = blob.sentiment.subjectivity  # 0 → 1
    # Normalise polarity to –1..+1 (already is, just be explicit)
    return {
        "tb_polarity":    round(polarity, 4),
        "tb_subjectivity": round(subjectivity, 4),
    }


def _score_transformer(pipe, text: str) -> dict:
    """Run DistilBERT SST-2; map to –1..+1 compound equivalent."""
    try:
        result = pipe(text[:512])[0]
        label  = result["label"]   # "POSITIVE" or "NEGATIVE"
        conf   = result["score"]
        # Convert to –1..+1 scale
        compound = conf if label == "POSITIVE" else -conf
        return {"transformer_compound": round(compound, 4), "transformer_label": label}
    except Exception:
        return {"transformer_compound": 0.0, "transformer_label": "ERROR"}


def _ensemble_score(
    vader_compound: float,
    tb_polarity: float,
    transformer_compound: Optional[float] = None,
) -> float:
    """
    Weighted ensemble of available models.
    When transformer is available: VADER 40% + TextBlob 25% + Transformer 35%
    Otherwise:                     VADER 55% + TextBlob 45%
    """
    if transformer_compound is not None:
        return round(
            0.40 * vader_compound +
            0.25 * tb_polarity +
            0.35 * transformer_compound,
            4,
        )
    return round(VADER_WEIGHT * vader_compound + TEXTBLOB_WEIGHT * tb_polarity, 4)


def classify_sentiment(score: float) -> str:
    if score >=  0.05: return "Positive"
    if score <= -0.05: return "Negative"
    return "Neutral"


def score_confidence(score: float) -> str:
    """Human-readable confidence bucket for the ensemble score."""
    abs_s = abs(score)
    if abs_s >= 0.7: return "High"
    if abs_s >= 0.35: return "Medium"
    return "Low"


# ──────────────────────────────────────────────────────────────
# BATCH ANALYSIS
# ──────────────────────────────────────────────────────────────

def analyse_dataframe(
    df: pd.DataFrame,
    text_col: str,
    sia: SentimentIntensityAnalyzer,
    use_transformer: bool,
    progress_bar,
    status_text,
) -> pd.DataFrame:
    """
    Scores every row; returns the original df with appended sentiment columns.
    Processes in batches so the progress bar actually moves.
    """
    transformer_pipe, transformer_ok = (
        _load_transformer() if use_transformer else (None, False)
    )
    if use_transformer and not transformer_ok:
        st.warning(
            "⚠️ Transformer model could not be loaded "
            "(missing `transformers`/`torch`). "
            "Falling back to VADER + TextBlob ensemble.",
            icon="⚠️",
        )

    n     = len(df)
    texts = df[text_col].tolist()

    results = []
    t0 = time.time()

    for i, raw in enumerate(texts):
        text = sanitize_text(raw)

        if not text:
            results.append(
                {
                    "CleanText":            "",
                    "vader_compound":       0.0,
                    "vader_pos":            0.0,
                    "vader_neu":            0.0,
                    "vader_neg":            0.0,
                    "tb_polarity":          0.0,
                    "tb_subjectivity":      0.0,
                    "EnsembleScore":        0.0,
                    "Sentiment":            "Neutral",
                    "Confidence":           "Low",
                }
            )
            continue

        v  = _score_vader(sia, text)
        tb = _score_textblob(text)

        t_compound = None
        t_label    = None
        if transformer_ok and transformer_pipe:
            t_data     = _score_transformer(transformer_pipe, text)
            t_compound = t_data["transformer_compound"]
            t_label    = t_data["transformer_label"]

        ensemble = _ensemble_score(v["vader_compound"], tb["tb_polarity"], t_compound)
        sentiment = classify_sentiment(ensemble)
        confidence = score_confidence(ensemble)

        row = {
            "CleanText":       text,
            "vader_compound":  v["vader_compound"],
            "vader_pos":       v["vader_pos"],
            "vader_neu":       v["vader_neu"],
            "vader_neg":       v["vader_neg"],
            "tb_polarity":     tb["tb_polarity"],
            "tb_subjectivity": tb["tb_subjectivity"],
            "EnsembleScore":   ensemble,
            "Sentiment":       sentiment,
            "Confidence":      confidence,
        }
        if transformer_ok:
            row["transformer_compound"] = t_compound
            row["transformer_label"]    = t_label

        results.append(row)

        # Update progress every BATCH_SIZE rows
        if (i + 1) % BATCH_SIZE == 0 or (i + 1) == n:
            pct      = (i + 1) / n
            elapsed  = time.time() - t0
            eta      = (elapsed / (i + 1)) * (n - i - 1)
            progress_bar.progress(pct)
            status_text.text(
                f"Processed {i + 1:,} / {n:,} rows  •  ETA {eta:.0f}s"
            )

    results_df = pd.DataFrame(results)
    out = pd.concat([df.reset_index(drop=True), results_df], axis=1)
    return out


# ──────────────────────────────────────────────────────────────
# VISUALIZATION HELPERS
# ──────────────────────────────────────────────────────────────

_PLOT_BG  = "#0f1117"
_AX_BG    = "#1e2130"
_SPINE    = "#2d3148"
_TEXT_CLR = "#e2e8f0"
_GRID_CLR = "#2d3148"


def _apply_dark_theme(fig: plt.Figure, axes):
    """Apply consistent dark theme to every axes object."""
    fig.patch.set_facecolor(_PLOT_BG)
    for ax in (axes if hasattr(axes, "__iter__") else [axes]):
        ax.set_facecolor(_AX_BG)
        ax.tick_params(colors=_TEXT_CLR, labelsize=9)
        ax.xaxis.label.set_color(_TEXT_CLR)
        ax.yaxis.label.set_color(_TEXT_CLR)
        ax.title.set_color(_TEXT_CLR)
        for sp in ax.spines.values():
            sp.set_edgecolor(_SPINE)
        ax.grid(color=_GRID_CLR, linewidth=0.5, linestyle="--", alpha=0.5)


def plot_distribution(df: pd.DataFrame) -> plt.Figure:
    counts = df["Sentiment"].value_counts().reindex(["Positive", "Neutral", "Negative"], fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(counts.index, counts.values,
                  color=[PALETTE[k] for k in counts.index],
                  width=0.5, zorder=3, edgecolor="none")
    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + counts.max() * 0.02,
            f"{val:,}",
            ha="center", va="bottom",
            color=_TEXT_CLR, fontsize=11, fontweight="bold",
        )
    ax.set_ylabel("Count", labelpad=8)
    ax.set_title("Sentiment Distribution", fontsize=13, fontweight="bold", pad=12)
    _apply_dark_theme(fig, ax)
    fig.tight_layout()
    return fig


def plot_score_histogram(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["EnsembleScore"], bins=50, color="#6366f1", edgecolor="none",
            alpha=0.85, zorder=3)
    ax.axvline( 0.05, color="#22c55e", lw=1.5, linestyle="--", label="Positive threshold")
    ax.axvline(-0.05, color="#ef4444", lw=1.5, linestyle="--", label="Negative threshold")
    ax.set_xlabel("Ensemble Score", labelpad=8)
    ax.set_ylabel("Frequency", labelpad=8)
    ax.set_title("Score Distribution", fontsize=13, fontweight="bold", pad=12)
    legend = ax.legend(framealpha=0, labelcolor=_TEXT_CLR, fontsize=8)
    _apply_dark_theme(fig, ax)
    fig.tight_layout()
    return fig


def plot_confidence_breakdown(df: pd.DataFrame) -> plt.Figure:
    ct = (
        df.groupby(["Sentiment", "Confidence"])
        .size()
        .unstack(fill_value=0)
        .reindex(["Positive", "Neutral", "Negative"])
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#4ade80", "#fde68a", "#f87171"]
    bottom = np.zeros(len(ct))
    for j, conf_level in enumerate(["High", "Medium", "Low"]):
        if conf_level in ct.columns:
            vals = ct[conf_level].values
            ax.bar(ct.index, vals, bottom=bottom,
                   label=conf_level, color=colors[j], width=0.5,
                   edgecolor="none", zorder=3)
            bottom += vals
    ax.set_ylabel("Count", labelpad=8)
    ax.set_title("Sentiment × Confidence", fontsize=13, fontweight="bold", pad=12)
    legend = ax.legend(framealpha=0, labelcolor=_TEXT_CLR, fontsize=9,
                       title="Confidence", title_fontsize=9)
    legend.get_title().set_color(_TEXT_CLR)
    _apply_dark_theme(fig, ax)
    fig.tight_layout()
    return fig


def plot_subjectivity_scatter(df: pd.DataFrame) -> plt.Figure:
    sample = df.sample(min(1000, len(df)), random_state=42)
    fig, ax = plt.subplots(figsize=(7, 4))
    for sent, color in PALETTE.items():
        mask = sample["Sentiment"] == sent
        ax.scatter(
            sample.loc[mask, "EnsembleScore"],
            sample.loc[mask, "tb_subjectivity"],
            c=color, s=12, alpha=0.55, label=sent, edgecolors="none",
        )
    ax.set_xlabel("Ensemble Score", labelpad=8)
    ax.set_ylabel("Subjectivity", labelpad=8)
    ax.set_title("Score vs Subjectivity (sample ≤ 1 000)", fontsize=12, fontweight="bold", pad=12)
    handles = [mpatches.Patch(color=c, label=l) for l, c in PALETTE.items()]
    legend  = ax.legend(handles=handles, framealpha=0, labelcolor=_TEXT_CLR, fontsize=9)
    _apply_dark_theme(fig, ax)
    fig.tight_layout()
    return fig


def _make_wordcloud(text: str, colormap: str, title: str) -> Optional[plt.Figure]:
    if not text.strip():
        return None
    wc = WordCloud(
        width=700, height=320,
        background_color=None, mode="RGBA",
        colormap=colormap,
        max_words=120,
        collocations=False,
        stopwords=STOPWORDS,
        prefer_horizontal=0.85,
    ).generate(text)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontsize=12, fontweight="bold", color=_TEXT_CLR, pad=8)
    fig.patch.set_facecolor(_PLOT_BG)
    fig.tight_layout(pad=0)
    return fig


def plot_timeline(df: pd.DataFrame, date_col: str) -> Optional[plt.Figure]:
    """Render a daily rolling mean if the user chose a date column."""
    try:
        tmp = df[[date_col, "EnsembleScore"]].copy()
        tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
        tmp = tmp.dropna(subset=[date_col]).sort_values(date_col)
        if len(tmp) < 2:
            return None
        daily = tmp.set_index(date_col).resample("D")["EnsembleScore"].mean().dropna()
        if len(daily) < 2:
            return None
        rolled = daily.rolling(7, min_periods=1).mean()
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.fill_between(daily.index, daily.values, alpha=0.18, color="#6366f1")
        ax.plot(daily.index, daily.values,  color="#818cf8", lw=1, alpha=0.5, label="Daily avg")
        ax.plot(rolled.index, rolled.values, color="#c084fc", lw=2, label="7-day rolling avg")
        ax.axhline(0, color=_SPINE, lw=1, linestyle="--")
        ax.set_ylabel("Avg Score", labelpad=8)
        ax.set_title("Sentiment Over Time", fontsize=13, fontweight="bold", pad=12)
        legend = ax.legend(framealpha=0, labelcolor=_TEXT_CLR, fontsize=9)
        _apply_dark_theme(fig, ax)
        fig.autofmt_xdate()
        fig.tight_layout()
        return fig
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────
# FILE I/O HELPERS
# ──────────────────────────────────────────────────────────────

def read_uploaded_file(uploaded_file) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Reads CSV or XLSX/XLS.
    Returns (dataframe, error_message).  error_message is "" on success.
    """
    fname  = uploaded_file.name.lower()
    nbytes = uploaded_file.size

    if nbytes > MAX_FILE_SIZE_BYTES:
        mb = nbytes / (1024 ** 2)
        return None, f"File is {mb:.1f} MB — maximum allowed is 50 MB."

    try:
        raw = uploaded_file.read()
        buf = io.BytesIO(raw)

        if fname.endswith(".csv"):
            # Try multiple encodings gracefully
            for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
                try:
                    buf.seek(0)
                    df = pd.read_csv(buf, encoding=enc, low_memory=False)
                    break
                except (UnicodeDecodeError, pd.errors.ParserError):
                    continue
            else:
                return None, "CSV could not be decoded. Please ensure UTF-8 encoding."

        elif fname.endswith((".xlsx", ".xls")):
            buf.seek(0)
            try:
                df = pd.read_excel(buf, engine="openpyxl" if fname.endswith(".xlsx") else "xlrd")
            except Exception as e:
                return None, f"Excel read error: {e}"
        else:
            return None, "Unsupported format. Please upload a .csv or .xlsx file."

    except Exception as e:
        return None, f"Unexpected read error: {e}"

    # Enforce row limit
    if len(df) > MAX_ROWS:
        return None, (
            f"Dataset has {len(df):,} rows, which exceeds the {MAX_ROWS:,}-row limit. "
            "Please trim and re-upload."
        )

    # Drop fully-empty rows
    df = df.dropna(how="all").reset_index(drop=True)

    if df.empty:
        return None, "The uploaded file appears to be empty."

    return df, ""


def df_to_bytes(df: pd.DataFrame, fmt: str) -> bytes:
    """Serialise a dataframe to CSV or XLSX bytes."""
    buf = io.BytesIO()
    if fmt == "csv":
        buf.write(df.to_csv(index=False).encode("utf-8"))
    else:
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="SentimentResults")
    return buf.getvalue()


# ──────────────────────────────────────────────────────────────
# PAGE CONFIG  &  MAIN
# ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SentimentIQ",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='font-family:Space Grotesk,sans-serif;font-size:1.3rem;"
        "font-weight:700;color:#a5b4fc;margin-bottom:1rem;'>⚙️ Settings</div>",
        unsafe_allow_html=True,
    )

    st.markdown("**Model selection**")
    use_transformer = st.toggle(
        "Enable DistilBERT (deep-learning)",
        value=False,
        help=(
            "Adds a fine-tuned DistilBERT SST-2 transformer to the ensemble. "
            "Significantly more accurate but requires `transformers` + `torch` "
            "and is slower on CPU. Leave off for large files."
        ),
    )

    st.markdown("---")
    st.markdown("**Export format**")
    export_fmt = st.radio("Download as", ["CSV", "XLSX"], horizontal=True)

    st.markdown("---")
    st.markdown("**Optional: date column**")
    date_col_name = st.text_input(
        "Date column name for timeline chart",
        placeholder="e.g. date, created_at",
        help="If your file contains a date/timestamp column, enter its name here for the timeline chart.",
    ).strip()

    st.markdown("---")
    st.markdown(
        "<small style='color:#64748b;'>SentimentIQ uses a weighted ensemble of "
        "<b>VADER</b> (rule-based, optimised for social text) and "
        "<b>TextBlob</b> (machine-learning polarity). "
        "When enabled, <b>DistilBERT</b> (transformer) is added for deep contextual understanding. "
        "Scores are normalised to [−1, +1].</small>",
        unsafe_allow_html=True,
    )

# ── Hero ─────────────────────────────────────────────────────
st.markdown("<div class='hero-title'>SentimentIQ</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='hero-sub'>"
    "Multi-model ensemble sentiment analysis · VADER + TextBlob + DistilBERT"
    "</div>",
    unsafe_allow_html=True,
)

# ── File Format Spec ─────────────────────────────────────────
with st.expander("📋 Required file format & column guide", expanded=False):
    st.markdown(
        """
**Accepted formats:** `.csv` · `.xlsx` · `.xls` &nbsp;|&nbsp; **Max size:** 50 MB &nbsp;|&nbsp; **Max rows:** 100 000

Your file **must** contain at least one text column. During upload you will choose which column to analyse.

| Column | Required | Example values |
|---|---|---|
| Any text column | ✅ Yes (you name it) | `"Great product, very happy!"` |
| Date / timestamp | ❌ Optional | `2024-03-15`, `2024-03-15 09:42:00` |
| Any other columns | ❌ Optional | IDs, ratings, categories — all preserved in output |

**Quick CSV template:**
```
text,date,product_id
"Absolutely loved this product!",2024-01-01,SKU-001
"Terrible experience, never again.",2024-01-02,SKU-002
"It was okay, nothing special.",2024-01-03,SKU-003
```
        """,
        unsafe_allow_html=True,
    )

# ── File Upload ───────────────────────────────────────────────
st.markdown("<div class='section-header'>Upload Dataset</div>", unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Drag & drop or click to browse",
    type=["csv", "xlsx", "xls"],
    help="Accepted: .csv, .xlsx, .xls  |  Max 50 MB  |  Max 100 000 rows",
)

# ── Main Logic ────────────────────────────────────────────────
if uploaded_file is not None:

    # ── 1. Read & validate ─────────────────────────────────
    with st.spinner("Reading file…"):
        df_raw, err = read_uploaded_file(uploaded_file)

    if err:
        st.error(f"❌ {err}")
        st.stop()

    st.success(
        f"✅ Loaded **{len(df_raw):,} rows** and **{len(df_raw.columns)} columns** "
        f"from `{uploaded_file.name}`",
        icon="✅",
    )

    # ── 2. Column selection ────────────────────────────────
    st.markdown("<div class='section-header'>Column Configuration</div>", unsafe_allow_html=True)
    all_cols = df_raw.columns.tolist()

    # Try to auto-suggest: first column whose name contains text-like keywords
    _text_hints = {"text", "review", "comment", "description", "body", "content",
                   "message", "feedback", "tweet", "post", "title", "summary"}
    default_col = next(
        (c for c in all_cols if any(h in c.lower() for h in _text_hints)),
        all_cols[0],
    )

    col_a, col_b = st.columns([3, 2])
    with col_a:
        text_col = st.selectbox(
            "Text column to analyse",
            options=all_cols,
            index=all_cols.index(default_col),
            help="This column will be passed to the sentiment models.",
        )
    with col_b:
        st.dataframe(
            df_raw[[text_col]].head(3).rename(columns={text_col: "Preview"}),
            use_container_width=True,
            hide_index=True,
        )

    # ── 3. Analyse ─────────────────────────────────────────
    st.markdown("<div class='section-header'>Analysis</div>", unsafe_allow_html=True)

    if st.button("🚀 Run Sentiment Analysis", type="primary", use_container_width=True):

        sia = _bootstrap_nlp()
        prog_bar   = st.progress(0.0)
        status_txt = st.empty()

        try:
            t_start = time.time()
            df_out  = analyse_dataframe(
                df_raw, text_col, sia,
                use_transformer, prog_bar, status_txt,
            )
            elapsed = time.time() - t_start
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            st.code(traceback.format_exc(), language="text")
            st.stop()

        prog_bar.progress(1.0)
        status_txt.empty()

        rows_s = len(df_out)
        st.success(
            f"✅ Analysed **{rows_s:,} texts** in **{elapsed:.1f}s** "
            f"({rows_s / elapsed:,.0f} rows/s)",
            icon="✅",
        )

        # ── 4. Summary metrics ────────────────────────────
        st.markdown("<div class='section-header'>Overview</div>", unsafe_allow_html=True)

        total     = len(df_out)
        n_pos     = (df_out["Sentiment"] == "Positive").sum()
        n_neg     = (df_out["Sentiment"] == "Negative").sum()
        n_neu     = (df_out["Sentiment"] == "Neutral").sum()
        avg_score = df_out["EnsembleScore"].mean()
        avg_subj  = df_out["tb_subjectivity"].mean()
        n_empty   = (df_out["CleanText"] == "").sum()

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        metrics = [
            (m1, "POSITIVE",   f"{n_pos:,}",   f"{n_pos/total:.1%}", "#22c55e"),
            (m2, "NEUTRAL",    f"{n_neu:,}",   f"{n_neu/total:.1%}", "#f59e0b"),
            (m3, "NEGATIVE",   f"{n_neg:,}",   f"{n_neg/total:.1%}", "#ef4444"),
            (m4, "AVG SCORE",  f"{avg_score:+.3f}", "ensemble mean",  "#818cf8"),
            (m5, "AVG SUBJECTIVITY", f"{avg_subj:.3f}", "0 = objective",  "#a78bfa"),
            (m6, "SKIPPED",    f"{n_empty:,}", "empty / null",       "#64748b"),
        ]
        for col, label, value, sub, color in metrics:
            with col:
                st.markdown(
                    f"<div class='metric-card'>"
                    f"<div class='label'>{label}</div>"
                    f"<div class='value' style='color:{color};'>{value}</div>"
                    f"<div class='sub'>{sub}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        # ── 5. Data preview ───────────────────────────────
        st.markdown("<div class='section-header'>Data Preview (top 20)</div>", unsafe_allow_html=True)

        show_cols = [text_col, "CleanText", "Sentiment", "Confidence",
                     "EnsembleScore", "vader_compound", "tb_polarity", "tb_subjectivity"]
        if "transformer_compound" in df_out.columns:
            show_cols.append("transformer_compound")
        show_cols = [c for c in show_cols if c in df_out.columns]

        st.dataframe(
            df_out[show_cols].head(20),
            use_container_width=True,
            hide_index=True,
        )

        # ── 6. Visualizations ─────────────────────────────
        st.markdown("<div class='section-header'>Visualizations</div>", unsafe_allow_html=True)

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Distribution",
            "📈 Score Histogram",
            "🎯 Confidence",
            "💬 Word Clouds",
            "🕐 Timeline",
        ])

        with tab1:
            c1, c2 = st.columns(2)
            with c1:
                st.pyplot(plot_distribution(df_out), use_container_width=True)
            with c2:
                st.pyplot(plot_subjectivity_scatter(df_out), use_container_width=True)

        with tab2:
            st.pyplot(plot_score_histogram(df_out), use_container_width=True)

        with tab3:
            st.pyplot(plot_confidence_breakdown(df_out), use_container_width=True)

        with tab4:
            wc_c1, wc_c2 = st.columns(2)
            pos_text = " ".join(df_out.loc[df_out["Sentiment"] == "Positive", "CleanText"].dropna().tolist())
            neg_text = " ".join(df_out.loc[df_out["Sentiment"] == "Negative", "CleanText"].dropna().tolist())
            pos_fig  = _make_wordcloud(pos_text, "Greens",  "🟢 Positive Keywords")
            neg_fig  = _make_wordcloud(neg_text, "Reds",    "🔴 Negative Keywords")
            with wc_c1:
                if pos_fig:
                    st.pyplot(pos_fig, use_container_width=True)
                else:
                    st.info("No positive texts to visualise.")
            with wc_c2:
                if neg_fig:
                    st.pyplot(neg_fig, use_container_width=True)
                else:
                    st.info("No negative texts to visualise.")

        with tab5:
            resolved_date_col = date_col_name if date_col_name in df_out.columns else None
            if resolved_date_col:
                tl_fig = plot_timeline(df_out, resolved_date_col)
                if tl_fig:
                    st.pyplot(tl_fig, use_container_width=True)
                else:
                    st.info("Could not parse dates in the specified column.")
            else:
                st.info(
                    "To see the timeline chart, enter a **date column name** "
                    "in the sidebar and re-run the analysis."
                )

        # ── 7. Download ───────────────────────────────────
        st.markdown("<div class='section-header'>Export Results</div>", unsafe_allow_html=True)
        fmt_lower = export_fmt.lower()
        mime_map  = {"csv": "text/csv", "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"}

        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                label=f"⬇️ Download Full Results ({export_fmt})",
                data=df_to_bytes(df_out, fmt_lower),
                file_name=f"sentimentiq_results.{fmt_lower}",
                mime=mime_map[fmt_lower],
                use_container_width=True,
            )
        with dl2:
            # Summary stats table
            summary = pd.DataFrame({
                "Metric":  ["Total Rows", "Positive", "Neutral", "Negative",
                            "Average Score", "Average Subjectivity", "Skipped (empty)"],
                "Value":   [total, n_pos, n_neu, n_neg,
                            f"{avg_score:+.4f}", f"{avg_subj:.4f}", n_empty],
                "Percent": ["-",
                            f"{n_pos/total:.2%}", f"{n_neu/total:.2%}", f"{n_neg/total:.2%}",
                            "-", "-", f"{n_empty/total:.2%}"],
            })
            st.download_button(
                label="⬇️ Download Summary Stats (CSV)",
                data=summary.to_csv(index=False).encode("utf-8"),
                file_name="sentimentiq_summary.csv",
                mime="text/csv",
                use_container_width=True,
            )

else:
    # ── Landing placeholder ───────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns([1, 1])
    with col_l:
        st.info(
            "👆 **Upload a file above to begin.**\n\n"
            "SentimentIQ will automatically score every text in your dataset "
            "and provide an interactive dashboard with distribution charts, word clouds, "
            "confidence breakdowns, and a downloadable results file.",
            icon="ℹ️",
        )
    with col_r:
        st.markdown(
            """
            <div style='background:#1e2130;border:1px solid #2d3148;border-radius:12px;padding:1.2rem;font-size:0.85rem;color:#94a3b8;'>
            <b style='color:#e2e8f0;'>Models in the ensemble</b><br><br>
            🔷 <b style='color:#a5b4fc;'>VADER</b> — Rule-based, optimised for social media & short text<br><br>
            🔶 <b style='color:#fb923c;'>TextBlob</b> — Naïve Bayes trained on movie reviews<br><br>
            🤖 <b style='color:#6ee7b7;'>DistilBERT</b> — 66 M-parameter transformer fine-tuned on SST-2 (optional, toggle in sidebar)
            </div>
            """,
            unsafe_allow_html=True,
        )
