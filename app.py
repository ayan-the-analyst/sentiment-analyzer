import streamlit as st
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- 1. Page Configuration ---
# Set wide layout and custom title
st.set_page_config(page_title="Codeayan Sentiment Engine", page_icon="⚡", layout="wide")

# --- 2. Custom CSS Injection ---
# This hides the default Streamlit watermarks and styles the UI
st.markdown("""
<style>
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Style the main title */
    .main-title {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-weight: 700;
        color: #3b82f6; /* Modern Blue */
        margin-bottom: 0px;
    }
    
    /* Style the subheader text */
    .sub-text {
        font-size: 1.1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    
    /* Style the metric cards */
    div[data-testid="metric-container"] {
        background-color: #1e293b;
        border: 1px solid #334155;
        padding: 5% 10% 5% 10%;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# --- 3. NLTK Setup (Cloud-Safe) ---
@st.cache_resource
def setup_nltk():
    nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
    os.makedirs(nltk_data_dir, exist_ok=True)
    nltk.data.path.append(nltk_data_dir)
    nltk.download('vader_lexicon', download_dir=nltk_data_dir, quiet=True)
    return SentimentIntensityAnalyzer()

sia = setup_nltk()

def get_compound_score(text):
    return sia.polarity_scores(str(text))['compound']

def classify_sentiment(score):
    if score >= 0.05: return 'Positive'
    elif score <= -0.05: return 'Negative'
    else: return 'Neutral'

# --- 4. Main App UI ---
st.markdown('<h1 class="main-title">⚡ Sentiment Intelligence Engine</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Upload your raw text or reviews to extract emotional polarity at scale.</p>', unsafe_allow_html=True)

# Wrap the uploader in a clean container
with st.container():
    uploaded_file = st.file_uploader("Upload CSV Dataset (Must contain a 'Text' column)", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    if 'Text' not in df.columns:
        st.error("⚠️ Column mismatch: We couldn't find a column named 'Text' in your CSV. Please rename your text column to 'Text' and try again.")
    else:
        # Show a professional loading spinner
        with st.spinner('Running NLP pipeline...'):
            df['SentimentScore'] = df['Text'].apply(get_compound_score)
            df['Sentiment'] = df['SentimentScore'].apply(classify_sentiment)
            
            # Calculate KPIs
            total_rows = len(df)
            pos_pct = round((len(df[df['Sentiment'] == 'Positive']) / total_rows) * 100, 1)
            neg_pct = round((len(df[df['Sentiment'] == 'Negative']) / total_rows) * 100, 1)
            neu_pct = round((len(df[df['Sentiment'] == 'Neutral']) / total_rows) * 100, 1)

        # --- Dashboard Layout: Metrics ---
        st.markdown("### 📊 Executive Summary")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Records", f"{total_rows:,}")
        col2.metric("Positive Sentiment", f"{pos_pct}%", "Healthy")
        col3.metric("Negative Sentiment", f"{neg_pct}%", "-Action Required")
        col4.metric("Neutral Sentiment", f"{neu_pct}%", "Baseline")
        
        st.markdown("---")
        
        # --- Dashboard Layout: Tabs ---
        tab1, tab2, tab3 = st.tabs(["📈 Visualizations", "🗄️ Raw Data", "⬇️ Export"])
        
        with tab1:
            st.markdown("#### Sentiment Distribution")
            # Using Streamlit's native bar chart for a cleaner, interactive look without Matplotlib borders
            sentiment_counts = df['Sentiment'].value_counts()
            st.bar_chart(sentiment_counts, color="#3b82f6")
            
        with tab2:
            st.markdown("#### Processed Dataset")
            # Interactive dataframe that users can sort and filter
            st.dataframe(df[['Text', 'Sentiment', 'SentimentScore']], use_container_width=True, height=400)
            
        with tab3:
            st.markdown("#### Download Results")
            st.info("Export your dataset with the appended VADER sentiment scores and labels.")
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download CSV Report",
                data=csv,
                file_name='Codeayan_Sentiment_Report.csv',
                mime='text/csv',
                use_container_width=True
            )
