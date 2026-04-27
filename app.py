import streamlit as st
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns

# Download VADER dictionary automatically in the background
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

def get_compound_score(text):
    return sia.polarity_scores(str(text))['compound']

def classify_sentiment(score):
    if score >= 0.05: return 'Positive'
    elif score <= -0.05: return 'Negative'
    else: return 'Neutral'

# --- Web App UI Configuration ---
st.set_page_config(page_title="Sentiment Analyzer", layout="centered")

st.title("📊 Text Sentiment Analyzer")
st.write("Upload a dataset to instantly analyze the emotional tone of text or reviews.")

# File Uploader Widget
uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=['csv'])

if uploaded_file is not None:
    # Read the user's uploaded file
    df = pd.read_csv(uploaded_file)
    
    # Check if a 'Text' column exists
    if 'Text' not in df.columns:
        st.error("Error: Your CSV must contain a column named exactly 'Text'.")
    else:
        st.success(f"File uploaded successfully! Processing {len(df)} rows...")
        
        with st.spinner('Analyzing sentiment...'):
            # Apply VADER scoring
            df['SentimentScore'] = df['Text'].apply(get_compound_score)
            df['Sentiment'] = df['SentimentScore'].apply(classify_sentiment)
            
        # Display the results
        st.subheader("Data Preview")
        st.dataframe(df[['Text', 'Sentiment', 'SentimentScore']].head(10))
        
        # Create Visualizations
        st.subheader("Overall Sentiment Distribution")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(x='Sentiment', data=df, palette=['#ff9999', '#66b3ff', '#99ff99'], ax=ax)
        ax.set_ylabel('Number of Reviews')
        st.pyplot(fig)
        
        # Provide a download button for the user to get their analyzed data back
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Analyzed Dataset",
            data=csv,
            file_name='sentiment_results.csv',
            mime='text/csv',
        )
