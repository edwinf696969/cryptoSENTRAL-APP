# ===========================================================================
# 1. IMPORT LIBRARIES AND INITIAL SETUP
# ===========================================================================
import json
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import requests
import nltk
import plotly.graph_objects as go

try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
except ImportError:
    st.error("Required libraries not found. Please ensure your requirements.txt file is correct.")
    st.stop()

# --- Page Configuration ---
st.set_page_config(
    page_title="CryptoSENTRAL | Market Sentiment",
    page_icon="📈",
    layout="wide",
)

# --- Add NLTK data path ---
NLTK_DATA_PATH = Path.cwd() / "nltk_data"
if NLTK_DATA_PATH.exists():
    nltk.data.path.append(str(NLTK_DATA_PATH))

# ===========================================================================
# 2. DATA LOADING AND CACHING
# ===========================================================================

@st.cache_data
def load_data():
    """Loads the clean structured and unstructured data from Part A."""
    structured_path = Path("stage_2_structured_features.csv")
    unstructured_path = Path("stage_2_news_processed.csv")

    if not structured_path.exists() or not unstructured_path.exists():
        st.error("Data files not found. Please ensure 'stage_2_structured_features.csv' and 'stage_2_news_processed.csv' are in your GitHub repository.")
        st.stop()
        
    structured_df = pd.read_csv(structured_path, parse_dates=['date'])
    unstructured_df = pd.read_csv(unstructured_path)
    return structured_df, unstructured_df

# ===========================================================================
# 3. SENTIMENT ANALYSIS (Cached)
# ===========================================================================

@st.cache_data
def run_sentiment_pipeline(unstructured_df):
    """Runs VADER analysis and constructs sentiment indices."""
    try:
        vader_analyzer = SentimentIntensityAnalyzer()
    except LookupError:
        st.error("VADER lexicon not found. Please ensure the 'nltk_data/sentiment/vader_lexicon.zip' file is in your GitHub repository.")
        st.stop()

    def get_vader_scores(text):
        if not isinstance(text, str): return {'compound': 0.0}
        return vader_analyzer.polarity_scores(text)

    # Show a progress bar for the long-running sentiment analysis
    progress_bar = st.progress(0, text="Analyzing news sentiment...")
    total_rows = len(unstructured_df)
    
    def apply_with_progress(df_column, func):
        results = []
        for i, row in enumerate(df_column):
            results.append(func(row))
            progress_bar.progress((i + 1) / total_rows)
        return results

    sentiment_scores = apply_with_progress(unstructured_df['normalized_text'], get_vader_scores)
    progress_bar.empty()

    sentiment_df = pd.json_normalize(sentiment_scores)
    news_with_sentiment = pd.concat([unstructured_df.reset_index(drop=True), sentiment_df], axis=1)
    
    df = news_with_sentiment.copy()
    df['date'] = pd.to_datetime(df['date'])
    market_sentiment = df.set_index('date')['compound'].resample('D').mean().rolling(window=7, min_periods=1).mean()
    market_sentiment.name = "market_sentiment_7d"
    
    return market_sentiment

# ===========================================================================
# 4. GEMINI API INTEGRATION
# ===========================================================================

@st.cache_data
def get_gemini_summary(sentiment_score):
    """Gets a sentiment summary from the Google Gemini API."""
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except (FileNotFoundError, KeyError):
        return "Error: Gemini API key not found. Please add it to your Streamlit secrets."

    if not api_key: return "Error: Gemini API key is not set in Streamlit secrets."

    sentiment_category = "Neutral"
    if sentiment_score > 0.1: sentiment_category = "Positive"
    if sentiment_score < -0.1: sentiment_category = "Negative"

    prompt = f"You are an expert financial analyst for a cryptocurrency investment platform called CryptoSENTRAL. The platform's overall market sentiment index, based on recent news, is currently showing a score of {sentiment_score:.3f}, which is '{sentiment_category}'. Write a concise, professional summary (2-3 sentences) for clients explaining what this sentiment level means for the market."
    
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={api_key}"
    
    try:
        response = requests.post(api_url, json={"contents": [{"parts": [{"text": prompt}]}]})
        response.raise_for_status()
        result = response.json()
        return result['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        return f"Could not generate Gemini summary. Error: {e}"

# ===========================================================================
# 5. UI COMPONENTS
# ===========================================================================

def create_fear_greed_gauge(score):
    """Creates a Plotly gauge chart for the sentiment score."""
    gauge_value = (score + 1) * 50
    
    category = "Neutral"
    color = "#FBBF24"
    if gauge_value > 60: category, color = "Greed", "#22C55E"
    if gauge_value > 80: category, color = "Extreme Greed", "#16A34A"
    if gauge_value < 40: category, color = "Fear", "#F97316"
    if gauge_value < 20: category, color = "Extreme Fear", "#EF4444"

    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=gauge_value,
        title={'text': category, 'font': {'size': 24, 'color': color}},
        number={'font': {'size': 48, 'color': "white"}},
        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': color},
               'steps': [
                   {'range': [0, 20], 'color': '#EF4444'}, {'range': [20, 40], 'color': '#F97316'},
                   {'range': [40, 60], 'color': '#FBBF24'}, {'range': [60, 80], 'color': '#22C55E'},
                   {'range': [80, 100], 'color': '#16A34A'}]}))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=250, margin=dict(l=20, r=20, t=50, b=20))
    return fig

# ===========================================================================
# 6. BUILD THE USER INTERFACE
# ===========================================================================

st.markdown("""
    <style>
        .main { background-color: #030712; }
        h1, h2, h3 { color: #F9FAFB; }
        .st-emotion-cache-16txtl3 {
            background-color: #111827; border: 1px solid #374151;
            border-radius: 0.75rem; padding: 1.5rem;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div style="text-align: center;">
        <h1 style="font-size: 3.5rem; font-weight: 800;">
            Crypto<span style="color: #22D3EE;">SENTRAL</span>
        </h1>
        <p style="color: #9CA3AF; font-size: 1.125rem;">A Comprehensive Analysis Dashboard for Cryptocurrency Markets</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

# --- Load and Process Data ---
structured_data, unstructured_data = load_data()
market_sentiment = run_sentiment_pipeline(unstructured_data)

# --- Display KPIs and Gemini Summary ---
latest_sentiment_score = market_sentiment.dropna().iloc[-1]
gemini_summary = get_gemini_summary(latest_sentiment_score)

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Market Sentiment Gauge")
    st.plotly_chart(create_fear_greed_gauge(latest_sentiment_score), use_container_width=True)
    
    st.subheader("Gemini Analyst Summary")
    st.info(gemini_summary)

with col2:
    st.subheader("Market Sentiment Over Time")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=market_sentiment.index, y=market_sentiment,
        mode='lines', name='7-Day Avg. Sentiment',
        line=dict(color='#22D3EE', width=2),
        fill='tozeroy', fillcolor='rgba(34, 211, 238, 0.1)'
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, color='white'),
        yaxis=dict(title='VADER Compound Score', gridcolor='#374151', color='white'),
        legend=dict(font=dict(color='white')), height=400
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.subheader("Raw News Data Explorer")
st.dataframe(unstructured_data.head())
