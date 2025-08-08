# ===========================================================================
# 1. IMPORT LIBRARIES AND INITIAL SETUP
# ===========================================================================
import logging
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import requests
import nltk

try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    from tqdm import tqdm
except ImportError:
    st.error("Required libraries not found. Please ensure your requirements.txt file is correct.")
    st.stop()

# --- Page Configuration ---
st.set_page_config(
    page_title="CryptoSENTRAL Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Add NLTK data path ---
# This tells NLTK to look for data in the 'nltk_data' folder in the repository
NLTK_DATA_PATH = Path.cwd() / "nltk_data"
if NLTK_DATA_PATH.exists():
    nltk.data.path.append(str(NLTK_DATA_PATH))
else:
    st.error("The 'nltk_data' folder was not found in the GitHub repository. Please ensure it has been uploaded correctly.")
    st.stop()


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
# 3. STATION 3: SENTIMENT ANALYSIS (Cached)
# ===========================================================================

@st.cache_data
def run_sentiment_pipeline(unstructured_df):
    """Runs VADER analysis and constructs sentiment indices."""
    
    # Explicitly check for the vader_lexicon file
    vader_path = NLTK_DATA_PATH / "sentiment" / "vader_lexicon.zip"
    if not vader_path.exists():
        st.error(f"VADER lexicon not found at the expected path: {vader_path}. Please ensure the 'nltk_data/sentiment/vader_lexicon.zip' file is in your GitHub repository.")
        st.stop()

    try:
        vader_analyzer = SentimentIntensityAnalyzer()
    except LookupError:
        st.error("NLTK could not find the VADER lexicon, even though the path was added. Please check the file integrity in your GitHub repository.")
        st.stop()

    def get_vader_scores(text):
        if not isinstance(text, str): return {'compound': 0.0}
        return vader_analyzer.polarity_scores(text)

    sentiment_scores = unstructured_df['normalized_text'].apply(get_vader_scores)
    sentiment_df = pd.json_normalize(sentiment_scores)
    news_with_sentiment = pd.concat([unstructured_df.reset_index(drop=True), sentiment_df], axis=1)
    
    df = news_with_sentiment.copy()
    df['date'] = pd.to_datetime(df['date'])
    market_sentiment = df.set_index('date')['compound'].resample('D').mean().rolling(window=7, min_periods=1).mean()
    market_sentiment.name = "market_sentiment_7d"

    asset_sentiment = pd.DataFrame()
    if 'mentioned_symbols' in df.columns:
        df_exploded = df.dropna(subset=['mentioned_symbols']).copy()
        df_exploded['mentioned_symbols'] = df_exploded['mentioned_symbols'].str.split(',')
        df_exploded = df_exploded.explode('mentioned_symbols')
        df_exploded['mentioned_symbols'] = df_exploded['mentioned_symbols'].str.strip().str.upper()
        asset_sentiment_raw = df_exploded.groupby(['date', 'mentioned_symbols'])['compound'].mean().unstack()
        asset_sentiment = asset_sentiment_raw.rolling(window=7, min_periods=1).mean()
    
    return market_sentiment, asset_sentiment

# ===========================================================================
# 4. STATION 4: BACKTESTING (Cached)
# ===========================================================================

@st.cache_data
def run_backtest(structured_df, asset_sentiment):
    """Runs the sentiment-driven backtest."""
    if asset_sentiment.empty:
        # This is not an error, but an expected outcome if symbols are missing
        return None

    weekly_returns = structured_df.pivot_table(index='date', columns='symbol', values='return')
    portfolio_returns = []
    
    for date in weekly_returns.index:
        sentiment_date = date - pd.Timedelta(days=1)
        if sentiment_date not in asset_sentiment.index:
            portfolio_returns.append(0)
            continue
            
        last_sentiment = asset_sentiment.loc[sentiment_date].dropna()
        if len(last_sentiment) < 20:
            portfolio_returns.append(0)
            continue
            
        top_10 = last_sentiment.nlargest(10).index
        bottom_10 = last_sentiment.nsmallest(10).index
        
        valid_longs = [s for s in top_10 if s in weekly_returns.columns and pd.notna(weekly_returns.loc[date, s])]
        valid_shorts = [s for s in bottom_10 if s in weekly_returns.columns and pd.notna(weekly_returns.loc[date, s])]
        
        if not valid_longs or not valid_shorts:
            portfolio_returns.append(0)
            continue

        long_return = weekly_returns.loc[date, valid_longs].mean()
        short_return = weekly_returns.loc[date, valid_shorts].mean()
        week_return = (long_return - short_return) / 2
        portfolio_returns.append(week_return if pd.notna(week_return) else 0)
        
    results_df = pd.DataFrame({'portfolio_return': portfolio_returns}, index=weekly_returns.index)
    results_df['cumulative_return'] = (1 + results_df['portfolio_return']).cumprod()
    return results_df

# ===========================================================================
# 5. GEMINI API INTEGRATION
# ===========================================================================

def get_gemini_summary(sentiment_score):
    """Gets a sentiment summary from the Google Gemini API."""
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except FileNotFoundError:
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
        logging.error(f"Gemini API call failed: {e}")
        return "Could not generate Gemini summary. There may be an issue with the API key or network."

# ===========================================================================
# 6. BUILD THE USER INTERFACE
# ===========================================================================

st.title("CryptoSENTRAL Dashboard")
st.markdown("A Comprehensive Analysis Dashboard for Cryptocurrency Markets")

structured_data, unstructured_data = load_data()
market_sentiment, asset_sentiment = run_sentiment_pipeline(unstructured_data)
backtest_results = run_backtest(structured_data, asset_sentiment)

latest_sentiment_score = market_sentiment.dropna().iloc[-1]
gemini_summary = get_gemini_summary(latest_sentiment_score)

col1, col2 = st.columns([1, 2])
with col1:
    st.subheader("Market Sentiment")
    sentiment_category = "Neutral"
    if latest_sentiment_score > 0.1: sentiment_category = "Positive"
    if latest_sentiment_score < -0.1: sentiment_category = "Negative"
    
    st.metric(
        label="7-Day Average Sentiment Score",
        value=f"{latest_sentiment_score:.3f}",
        delta=sentiment_category,
        delta_color=("off" if sentiment_category == "Neutral" else "normal")
    )
    
    st.subheader("Gemini Analyst Summary")
    st.info(gemini_summary)

with col2:
    st.subheader("Sentiment Strategy Performance")
    if backtest_results is not None:
        st.line_chart(backtest_results['cumulative_return'])
    else:
        st.info("Performance chart is unavailable because the raw news data does not contain asset-specific symbols needed for the backtest.")

st.markdown("---")
st.subheader("Data Explorer")
st.dataframe(structured_data.head())
