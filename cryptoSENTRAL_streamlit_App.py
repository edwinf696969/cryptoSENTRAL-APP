# ===========================================================================
# 1. IMPORT LIBRARIES AND INITIAL SETUP
# ===========================================================================
import json
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import requests
import re

# --- Page Configuration ---
st.set_page_config(
    page_title="CryptoSENTRAL Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===========================================================================
# 2. EMBEDDED VADER LEXICON AND SENTIMENT LOGIC
# This section makes the script self-contained and removes the need for NLTK downloads.
# ===========================================================================

def get_vader_lexicon():
    """Returns the VADER sentiment lexicon as a dictionary."""
    # This is a subset of the full VADER lexicon for brevity.
    # In a real application, the full file would be loaded.
    return {
        'positive': 2.0, 'trust': 1.5, 'good': 1.9, 'great': 3.1, 'excellent': 3.4,
        'amazing': 4.0, 'fantastic': 4.0, 'love': 3.2, 'like': 2.0, 'happy': 2.7,
        'pleased': 2.4, 'success': 2.8, 'win': 2.8, 'gain': 2.4, 'profit': 2.2,
        'up': 1.0, 'increase': 1.5, 'strong': 1.8, 'bullish': 2.9, 'boom': 2.0,
        'negative': -2.0, 'sad': -2.1, 'bad': -2.5, 'terrible': -3.1, 'horrible': -3.1,
        'hate': -2.7, 'loss': -2.3, 'fail': -2.4, 'down': -1.0, 'decrease': -1.5,
        'weak': -1.7, 'bearish': -2.9, 'crash': -3.0, 'risk': -1.5, 'scam': -2.5,
        'fraud': -2.5, 'hack': -2.0, 'stolen': -2.2, 'illegal': -2.6, 'ban': -2.6,
        'fear': -1.7, 'uncertainty': -1.4, 'doubt': -1.1, 'fud': -2.0, 'hodl': 0.5,
        'moon': 2.5, 'diamond hands': 2.0, 'paper hands': -1.5, 'shill': -1.0,
        'not': -1, 'no': -1, 'never': -1,
    }

def get_simple_vader_score(text: str, lexicon: dict) -> float:
    """
    A simplified VADER-style sentiment scoring function using an embedded lexicon.
    """
    if not isinstance(text, str):
        return 0.0
    
    words = text.lower().split()
    score = 0.0
    
    for word in words:
        score += lexicon.get(word, 0.0)
        
    # Simple normalization
    if score != 0:
        score = score / np.sqrt((score * score) + 15)
        
    return score

# ===========================================================================
# 3. DATA LOADING AND CACHING
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
# 4. STATION 3: SENTIMENT ANALYSIS (Cached)
# ===========================================================================

@st.cache_data
def run_sentiment_pipeline(unstructured_df):
    """Runs VADER analysis and constructs sentiment indices."""
    st.info("Running sentiment analysis on news data...")
    lexicon = get_vader_lexicon()
    
    # Use the simplified, self-contained VADER scoring function
    unstructured_df['compound'] = unstructured_df['normalized_text'].apply(lambda x: get_simple_vader_score(x, lexicon))
    
    df = unstructured_df.copy()
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
# 5. STATION 4: BACKTESTING (Cached)
# ===========================================================================

@st.cache_data
def run_backtest(structured_df, asset_sentiment):
    """Runs the sentiment-driven backtest."""
    if asset_sentiment.empty:
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
# 6. GEMINI API INTEGRATION
# ===========================================================================

def get_gemini_summary(sentiment_score):
    """Gets a sentiment summary from the Google Gemini API."""
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except (FileNotFoundError, KeyError):
        return "Error: Gemini API key not found. Please add it to your Streamlit secrets."

    if not api_key: return "Error: Gemini API key is not set in Streamlit secrets."

    sentiment_category = "Neutral"
    if sentiment_score > 0.05: sentiment_category = "Positive"
    if sentiment_score < -0.05: sentiment_category = "Negative"

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
# 7. BUILD THE USER INTERFACE
# ===========================================================================

st.title("CryptoSENTRAL Dashboard")
st.markdown("A Comprehensive Analysis Dashboard for Cryptocurrency Markets")

# --- Load and Process Data ---
structured_data, unstructured_data = load_data()
market_sentiment, asset_sentiment = run_sentiment_pipeline(unstructured_data)
backtest_results = run_backtest(structured_data, asset_sentiment)

# --- Display KPIs and Gemini Summary ---
latest_sentiment_score = market_sentiment.dropna().iloc[-1]
gemini_summary = get_gemini_summary(latest_sentiment_score)

col1, col2 = st.columns([1, 2])
with col1:
    st.subheader("Market Sentiment")
    sentiment_category = "Neutral"
    if latest_sentiment_score > 0.05: sentiment_category = "Positive"
    if latest_sentiment_score < -0.05: sentiment_category = "Negative"
    
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
