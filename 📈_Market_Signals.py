# ===========================================================================
# 1. IMPORT LIBRARIES AND INITIAL SETUP
# ===========================================================================
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# --- Page Configuration ---
st.set_page_config(
    page_title="CryptoSENTRAL | Market Signals",
    page_icon="📈",
    layout="wide",
)

# ===========================================================================
# 2. DATA LOADING AND CACHING
# ===========================================================================

@st.cache_data
def load_structured_data():
    """Loads the clean structured data from Part A."""
    structured_path = Path("stage_2_structured_features.csv")
    if not structured_path.exists():
        st.error("Data file 'stage_2_structured_features.csv' not found. Please ensure it's in your GitHub repository.")
        st.stop()
    return pd.read_csv(structured_path, parse_dates=['date'])

# ===========================================================================
# 3. UI COMPONENTS
# ===========================================================================

def plot_momentum_bar_chart(df):
    """Creates an interactive bar chart for token momentum."""
    # Get the most recent data for each symbol
    latest_data = df.loc[df.groupby('symbol')['date'].idxmax()]
    
    # Select a subset of top coins by recent volume for clarity
    top_symbols = latest_data.nlargest(12, 'volume_usd')['symbol']
    plot_df = latest_data[latest_data['symbol'].isin(top_symbols)]

    fig = px.bar(
        plot_df,
        x='symbol',
        y=['momentum_7d', 'momentum_30d'],
        title="Token Momentum (7-Day vs 30-Day Returns)",
        labels={'value': 'Momentum (Cumulative Return)', 'symbol': 'Symbol'},
        barmode='group',
        color_discrete_map={'momentum_7d': '#22D3EE', 'momentum_30d': '#A78BFA'}
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        legend_title_text='Period'
    )
    return fig

def plot_rolling_volatility(df):
    """Creates an interactive line chart for rolling volatility."""
    # Select a subset of top coins for clarity
    top_symbols = df.loc[df.groupby('symbol')['date'].idxmax()].nlargest(8, 'volume_usd')['symbol']
    plot_df = df[df['symbol'].isin(top_symbols)]
    
    fig = px.line(
        plot_df,
        x='date',
        y='volatility_14d',
        color='symbol',
        title="Rolling Volatility (14-Day)",
        labels={'volatility_14d': '14-Day Ann. Volatility', 'date': 'Date'}
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        legend_title_text='Symbol'
    )
    return fig

def plot_correlation_matrix(df):
    """Creates an interactive heatmap for the token correlation matrix."""
    # Pivot to get returns in wide format
    returns_wide = df.pivot(index='date', columns='symbol', values='return')
    
    # Select top symbols
    top_symbols = df.loc[df.groupby('symbol')['date'].idxmax()].nlargest(10, 'volume_usd')['symbol']
    corr_matrix = returns_wide[top_symbols].corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title="Token Correlation Matrix (Weekly Returns)",
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white"
    )
    return fig

# ===========================================================================
# 4. BUILD THE USER INTERFACE
# ===========================================================================

# --- Custom CSS for a professional, dark theme ---
st.markdown("""
    <style>
        .main { background-color: #030712; }
        h1, h2, h3, h4 { color: #F9FAFB; }
        .st-emotion-cache-16txtl3 {
            background-color: #111827; border: 1px solid #374151;
            border-radius: 0.75rem; padding: 1.5rem;
        }
    </style>
""", unsafe_allow_html=True)

# --- Main App Header ---
st.title("Market Signals")
st.markdown("---")

# --- Load Data ---
structured_data = load_structured_data()

# --- Display Charts ---
st.plotly_chart(plot_momentum_bar_chart(structured_data), use_container_width=True)
st.plotly_chart(plot_rolling_volatility(structured_data), use_container_width=True)
st.plotly_chart(plot_correlation_matrix(structured_data), use_container_width=True)