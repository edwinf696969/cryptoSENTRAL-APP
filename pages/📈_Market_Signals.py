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
    # The script looks for the data file in the root directory of the repository
    structured_path = Path.cwd() / "stage_2_structured_features.csv"
    if not structured_path.exists():
        st.error("Data file 'stage_2_structured_features.csv' not found. Please ensure it's in the root of your GitHub repository.")
        st.stop()
    return pd.read_csv(structured_path, parse_dates=['date'])

# ===========================================================================
# 3. UI COMPONENTS
# ===========================================================================

def plot_momentum_bar_chart(df):
    """Creates an interactive bar chart for token momentum."""
    # Check if required columns exist
    required_cols = ['symbol', 'date', 'volume_usd', 'momentum_14d', 'momentum_42d']
    if not all(col in df.columns for col in required_cols):
        st.error("Momentum data is missing from the input file. Please ensure the Part A script ran successfully and generated the 'momentum_14d' and 'momentum_42d' columns.")
        return None

    latest_data = df.loc[df.groupby('symbol')['date'].idxmax()]
    top_symbols = latest_data.nlargest(12, 'volume_usd')['symbol']
    plot_df = latest_data[latest_data['symbol'].isin(top_symbols)]

    # Melt the dataframe to have a long format for Plotly Express
    plot_df_melted = plot_df.melt(
        id_vars='symbol',
        value_vars=['momentum_14d', 'momentum_42d'],
        var_name='Period',
        value_name='Momentum'
    )
    
    fig = px.bar(
        plot_df_melted, x='symbol', y='Momentum', color='Period',
        title="Token Momentum (14-Day vs 42-Day)",
        labels={'Momentum': 'Momentum (Cumulative Return)', 'symbol': 'Symbol'},
        barmode='group',
        color_discrete_map={'momentum_14d': '#22D3EE', 'momentum_42d': '#A78BFA'}
    )
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white")
    return fig

def plot_rolling_volatility(df):
    """Creates an interactive line chart for rolling volatility."""
    # Check if required columns exist
    required_cols = ['symbol', 'date', 'volume_usd', 'volatility_28d']
    if not all(col in df.columns for col in required_cols):
        st.error("Volatility data is missing from the input file. Please ensure the Part A script ran successfully and generated the 'volatility_28d' column.")
        return None

    top_symbols = df.loc[df.groupby('symbol')['date'].idxmax()].nlargest(8, 'volume_usd')['symbol']
    plot_df = df[df['symbol'].isin(top_symbols)]
    
    fig = px.line(
        plot_df, x='date', y='volatility_28d', color='symbol',
        title="Rolling Volatility (28-Day)",
        labels={'volatility_28d': '28-Day Ann. Volatility', 'date': 'Date'}
    )
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white", legend_title_text='Symbol')
    return fig

def plot_correlation_matrix(df):
    """Creates an interactive heatmap for the token correlation matrix."""
    # Check if required columns exist
    required_cols = ['date', 'symbol', 'return', 'volume_usd']
    if not all(col in df.columns for col in required_cols):
        st.error("Return data is missing from the input file, which is needed for the correlation matrix.")
        return None
        
    returns_wide = df.pivot(index='date', columns='symbol', values='return')
    top_symbols = df.loc[df.groupby('symbol')['date'].idxmax()].nlargest(10, 'volume_usd')['symbol']
    corr_matrix = returns_wide[top_symbols].corr()
    
    fig = px.imshow(
        corr_matrix, text_auto=".2f", aspect="auto",
        title="Token Correlation Matrix (Weekly Returns)",
        color_continuous_scale='RdBu_r', zmin=-1, zmax=1
    )
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white")
    return fig

# ===========================================================================
# 4. BUILD THE USER INTERFACE
# ===========================================================================

st.markdown("""<style>.main { background-color: #030712; } h1, h2, h3, h4 { color: #F9FAFB; } .st-emotion-cache-16txtl3 { background-color: #111827; border: 1px solid #374151; border-radius: 0.75rem; padding: 1.5rem; }</style>""", unsafe_allow_html=True)
st.title("📈 Market Signals")
st.markdown("---")

structured_data = load_structured_data()

factor_to_view = st.selectbox(
    'Select a Factor to View',
    ('Momentum', 'Volatility', 'Correlation')
)

if factor_to_view == 'Momentum':
    fig = plot_momentum_bar_chart(structured_data)
    if fig: st.plotly_chart(fig, use_container_width=True)
elif factor_to_view == 'Volatility':
    fig = plot_rolling_volatility(structured_data)
    if fig: st.plotly_chart(fig, use_container_width=True)
elif factor_to_view == 'Correlation':
    fig = plot_correlation_matrix(structured_data)
    if fig: st.plotly_chart(fig, use_container_width=True)