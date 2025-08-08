# ===========================================================================
# 1. IMPORT LIBRARIES AND INITIAL SETUP
# ===========================================================================
import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.express as px
from pypfopt import EfficientFrontier, risk_models, expected_returns

# --- Page Configuration ---
st.set_page_config(
    page_title="CryptoSENTRAL | Portfolio Builder",
    page_icon="🛠️",
    layout="wide",
)

# ===========================================================================
# 2. DATA LOADING AND CACHING
# ===========================================================================

@st.cache_data
def load_structured_data():
    """Loads the clean structured data from Part A."""
    structured_path = Path.cwd() / "stage_2_structured_features.csv"
    if not structured_path.exists():
        st.error("Data file 'stage_2_structured_features.csv' not found.")
        st.stop()
    return pd.read_csv(structured_path, parse_dates=['date'])

# ===========================================================================
# 3. PORTFOLIO OPTIMIZATION LOGIC
# ===========================================================================

def optimize_portfolio(df, selected_assets):
    """
    Performs Mean-Variance Optimization to find the portfolio with the max Sharpe ratio.
    """
    if len(selected_assets) < 2:
        return None, None

    # Filter for selected assets and pivot to get weekly returns in wide format
    returns_wide = df[df['symbol'].isin(selected_assets)].pivot(index='date', columns='symbol', values='return')
    returns_wide = returns_wide.dropna()

    if returns_wide.empty or len(returns_wide) < 60: # Need sufficient data for stable calculation
        return None, None

    # Calculate expected returns and sample covariance matrix
    mu = expected_returns.mean_historical_return(returns_wide)
    S = risk_models.sample_cov(returns_wide)

    # Optimize for the maximal Sharpe ratio
    ef = EfficientFrontier(mu, S)
    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()
    
    performance = ef.portfolio_performance(verbose=False)
    return cleaned_weights, performance

# ===========================================================================
# 4. BUILD THE USER INTERFACE
# ===========================================================================

st.markdown("""<style>.main { background-color: #030712; } h1, h2, h3, h4 { color: #F9FAFB; } .st-emotion-cache-16txtl3 { background-color: #111827; border: 1px solid #374151; border-radius: 0.75rem; padding: 1.5rem; }</style>""", unsafe_allow_html=True)
st.title("🛠️ Build My Portfolio")
st.markdown("---")

# --- Load Data ---
structured_data = load_structured_data()
# Get a list of assets with the most data for reliability
asset_counts = structured_data['symbol'].value_counts()
reliable_assets = asset_counts[asset_counts > 100].index.tolist()
available_assets = sorted(reliable_assets)


# --- User Input Section ---
st.subheader("1. Select Your Assets")
selected_assets = st.multiselect(
    "Choose at least two assets for your portfolio:",
    available_assets,
    default=['BTC', 'ETH', 'SOL', 'XRP', 'ADA', 'DOGE', 'AVAX']
)

if len(selected_assets) < 2:
    st.warning("Please select at least two assets to build an optimized portfolio.")
else:
    st.subheader("2. Run Optimization")
    if st.button("✨ Optimize My Portfolio"):
        with st.spinner("Calculating optimal weights based on historical risk and return..."):
            weights, performance = optimize_portfolio(structured_data, selected_assets)
        
        if weights:
            st.subheader("3. Optimized Portfolio Allocation (Max Sharpe Ratio)")
            
            weights_df = pd.DataFrame.from_dict(weights, orient='index', columns=['Weight'])
            weights_df.index.name = 'Asset'
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.dataframe(weights_df.style.format({"Weight": "{:.2%}"}))
                st.metric("Expected Annual Return", f"{performance[0]:.2%}")
                st.metric("Annual Volatility", f"{performance[1]:.2%}")
                st.metric("Sharpe Ratio", f"{performance[2]:.2f}")
            
            with col2:
                fig = px.pie(
                    weights_df, values='Weight', names=weights_df.index,
                    title='Optimized Portfolio Allocation', hole=.3
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Could not optimize the portfolio. There might be insufficient historical data for the selected assets. Please try a different combination.")