import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# Page config
st.set_page_config(
    page_title="Falling Knife Detector",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject CSS directly into head for maximum priority
components.html("""
<script>
(function() {
    var style = document.createElement('style');
    style.type = 'text/css';
    style.innerHTML = `
    <style>
    /* Force dark theme on ALL elements - highest priority */
    * {
        color: #fafafa !important;
    }
    
    /* Main app dark theme */
    .stApp, .stAppViewContainer, #root {
        background-color: #0e1117 !important;
        color: #fafafa !important;
    }
    
    /* Main container */
    .main .block-container, [data-testid="stAppViewContainer"] {
        background-color: #0e1117 !important;
        color: #fafafa !important;
    }
    
    /* Force all divs to dark */
    div {
        background-color: transparent !important;
    }
    div[style*="background"] {
        background-color: #0e1117 !important;
    }
    
    /* Header styling */
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #FFD700;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #888;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    /* Sidebar dark theme */
    .css-1d391kg {
        background-color: #1e1e1e !important;
    }
    [data-testid="stSidebar"] {
        background-color: #1e1e1e !important;
    }
    
    /* Metrics dark theme */
    .stMetric {
        background-color: #1e1e1e !important;
        border: 1px solid #333 !important;
        padding: 10px !important;
        border-radius: 5px !important;
    }
    .stMetric label {
        color: #fafafa !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #fafafa !important;
    }
    
    /* Text inputs */
    .stTextInput > div > div > input {
        background-color: #1e1e1e !important;
        color: #fafafa !important;
        border: 1px solid #333 !important;
    }
    .stTextInput label {
        color: #fafafa !important;
    }
    
    /* Selectbox (theme selector) */
    .stSelectbox > div > div {
        background-color: #1e1e1e !important;
    }
    .stSelectbox > div > div > select {
        background-color: #1e1e1e !important;
        color: #fafafa !important;
        border: 1px solid #333 !important;
    }
    .stSelectbox label {
        color: #fafafa !important;
    }
    /* Selectbox dropdown options */
    .stSelectbox [data-baseweb="select"] {
        background-color: #1e1e1e !important;
    }
    .stSelectbox [data-baseweb="popover"] {
        background-color: #1e1e1e !important;
    }
    .stSelectbox [data-baseweb="menu"] {
        background-color: #1e1e1e !important;
    }
    .stSelectbox [data-baseweb="option"] {
        background-color: #1e1e1e !important;
        color: #fafafa !important;
    }
    .stSelectbox [data-baseweb="option"]:hover {
        background-color: #2e2e2e !important;
    }
    
    /* Date inputs */
    .stDateInput > div > div > input {
        background-color: #1e1e1e !important;
        color: #fafafa !important;
        border: 1px solid #333 !important;
    }
    .stDateInput label {
        color: #fafafa !important;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #4CAF50 !important;
        color: white !important;
        border-radius: 5px !important;
        border: none !important;
    }
    .stButton > button:hover {
        background-color: #45a049 !important;
    }
    
    /* Dataframes */
    .dataframe {
        background-color: #1e1e1e !important;
        color: #fafafa !important;
    }
    .dataframe thead {
        background-color: #2e2e2e !important;
        color: #fafafa !important;
    }
    .dataframe tbody tr {
        background-color: #1e1e1e !important;
        color: #fafafa !important;
    }
    .dataframe tbody tr:nth-child(even) {
        background-color: #252525 !important;
    }
    /* Fix Streamlit dataframe component */
    [data-testid="stDataFrame"] {
        background-color: #1e1e1e !important;
    }
    [data-testid="stDataFrame"] table {
        background-color: #1e1e1e !important;
    }
    [data-testid="stDataFrame"] thead {
        background-color: #2e2e2e !important;
    }
    [data-testid="stDataFrame"] tbody {
        background-color: #1e1e1e !important;
    }
    [data-testid="stDataFrame"] td, [data-testid="stDataFrame"] th {
        background-color: #1e1e1e !important;
        color: #fafafa !important;
        border-color: #333 !important;
    }
    [data-testid="stDataFrame"] thead th {
        background-color: #2e2e2e !important;
        color: #fafafa !important;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background-color: #1e1e1e !important;
        color: #fafafa !important;
        border: 1px solid #333 !important;
    }
    .streamlit-expanderContent {
        background-color: #1e1e1e !important;
        color: #fafafa !important;
    }
    /* Fix expander container background */
    [data-testid="stExpander"] {
        background-color: #1e1e1e !important;
    }
    [data-testid="stExpander"] > div {
        background-color: #1e1e1e !important;
    }
    
    /* Info/Warning/Error messages */
    .stInfo {
        background-color: #1e3a5f !important;
        border-left: 4px solid #2196F3 !important;
    }
    .stWarning {
        background-color: #5a3a1e !important;
        border-left: 4px solid #FF9800 !important;
    }
    .stError {
        background-color: #5a1e1e !important;
        border-left: 4px solid #F44336 !important;
    }
    
    /* All text should be readable */
    p, div, span, h1, h2, h3, h4, h5, h6 {
        color: #fafafa !important;
    }
    
    /* Fix any remaining white backgrounds */
    section[data-testid="stSidebar"] > div {
        background-color: #1e1e1e !important;
    }
    .element-container {
        color: #fafafa !important;
    }
    
    /* Fix Streamlit's default white elements */
    .stApp > header {
        background-color: #0e1117 !important;
    }
    div[data-baseweb="base-input"] {
        background-color: #1e1e1e !important;
    }
    div[data-baseweb="select"] {
        background-color: #1e1e1e !important;
    }
    
    /* Fix table cells */
    .dataframe td, .dataframe th {
        color: #fafafa !important;
        background-color: #1e1e1e !important;
    }
    
    /* Fix expander content text */
    .streamlit-expanderContent p, .streamlit-expanderContent div {
        color: #fafafa !important;
    }
    
    /* Fix all Streamlit widgets */
    [data-baseweb="input"] {
        background-color: #1e1e1e !important;
        color: #fafafa !important;
    }
    [data-baseweb="input"] input {
        background-color: #1e1e1e !important;
        color: #fafafa !important;
    }
    
    /* Fix date picker */
    .stDateInput [data-baseweb="calendar"] {
        background-color: #1e1e1e !important;
    }
    .stDateInput [data-baseweb="popover"] {
        background-color: #1e1e1e !important;
    }
    .stDateInput [data-baseweb="input"] {
        background-color: #1e1e1e !important;
        border: 1px solid #333 !important;
    }
    .stDateInput [data-baseweb="input"] input {
        background-color: #1e1e1e !important;
        color: #fafafa !important;
    }
    /* Fix calendar popup */
    [data-baseweb="popover"] {
        background-color: #1e1e1e !important;
    }
    [data-baseweb="calendar"] {
        background-color: #1e1e1e !important;
    }
    /* Remove white squares for invalid dates */
    [data-baseweb="calendar"] [aria-disabled="true"] {
        background-color: transparent !important;
        color: #666 !important;
        visibility: hidden !important;
    }
    /* Calendar day cells */
    [data-baseweb="calendar"] button {
        background-color: #2e2e2e !important;
        color: #fafafa !important;
        border: 1px solid #333 !important;
    }
    [data-baseweb="calendar"] button:hover {
        background-color: #3e3e3e !important;
    }
    [data-baseweb="calendar"] button[aria-selected="true"] {
        background-color: #4CAF50 !important;
        color: white !important;
    }
    /* Calendar header */
    [data-baseweb="calendar"] [data-baseweb="select"] {
        background-color: #1e1e1e !important;
        color: #fafafa !important;
    }
    
    /* Fix any remaining white backgrounds in containers */
    #MainMenu {
        visibility: hidden;
    }
    footer {
        visibility: hidden;
    }
    header {
        visibility: hidden;
    }
    
    /* Fix markdown text colors */
    .stMarkdown, .stMarkdown * {
        color: #fafafa !important;
    }
    .stMarkdown p, .stMarkdown div, .stMarkdown span, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #fafafa !important;
    }
    
    /* Fix column containers */
    [data-testid="column"], [data-testid="stVerticalBlock"] {
        background-color: transparent !important;
    }
    
    /* Fix ALL white backgrounds - catch everything */
    [style*="background-color: white"], [style*="background-color: #fff"], [style*="background-color: #ffffff"] {
        background-color: #1e1e1e !important;
    }
    
    /* Fix Streamlit's internal white elements */
    .stApp > div, .stApp > div > div {
        background-color: #0e1117 !important;
    }
    
    /* Fix plotly container backgrounds */
    .js-plotly-plot {
        background-color: transparent !important;
    }
    
    /* Fix any table or list backgrounds */
    table, thead, tbody, tr, td, th, ul, ol, li {
        background-color: #1e1e1e !important;
        color: #fafafa !important;
    }
    
    /* Theme selector styling */
    .theme-selector-container {
        display: flex;
        align-items: center;
        height: 100%;
    }
    
    /* Override Streamlit's default white with !important everywhere */
    body {
        background-color: #0e1117 !important;
        color: #fafafa !important;
    }
    
    /* Fix expander behind content */
    [data-testid="stExpander"] {
        background-color: #1e1e1e !important;
    }
    [data-testid="stExpander"] > div {
        background-color: #1e1e1e !important;
    }
    [data-testid="stExpander"] > div > div {
        background-color: #1e1e1e !important;
    }
    
    /* Fix date input visibility */
    [data-baseweb="input"] {
        background-color: #1e1e1e !important;
        border: 1px solid #555 !important;
    }
    [data-baseweb="input"] input {
        background-color: #1e1e1e !important;
        color: #fafafa !important;
    }
    
    /* Fix calendar completely */
    [data-baseweb="popover"] {
        background-color: #1e1e1e !important;
    }
    [data-baseweb="calendar"] {
        background-color: #1e1e1e !important;
    }
    [data-baseweb="calendar"] button[aria-disabled="true"] {
        visibility: hidden !important;
    }
    [data-baseweb="calendar"] button {
        background-color: #2e2e2e !important;
        color: #fafafa !important;
        border: 1px solid #333 !important;
    }
    
    /* Fix dataframe completely */
    [data-testid="stDataFrame"] * {
        background-color: #1e1e1e !important;
        color: #fafafa !important;
    }
    [data-testid="stDataFrame"] thead * {
        background-color: #2e2e2e !important;
        color: #fafafa !important;
    }
    `;
    document.head.appendChild(style);
    
    // Aggressive dark theme enforcement
    function forceDarkTheme() {
        // Fix expander backgrounds
        document.querySelectorAll('[data-testid="stExpander"]').forEach(function(el) {
            el.style.backgroundColor = '#1e1e1e';
            var children = el.querySelectorAll('*');
            children.forEach(function(child) {
                var bg = window.getComputedStyle(child).backgroundColor;
                if (bg === 'rgb(255, 255, 255)' || bg === 'white') {
                    child.style.backgroundColor = '#1e1e1e';
                }
            });
        });
        
        // Fix date inputs
        document.querySelectorAll('[data-baseweb="input"]').forEach(function(el) {
            el.style.backgroundColor = '#1e1e1e';
            el.style.border = '1px solid #555';
            var input = el.querySelector('input');
            if (input) {
                input.style.backgroundColor = '#1e1e1e';
                input.style.color = '#fafafa';
            }
        });
        
        // Fix calendar
        document.querySelectorAll('[data-baseweb="calendar"] button[aria-disabled="true"]').forEach(function(el) {
            el.style.visibility = 'hidden';
        });
        
        // Fix dataframe
        document.querySelectorAll('[data-testid="stDataFrame"] td, [data-testid="stDataFrame"] th').forEach(function(el) {
            el.style.backgroundColor = '#1e1e1e';
            el.style.color = '#fafafa';
        });
        document.querySelectorAll('[data-testid="stDataFrame"] thead th').forEach(function(el) {
            el.style.backgroundColor = '#2e2e2e';
            el.style.color = '#fafafa';
        });
        
        // Fix Plotly legend text
        document.querySelectorAll('.legendtext').forEach(function(el) {
            el.style.fill = '#fafafa';
            el.style.color = '#fafafa';
        });
    }
    
    // Run immediately and on intervals
    forceDarkTheme();
    setInterval(forceDarkTheme, 300);
    
    // Watch for new elements
    var observer = new MutationObserver(function(mutations) {
        forceDarkTheme();
    });
    
    observer.observe(document.body, {
        childList: true,
        subtree: true,
        attributes: true,
        attributeFilter: ['style', 'class']
    });
    
    window.addEventListener('load', forceDarkTheme);
    document.addEventListener('DOMContentLoaded', forceDarkTheme);
})();
</script>
""", height=0)


def download_price_data(ticker: str, start: str) -> pd.DataFrame:
    """Download price data from yfinance."""
    with st.spinner(f"Downloading data for {ticker}..."):
        data = yf.download(
            ticker,
            start=start,
            progress=False,
            auto_adjust=False,
        )
    if data.empty:
        raise ValueError(f"No price data returned for ticker '{ticker}'.")
    return data


def select_price_series(prices: pd.DataFrame, ticker: str) -> pd.Series:
    """Select the appropriate price series from downloaded data."""
    if isinstance(prices.columns, pd.MultiIndex):
        for candidate in ("Adj Close", "Close"):
            try:
                series = prices.xs(candidate, axis=1, level=0)
            except KeyError:
                continue
            if isinstance(series, pd.DataFrame):
                if series.shape[1] == 1:
                    return series.iloc[:, 0]
                if ticker.upper() in series.columns:
                    return series[ticker.upper()]
                return series.iloc[:, 0]
            else:
                return series
    else:
        for candidate in ("Adj Close", "Close"):
            if candidate in prices.columns:
                return prices[candidate]
    raise KeyError("Could not find an adjusted or close price column in the downloaded data.")


def select_volume_series(prices: pd.DataFrame, ticker: str) -> pd.Series:
    """Select the appropriate volume series from downloaded data."""
    if isinstance(prices.columns, pd.MultiIndex):
        try:
            series = prices.xs("Volume", axis=1, level=0)
        except KeyError:
            raise KeyError("Could not find Volume column in the downloaded data.")
        if isinstance(series, pd.DataFrame):
            if series.shape[1] == 1:
                return series.iloc[:, 0]
            if ticker.upper() in series.columns:
                return series[ticker.upper()]
            return series.iloc[:, 0]
        else:
            return series
    else:
        if "Volume" in prices.columns:
            return prices["Volume"]
    raise KeyError("Could not find Volume column in the downloaded data.")


def calculate_panic_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate panic signals with clustering and severity scoring."""
    # Daily return
    df["ret"] = df["Adj Close"].pct_change()
    
    # 1) RETURN Z-SCORE MODEL
    ret_mean = df["ret"].rolling(60).mean()
    ret_std = df["ret"].rolling(60).std()
    df["z_return"] = np.where(
        ret_std != 0,
        (df["ret"] - ret_mean) / ret_std,
        np.nan
    )
    
    # 2) VOLATILITY SPIKE MODEL
    df["vol20"] = df["ret"].rolling(20).std() * np.sqrt(252)
    df["vol100"] = df["ret"].rolling(100).std() * np.sqrt(252)
    df["vol_spike"] = np.where(
        df["vol100"] != 0,
        df["vol20"] > 2 * df["vol100"],
        False
    )
    
    # 3) VOLUME SPIKE
    vol_mean = df["Volume"].rolling(60).mean()
    vol_std = df["Volume"].rolling(60).std()
    df["z_volume"] = np.where(
        vol_std != 0,
        (df["Volume"] - vol_mean) / vol_std,
        np.nan
    )
    
    df["volume_spike"] = df["z_volume"] > 3
    
    # Individual panic type flags
    df["return_panic"] = df["z_return"] < -3
    df["volatility_panic"] = df["vol_spike"]
    df["volume_panic"] = df["volume_spike"]
    
    # Raw panic score (0-3)
    df["raw_panic_score"] = (
        df["return_panic"].astype(int) +
        df["volatility_panic"].astype(int) +
        df["volume_panic"].astype(int)
    )
    
    # PANIC CLUSTERING
    df["panic_cluster_count"] = (
        df["raw_panic_score"]
        .rolling(window=11, center=True, min_periods=1)
        .sum()
        .fillna(0)
    )
    
    # SEVERITY SCORING (0-3)
    df["panic_severity"] = 0
    
    # Severity 1: Single panic type, no clustering
    df.loc[
        (df["raw_panic_score"] == 1) & (df["panic_cluster_count"] <= 1),
        "panic_severity"
    ] = 1
    
    # Severity 2: Multiple panic types OR clustering
    df.loc[
        ((df["raw_panic_score"] >= 2) | (df["panic_cluster_count"] >= 2)) &
        (df["panic_severity"] == 0),
        "panic_severity"
    ] = 2
    
    # Severity 3: All three conditions + clustering
    df.loc[
        (df["return_panic"] & df["volatility_panic"] & df["volume_panic"]) &
        (df["panic_cluster_count"] >= 2),
        "panic_severity"
    ] = 3
    
    # LONG-TERM BOTTOM DETECTOR
    df["long_term_bottom"] = (
        df["return_panic"] &
        (df["volume_panic"] | df["volatility_panic"]) &
        (df["panic_cluster_count"] >= 2) &
        (df["panic_severity"] >= 2)
    )
    
    df["panic_signal"] = df["panic_severity"] >= 2
    
    # Calculate forward returns
    df["return_30d"] = df["Adj Close"].pct_change(30).shift(-30) * 100
    df["return_60d"] = df["Adj Close"].pct_change(60).shift(-60) * 100
    df["return_120d"] = df["Adj Close"].pct_change(120).shift(-120) * 100
    
    return df


def identify_panic_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """Group panic signals into clusters and identify cluster centers."""
    df = df.copy()
    df["cluster_id"] = 0
    df["is_cluster_center"] = False
    df["cluster_severity"] = 0
    df["cluster_start"] = None
    df["cluster_end"] = None
    
    panic_dates = df[df["panic_signal"]].index
    if len(panic_dates) == 0:
        return df
    
    # Group consecutive or nearby panic dates into clusters
    clusters = []
    current_cluster = []
    cluster_gap_days = 5
    
    for i, date in enumerate(panic_dates):
        if len(current_cluster) == 0:
            current_cluster.append(date)
        else:
            days_gap = (date - current_cluster[-1]).days
            if days_gap <= cluster_gap_days:
                current_cluster.append(date)
            else:
                if len(current_cluster) > 0:
                    clusters.append(current_cluster)
                current_cluster = [date]
    
    if len(current_cluster) > 0:
        clusters.append(current_cluster)
    
    # Process each cluster
    cluster_num = 1
    for cluster in clusters:
        if len(cluster) == 0:
            continue
        
        cluster_dates = pd.DatetimeIndex(cluster)
        cluster_df = df.loc[cluster_dates]
        
        cluster_center = cluster_df.loc[cluster_df["Adj Close"].idxmin()]
        center_date = cluster_center.name
        
        max_severity = cluster_df["panic_severity"].max()
        is_long_term_bottom = cluster_df["long_term_bottom"].any()
        
        df.loc[cluster_dates, "cluster_id"] = cluster_num
        df.loc[center_date, "is_cluster_center"] = True
        df.loc[center_date, "cluster_severity"] = max_severity
        df.loc[center_date, "cluster_start"] = cluster_dates.min()
        df.loc[center_date, "cluster_end"] = cluster_dates.max()
        
        if is_long_term_bottom:
            df.loc[center_date, "long_term_bottom"] = True
        
        cluster_num += 1
    
    return df


def process_ticker(ticker: str, start: str) -> pd.DataFrame:
    """Process a single ticker and return the dataframe with panic signals."""
    prices = download_price_data(ticker, start)
    price_series = select_price_series(prices, ticker)
    volume_series = select_volume_series(prices, ticker)
    
    df = pd.DataFrame(index=prices.index)
    df["Adj Close"] = price_series
    df["Volume"] = volume_series
    
    df = calculate_panic_signals(df)
    df = df.dropna(subset=["ret", "z_return", "vol20", "vol100", "z_volume"])
    
    if df.empty:
        raise ValueError("Not enough data to compute panic signals. Try an earlier start date.")
    
    df = identify_panic_clusters(df)
    return df


def get_color_theme(theme_name: str) -> dict:
    """Get color theme configuration."""
    themes = {
        "Dark Mode (Default)": {
            "price_color": "#FFD700",
            "strong_bottom_color": "#FFFFFF",
            "strong_bottom_border": "#FF0000",
            "severity3_color": "#FF0000",
            "severity3_border": "#FFFFFF",
            "severity2_color": "#FFA500",
            "severity2_border": "#FFFFFF",
            "bg_color": "rgba(0,0,0,0)",
            "text_color": "#fafafa",
            "grid_color": "rgba(255, 255, 255, 0.1)"
        },
        "Blue Ocean": {
            "price_color": "#00D4FF",
            "strong_bottom_color": "#FFFF00",
            "strong_bottom_border": "#FF6B6B",
            "severity3_color": "#FF6B6B",
            "severity3_border": "#FFFFFF",
            "severity2_color": "#4ECDC4",
            "severity2_border": "#FFFFFF",
            "bg_color": "rgba(0,0,0,0)",
            "text_color": "#fafafa",
            "grid_color": "rgba(0, 212, 255, 0.1)"
        },
        "Green Energy": {
            "price_color": "#00FF88",
            "strong_bottom_color": "#FFD700",
            "strong_bottom_border": "#FF4444",
            "severity3_color": "#FF4444",
            "severity3_border": "#FFFFFF",
            "severity2_color": "#FFAA00",
            "severity2_border": "#FFFFFF",
            "bg_color": "rgba(0,0,0,0)",
            "text_color": "#fafafa",
            "grid_color": "rgba(0, 255, 136, 0.1)"
        },
        "Purple Night": {
            "price_color": "#B794F6",
            "strong_bottom_color": "#FED7AA",
            "strong_bottom_border": "#F87171",
            "severity3_color": "#F87171",
            "severity3_border": "#FFFFFF",
            "severity2_color": "#FBBF24",
            "severity2_border": "#FFFFFF",
            "bg_color": "rgba(0,0,0,0)",
            "text_color": "#fafafa",
            "grid_color": "rgba(183, 148, 246, 0.1)"
        },
        "White Line": {
            "price_color": "#FFFFFF",
            "strong_bottom_color": "#FFFFFF",
            "strong_bottom_border": "#FF0000",
            "severity3_color": "#FF0000",
            "severity3_border": "#FFFFFF",
            "severity2_color": "#FFA500",
            "severity2_border": "#FFFFFF",
            "bg_color": "rgba(0,0,0,0)",
            "text_color": "#FFFFFF",
            "grid_color": "rgba(255, 255, 255, 0.2)"
        }
    }
    return themes.get(theme_name, themes["Dark Mode (Default)"])


def create_plotly_chart(ticker: str, df: pd.DataFrame, theme: dict) -> go.Figure:
    """Create an interactive Plotly chart."""
    fig = go.Figure()
    
    # Price line
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["Adj Close"],
        mode='lines',
        name='Price',
        line=dict(color=theme["price_color"], width=2),
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Price: $%{y:.2f}<extra></extra>'
    ))
    
    # Get cluster centers
    cluster_centers = df[df["is_cluster_center"]].copy()
    
    if len(cluster_centers) > 0:
        # Add shaded regions for clusters
        for _, row in cluster_centers.iterrows():
            if pd.notna(row["cluster_start"]) and pd.notna(row["cluster_end"]):
                start = row["cluster_start"]
                end = row["cluster_end"]
                severity = row["cluster_severity"]
                
                if severity == 3:
                    color = "rgba(255, 0, 0, 0.2)"
                elif severity == 2:
                    color = "rgba(255, 165, 0, 0.12)"
                else:
                    color = "rgba(255, 165, 0, 0.08)"
                
                fig.add_vrect(
                    x0=start,
                    x1=end,
                    fillcolor=color,
                    layer="below",
                    line_width=0,
                )
        
        # Plot markers by type
        long_term_bottoms = cluster_centers[cluster_centers["long_term_bottom"]]
        severity_3 = cluster_centers[(cluster_centers["cluster_severity"] == 3) & ~cluster_centers["long_term_bottom"]]
        severity_2 = cluster_centers[(cluster_centers["cluster_severity"] == 2) & ~cluster_centers["long_term_bottom"]]
        
        # Long-term bottoms - star
        if len(long_term_bottoms) > 0:
            fig.add_trace(go.Scatter(
                x=long_term_bottoms.index,
                y=long_term_bottoms["Adj Close"],
                mode='markers',
                name='Strong Bottom *',
                marker=dict(
                    symbol='star',
                    size=15,
                    color=theme["strong_bottom_color"],
                    line=dict(color=theme["strong_bottom_border"], width=2)
                ),
                hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Price: $%{y:.2f}<br>Strong Bottom<extra></extra>'
            ))
        
        # Severity 3 - circle
        if len(severity_3) > 0:
            fig.add_trace(go.Scatter(
                x=severity_3.index,
                y=severity_3["Adj Close"],
                mode='markers',
                name='Severity 3',
                marker=dict(
                    size=12,
                    color=theme["severity3_color"],
                    line=dict(color=theme["severity3_border"], width=1.5)
                ),
                hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Price: $%{y:.2f}<br>Severity 3<extra></extra>'
            ))
        
        # Severity 2 - triangle
        if len(severity_2) > 0:
            fig.add_trace(go.Scatter(
                x=severity_2.index,
                y=severity_2["Adj Close"],
                mode='markers',
                name='Panic Signal',
                marker=dict(
                    symbol='triangle-up',
                    size=10,
                    color=theme["severity2_color"],
                    line=dict(color=theme["severity2_border"], width=1)
                ),
                hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Price: $%{y:.2f}<br>Panic Signal<extra></extra>'
            ))
    
    # Update layout with theme
    template = 'plotly_dark' if theme["bg_color"] == "rgba(0,0,0,0)" else 'plotly'
    fig.update_layout(
        title=dict(
            text=f"{ticker.upper()} – Long-Term Bottom Detector",
            font=dict(size=24, color=theme["price_color"]),
            x=0.5,
            xanchor='center'
        ),
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode='x unified',
        template=template,
        height=500,
        plot_bgcolor=theme["bg_color"],
        paper_bgcolor=theme["bg_color"],
        font=dict(color=theme["text_color"]),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(0,0,0,0.5)' if theme["bg_color"] == "rgba(0,0,0,0)" else 'rgba(255,255,255,0.8)',
            bordercolor='rgba(255,255,255,0.2)' if theme["bg_color"] == "rgba(0,0,0,0)" else 'rgba(0,0,0,0.2)',
            borderwidth=1,
            font=dict(color="#fafafa", size=12),
            itemfont=dict(color="#fafafa")
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor=theme["grid_color"],
            color=theme["text_color"]
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=theme["grid_color"],
            color=theme["text_color"]
        )
    )
    
    return fig


def main():
    # Initialize session state
    if 'df_data' not in st.session_state:
        st.session_state.df_data = None
    if 'current_ticker' not in st.session_state:
        st.session_state.current_ticker = None
    if 'start_str' not in st.session_state:
        st.session_state.start_str = None
    if 'top_bar_expanded' not in st.session_state:
        st.session_state.top_bar_expanded = True
    
    # Header with theme selector aligned with title
    header_col1, header_col2, header_col3 = st.columns([1, 8, 1])
    
    with header_col1:
        st.markdown('<div style="height: 60px; display: flex; align-items: center; padding-top: 10px;">', unsafe_allow_html=True)
        color_theme = st.selectbox(
            "🎨 Color Theme",
            ["Dark Mode (Default)", "Blue Ocean", "Green Energy", "Purple Night", "White Line"],
            index=0,
            help="Choose a color theme for the chart",
            key="theme_selector"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with header_col2:
        # Title (centered, compact)
        st.markdown('<h1 class="main-header" style="text-align: center; margin-bottom: 0.5rem; margin-top: 0;">📉 Falling Knife Detector</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header" style="text-align: center; margin-bottom: 1rem;">Long-Term Bottom Detection & Panic Signal Analysis</p>', unsafe_allow_html=True)
    
    with header_col3:
        st.markdown("")  # Empty space for balance
    
    # Get selected theme
    theme = get_color_theme(color_theme)
    
    # Collapsible top bar (open by default)
    with st.expander("⚙️ Settings", expanded=True):
        # Ticker and date inputs in columns
        col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
        
        with col1:
            ticker = st.text_input(
                "Ticker Symbol",
                value="AAPL",
                placeholder="Enter ticker (e.g., AAPL, TSLA, MSFT)",
                help="Enter any stock ticker symbol"
            ).upper().strip()
        
        with col2:
            start_date = st.date_input(
                "Start Date",
                value=datetime(2015, 1, 1),
                max_value=datetime.today(),
                help="Start date for historical data"
            )
        
        with col3:
            end_date = st.date_input(
                "End Date",
                value=datetime.today(),
                max_value=datetime.today(),
                help="End date for historical data"
            )
        
        with col4:
            st.markdown("<br>", unsafe_allow_html=True)  # Spacing
            analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)
        
        start_str = start_date.strftime("%Y-%m-%d")
    
    # Check if we need to re-analyze (ticker changed or button clicked)
    ticker_changed = ticker != st.session_state.current_ticker
    start_changed = start_str != st.session_state.start_str
    
    # Use cached data if available and only theme/end_date changed
    if st.session_state.df_data is not None and st.session_state.current_ticker == ticker and st.session_state.start_str == start_str:
        df = st.session_state.df_data.copy()
        # Re-filter by end date if changed
        if end_date < datetime.today().date():
            df = df[df.index.date <= end_date]
        cluster_centers = df[df["is_cluster_center"]].copy()
        
        # Chart (centered)
        fig = create_plotly_chart(ticker, df, theme)
        st.plotly_chart(fig, use_container_width=True)
        
        # Cluster details (centered)
        if len(cluster_centers) > 0:
            st.markdown('<h2 style="text-align: center; color: #FFD700; margin-top: 20px;">📊 Panic Cluster Details</h2>', unsafe_allow_html=True)
            
            # Prepare display dataframe
            display_df = cluster_centers.copy()
            display_df["Date"] = display_df.index.strftime("%Y-%m-%d")
            display_df["Price"] = display_df["Adj Close"].apply(lambda x: f"${x:.2f}")
            display_df["Signal_Type"] = display_df.apply(
                lambda row: "* STRONG BOTTOM" if row["long_term_bottom"] 
                else "Panic Signal" if row['cluster_severity'] == 2
                else f"Severity {int(row['cluster_severity'])}",
                axis=1
            )
            display_df["Duration"] = display_df.apply(
                lambda row: f"{(row['cluster_end'] - row['cluster_start']).days + 1} days" 
                if pd.notna(row["cluster_start"]) and pd.notna(row["cluster_end"]) 
                else "1 day", axis=1
            )
            
            # Show table
            table_df = display_df[["Date", "Price", "Signal_Type", "Duration"]].sort_index(ascending=False)
            st.dataframe(
                table_df,
                use_container_width=True,
                hide_index=True
            )
            
            # Strong bottom candidates (centered)
            long_term_bottoms = cluster_centers[cluster_centers["long_term_bottom"]]
            if len(long_term_bottoms) > 0:
                st.markdown('<h2 style="text-align: center; color: #FFD700; margin-top: 20px;">🎯 Strong Bottom Candidates (Best Swing Entry Points)</h2>', unsafe_allow_html=True)
                
                for date in long_term_bottoms.sort_index(ascending=False).index[:10]:
                    row = long_term_bottoms.loc[date]
                    duration = (row["cluster_end"] - row["cluster_start"]).days + 1 if pd.notna(row["cluster_start"]) and pd.notna(row["cluster_end"]) else 1
                    
                    with st.expander(f"📅 {date.strftime('%Y-%m-%d')} - ${row['Adj Close']:.2f}"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.write(f"**Price:** ${row['Adj Close']:.2f}")
                            st.write(f"**Severity:** {int(row['cluster_severity'])}")
                        with col2:
                            st.write(f"**Duration:** {int(duration)} days")
                            st.write(f"**Return Z-Score:** {row['z_return']:.2f}")
                        with col3:
                            if pd.notna(row.get('return_30d')):
                                st.write(f"**30d Return:** {row['return_30d']:.1f}%")
                            if pd.notna(row.get('return_60d')):
                                st.write(f"**60d Return:** {row['return_60d']:.1f}%")
        else:
            st.info("ℹ️ No panic clusters detected (Severity ≥ 2). No serious multi-day capitulation events found.")
        
        # Metrics at the bottom
        st.markdown('<h2 style="text-align: center; color: #FFD700; margin-top: 20px;">📈 Summary Statistics</h2>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Total Clusters",
                len(cluster_centers),
                help="Total panic clusters detected"
            )
        
        with col2:
            severity_2_count = (cluster_centers['cluster_severity'] == 2).sum() if len(cluster_centers) > 0 else 0
            st.metric(
                "Panic Signal",
                severity_2_count,
                help="Moderate panic signals (less severe than Strong Bottom)"
            )
        
        with col3:
            strong_bottom_count = cluster_centers['long_term_bottom'].sum() if len(cluster_centers) > 0 else 0
            st.metric(
                "Strong Bottoms",
                strong_bottom_count,
                help="Long-term bottom candidates (includes Severity 3)"
            )
    
    # Main content - analyze if button clicked or ticker/start date changed
    elif analyze_btn or (ticker_changed and ticker) or (start_changed and ticker):
        if not ticker or not ticker.strip():
            st.warning("⚠️ Please enter a ticker symbol")
        else:
            try:
                # Process ticker
                df = process_ticker(ticker, start_str)
                
                # Filter by end date if needed
                if end_date < datetime.today().date():
                    df = df[df.index.date <= end_date]
                
                # Save to session state
                st.session_state.df_data = df.copy()
                st.session_state.current_ticker = ticker
                st.session_state.start_str = start_str
                
                # Get cluster centers
                cluster_centers = df[df["is_cluster_center"]].copy()
                
                # Chart (centered)
                fig = create_plotly_chart(ticker, df, theme)
                st.plotly_chart(fig, use_container_width=True)
                
                # Cluster details (centered)
                if len(cluster_centers) > 0:
                    st.markdown('<h2 style="text-align: center; color: #FFD700; margin-top: 20px;">📊 Panic Cluster Details</h2>', unsafe_allow_html=True)
                    
                    # Prepare display dataframe
                    display_df = cluster_centers.copy()
                    display_df["Date"] = display_df.index.strftime("%Y-%m-%d")
                    display_df["Price"] = display_df["Adj Close"].apply(lambda x: f"${x:.2f}")
                    display_df["Signal_Type"] = display_df.apply(
                        lambda row: "* STRONG BOTTOM" if row["long_term_bottom"] 
                        else f"Severity {int(row['cluster_severity'])}",
                        axis=1
                    )
                    display_df["Duration"] = display_df.apply(
                        lambda row: f"{(row['cluster_end'] - row['cluster_start']).days + 1} days" 
                        if pd.notna(row["cluster_start"]) and pd.notna(row["cluster_end"]) 
                        else "1 day", axis=1
                    )
                    
                    # Show table
                    table_df = display_df[["Date", "Price", "Signal_Type", "Duration"]].sort_index(ascending=False)
                    st.dataframe(
                        table_df,
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Strong bottom candidates (centered)
                    long_term_bottoms = cluster_centers[cluster_centers["long_term_bottom"]]
                    if len(long_term_bottoms) > 0:
                        st.markdown("---")
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown('<h2 style="text-align: center; color: #FFD700;">🎯 Strong Bottom Candidates (Best Swing Entry Points)</h2>', unsafe_allow_html=True)
                        
                        for date in long_term_bottoms.sort_index(ascending=False).index[:10]:
                            row = long_term_bottoms.loc[date]
                            duration = (row["cluster_end"] - row["cluster_start"]).days + 1 if pd.notna(row["cluster_start"]) and pd.notna(row["cluster_end"]) else 1
                            
                            with st.expander(f"📅 {date.strftime('%Y-%m-%d')} - ${row['Adj Close']:.2f}"):
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.write(f"**Price:** ${row['Adj Close']:.2f}")
                                    st.write(f"**Severity:** {int(row['cluster_severity'])}")
                                with col2:
                                    st.write(f"**Duration:** {int(duration)} days")
                                    st.write(f"**Return Z-Score:** {row['z_return']:.2f}")
                                with col3:
                                    if pd.notna(row.get('return_30d')):
                                        st.write(f"**30d Return:** {row['return_30d']:.1f}%")
                                    if pd.notna(row.get('return_60d')):
                                        st.write(f"**60d Return:** {row['return_60d']:.1f}%")
                else:
                    st.info("ℹ️ No panic clusters detected (Severity ≥ 2). No serious multi-day capitulation events found.")
                
                # Metrics at the bottom
                st.markdown('<h2 style="text-align: center; color: #FFD700; margin-top: 20px;">📈 Summary Statistics</h2>', unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Total Clusters",
                        len(cluster_centers),
                        help="Total panic clusters detected"
                    )
                
                with col2:
                    severity_2_count = (cluster_centers['cluster_severity'] == 2).sum() if len(cluster_centers) > 0 else 0
                    st.metric(
                        "Panic Signal",
                        severity_2_count,
                        help="Moderate panic signals (less severe than Strong Bottom)"
                    )
                
                with col3:
                    strong_bottom_count = cluster_centers['long_term_bottom'].sum() if len(cluster_centers) > 0 else 0
                    st.metric(
                        "Strong Bottoms",
                        strong_bottom_count,
                        help="Long-term bottom candidates (includes Severity 3)"
                    )
                
            except ValueError as e:
                st.error(f"❌ Error: {e}")
            except Exception as e:
                st.error(f"❌ Unexpected error: {e}")
                st.exception(e)
    else:
        # Welcome message (centered, compact)
        st.markdown('<div style="text-align: center; padding: 20px;">', unsafe_allow_html=True)
        st.markdown('<h2 style="color: #FFD700;">👈 Enter a ticker symbol in Settings and click "Analyze" to get started!</h2>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
