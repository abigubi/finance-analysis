import streamlit as st
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

# Minimal CSS - let Replit's native theme handle most styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #FFD700;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


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
    raise KeyError("Could not find an adjusted or close price column.")


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
    """Calculate panic signals with clustering and severity scoring for swing-worthy bottoms."""
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
    
    # PANIC CLUSTERING: Count panics within 5 trading days
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
    
    # Severity 2: Multiple panic types OR clustering (2+ days of panic)
    df.loc[
        ((df["raw_panic_score"] >= 2) | (df["panic_cluster_count"] >= 2)) &
        (df["panic_severity"] == 0),
        "panic_severity"
    ] = 2
    
    # Severity 3: All three conditions + clustering (full capitulation)
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
    
    # Only show severity >= 2 (serious panics and capitulation)
    df["panic_signal"] = df["panic_severity"] >= 2
    
    # Calculate forward returns for backtesting
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
        
        # Find cluster center (date with lowest price)
        cluster_center = cluster_df.loc[cluster_df["Adj Close"].idxmin()]
        center_date = cluster_center.name
        
        # Get max severity in cluster
        max_severity = cluster_df["panic_severity"].max()
        is_long_term_bottom = cluster_df["long_term_bottom"].any()
        
        # Mark all dates in cluster
        df.loc[cluster_dates, "cluster_id"] = cluster_num
        df.loc[center_date, "is_cluster_center"] = True
        df.loc[center_date, "cluster_severity"] = max_severity
        df.loc[center_date, "cluster_start"] = cluster_dates.min()
        df.loc[center_date, "cluster_end"] = cluster_dates.max()
        
        # If it's a long-term bottom, mark the center
        if is_long_term_bottom:
            df.loc[center_date, "long_term_bottom"] = True
        
        cluster_num += 1
    
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
        },
        "Black Line": {
            "price_color": "#000000",
            "strong_bottom_color": "#FFFFFF",
            "strong_bottom_border": "#FF0000",
            "severity3_color": "#FF0000",
            "severity3_border": "#FFFFFF",
            "severity2_color": "#FFA500",
            "severity2_border": "#FFFFFF",
            "bg_color": "rgba(0,0,0,0)",
            "text_color": "#000000",
            "grid_color": "rgba(0, 0, 0, 0.3)"
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
        dragmode='pan',  # Default to pan instead of zoom
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
            font=dict(color=theme["text_color"], size=12)
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
    
    # Title with color theme selector on the same row - theme on left, title centered
    title_col1, title_col2, title_col3 = st.columns([2, 6, 2])
    with title_col1:
        st.markdown("<br>", unsafe_allow_html=True)
        color_theme = st.selectbox(
            "🎨 Color Theme",
            ["Dark Mode (Default)", "Blue Ocean", "Green Energy", "Purple Night", "White Line", "Black Line"],
            index=0,
            help="Choose a color theme for the chart",
            key="theme_selector"
        )
    with title_col2:
        st.markdown('<h1 class="main-header" style="text-align: center; margin-top: -20px; margin-bottom: 0rem;">📉 Falling Knife Detector</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header" style="text-align: center; margin-top: 0rem; margin-bottom: 1rem;">Long-Term Bottom Detection & Panic Signal Analysis</p>', unsafe_allow_html=True)
    with title_col3:
        st.empty()  # Empty column for balance
    
    theme = get_color_theme(color_theme)
    
    with st.expander("⚙️ Settings", expanded=True):
        col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
        with col1:
            ticker = st.text_input("Ticker Symbol", value="AAPL", placeholder="Enter ticker (e.g., AAPL, TSLA, MSFT)", help="Enter any stock ticker symbol").upper().strip()
        with col2:
            start_date = st.date_input("Start Date", value=datetime(2020, 1, 1), max_value=datetime.today(), help="Start date for historical data")
        with col3:
            end_date = st.date_input("End Date", value=datetime.today(), max_value=datetime.today(), help="End date for historical data")
        with col4:
            st.markdown("<br>", unsafe_allow_html=True)
            analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)
        start_str = start_date.strftime("%Y-%m-%d")
    
    ticker_changed = ticker != st.session_state.current_ticker
    start_changed = start_str != st.session_state.start_str
    
    if st.session_state.df_data is not None and st.session_state.current_ticker == ticker and st.session_state.start_str == start_str:
        df = st.session_state.df_data.copy()
        if end_date < datetime.today().date():
            df = df[df.index.date <= end_date]
        cluster_centers = df[df["is_cluster_center"]].copy()
        
        # Chart (centered)
        fig = create_plotly_chart(ticker, df, theme)
        st.plotly_chart(fig, use_container_width=True)
        
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
            
            # Strong Bottom Candidates
            long_term_bottoms = cluster_centers[cluster_centers["long_term_bottom"]]
            if len(long_term_bottoms) > 0:
                st.markdown('<h2 style="text-align: center; color: #FFD700; margin-top: 20px;">🎯 Strong Bottom Candidates (Best Swing Entry Points)</h2>', unsafe_allow_html=True)
                
                for date in long_term_bottoms.sort_index(ascending=False).index[:10]:
                    row = long_term_bottoms.loc[date]
                    with st.expander(f"📅 {date.strftime('%Y-%m-%d')} - ${row['Adj Close']:.2f}", expanded=False):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.write(f"**Price:** ${row['Adj Close']:.2f}")
                            if pd.notna(row.get('cluster_start')) and pd.notna(row.get('cluster_end')):
                                duration = (row['cluster_end'] - row['cluster_start']).days + 1
                                st.write(f"**Duration:** {int(duration)} days")
                            st.write(f"**Return Z-Score:** {row['z_return']:.2f}")
                        with col3:
                            if pd.notna(row.get('return_30d')):
                                st.write(f"**30d Return:** {row['return_30d']:.1f}%")
                            if pd.notna(row.get('return_60d')):
                                st.write(f"**60d Return:** {row['return_60d']:.1f}%")
                            if pd.notna(row.get('return_120d')):
                                st.write(f"**120d Return:** {row['return_120d']:.1f}%")
        
        # Summary metrics
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
    
    elif analyze_btn or (ticker_changed and ticker) or (start_changed and ticker):
        try:
            # Download and process data
            prices = download_price_data(ticker, start_str)
            price_series = select_price_series(prices, ticker)
            volume_series = select_volume_series(prices, ticker)
            
            df = pd.DataFrame({
                "Adj Close": price_series,
                "Volume": volume_series
            })
            
            # Calculate signals using the correct logic
            df = calculate_panic_signals(df)
            df = identify_panic_clusters(df)
            
            # Filter by end date if needed
            if end_date < datetime.today().date():
                df = df[df.index.date <= end_date]
            
            # Save to session state
            st.session_state.df_data = df
            st.session_state.current_ticker = ticker
            st.session_state.start_str = start_str
            
            cluster_centers = df[df["is_cluster_center"]].copy()
            
            # Chart (centered)
            fig = create_plotly_chart(ticker, df, theme)
            st.plotly_chart(fig, use_container_width=True)
            
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
                
                # Strong Bottom Candidates
                long_term_bottoms = cluster_centers[cluster_centers["long_term_bottom"]]
                if len(long_term_bottoms) > 0:
                    st.markdown('<h2 style="text-align: center; color: #FFD700; margin-top: 20px;">🎯 Strong Bottom Candidates (Best Swing Entry Points)</h2>', unsafe_allow_html=True)
                    
                    for date in long_term_bottoms.sort_index(ascending=False).index[:10]:
                        row = long_term_bottoms.loc[date]
                        with st.expander(f"📅 {date.strftime('%Y-%m-%d')} - ${row['Adj Close']:.2f}", expanded=False):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.write(f"**Price:** ${row['Adj Close']:.2f}")
                                if pd.notna(row.get('cluster_start')) and pd.notna(row.get('cluster_end')):
                                    duration = (row['cluster_end'] - row['cluster_start']).days + 1
                                    st.write(f"**Duration:** {int(duration)} days")
                                st.write(f"**Return Z-Score:** {row['z_return']:.2f}")
                            with col3:
                                if pd.notna(row.get('return_30d')):
                                    st.write(f"**30d Return:** {row['return_30d']:.1f}%")
                                if pd.notna(row.get('return_60d')):
                                    st.write(f"**60d Return:** {row['return_60d']:.1f}%")
                                if pd.notna(row.get('return_120d')):
                                    st.write(f"**120d Return:** {row['return_120d']:.1f}%")
            
            # Summary metrics
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
        
        except Exception as e:
            st.error(f"Error processing {ticker}: {str(e)}")
    
    else:
        st.markdown('<div style="text-align: center; padding: 20px;">', unsafe_allow_html=True)
        st.markdown('<h2 style="color: #FFD700;">👈 Enter a ticker symbol in Settings and click "Analyze" to get started!</h2>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Bulk Scanner Section
    st.markdown("---")
    st.markdown('<h2 style="text-align: center; color: #FFD700; margin-top: 30px;">🔍 Bulk Ticker Scanner</h2>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; margin-bottom: 20px;">Scan multiple tickers for recent panic signals and strong bottoms</p>', unsafe_allow_html=True)
    
    with st.expander("📊 Scanner Settings", expanded=False):
        col1, col2 = st.columns([3, 1])
        with col1:
            tickers_input = st.text_area(
                "Enter ticker symbols (one per line, or comma/space separated)",
                placeholder="AAPL\nTSLA\nMSFT\nGOOGL\nNVDA\n...",
                height=150,
                help="Enter multiple ticker symbols to scan"
            )
        with col2:
            scanner_days = st.number_input(
                "Days to look back",
                min_value=1,
                max_value=365,
                value=30,
                help="Number of days to scan for signals"
            )
            scan_button = st.button("🚀 Scan All Tickers", type="primary", use_container_width=True)
    
    # Initialize scanner results in session state
    if 'scanner_results' not in st.session_state:
        st.session_state.scanner_results = None
    
    if scan_button and tickers_input.strip():
        # Parse tickers
        tickers_list = []
        for line in tickers_input.strip().split('\n'):
            if ',' in line:
                tickers_list.extend([t.strip().upper() for t in line.split(',') if t.strip()])
            elif ' ' in line:
                tickers_list.extend([t.strip().upper() for t in line.split() if t.strip()])
            elif line.strip():
                tickers_list.append(line.strip().upper())
        
        tickers_list = list(dict.fromkeys([t for t in tickers_list if t]))  # Remove duplicates
        
        if tickers_list:
            progress_bar = st.progress(0)
            status_text = st.empty()
            results = []
            
            for i, ticker in enumerate(tickers_list):
                status_text.text(f"Scanning {i+1}/{len(tickers_list)}: {ticker}...")
                progress_bar.progress((i + 1) / len(tickers_list))
                
                result = scan_ticker_for_scanner(ticker, scanner_days)
                results.append(result)
            
            st.session_state.scanner_results = results
            status_text.empty()
            progress_bar.empty()
    
    # Display scanner results
    if st.session_state.scanner_results is not None:
        results = st.session_state.scanner_results
        
        # Filter results
        signals_only = [r for r in results if r["signals_found"]]
        errors = [r for r in results if r["status"] == "ERROR"]
        clean = [r for r in results if r["status"] == "OK" and not r["signals_found"]]
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Scanned", len(results))
        with col2:
            st.metric("With Signals", len(signals_only), delta=f"{len(signals_only)/len(results)*100:.1f}%")
        with col3:
            st.metric("Strong Bottoms", sum(r["strong_bottoms"] for r in signals_only))
        with col4:
            st.metric("Panic Signals", sum(r["panic_signals"] for r in signals_only))
        
        # Filter options
        st.markdown("### Filter Results")
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        with filter_col1:
            show_signals_only = st.checkbox("Show only tickers with signals", value=True)
        with filter_col2:
            show_strong_bottoms = st.checkbox("Show only Strong Bottoms", value=False)
        with filter_col3:
            show_errors = st.checkbox("Show errors", value=False)
        
        # Filter results based on options
        filtered_results = []
        if show_signals_only:
            filtered_results = signals_only.copy()
            if show_strong_bottoms:
                filtered_results = [r for r in filtered_results if r["strong_bottoms"] > 0]
        else:
            filtered_results = results.copy()
            if show_strong_bottoms:
                filtered_results = [r for r in filtered_results if r["strong_bottoms"] > 0]
        
        if show_errors:
            filtered_results.extend(errors)
        
        # Create display dataframe
        if filtered_results:
            display_data = []
            for result in filtered_results:
                if result["signals_found"]:
                    for detail in result["details"]:
                        display_data.append({
                            "Ticker": result["ticker"],
                            "Date": detail["date"],
                            "Price": detail["price"],
                            "Signal Type": detail["signal_type"],
                            "Severity": detail["severity"],
                            "Duration": detail["duration"],
                            "Strong Bottom": "Yes" if "STRONG BOTTOM" in detail["signal_type"] else "No"
                        })
                elif result["status"] == "ERROR":
                    display_data.append({
                        "Ticker": result["ticker"],
                        "Date": "N/A",
                        "Price": "N/A",
                        "Signal Type": f"ERROR: {result.get('error', 'Unknown')}",
                        "Severity": 0,
                        "Duration": "N/A",
                        "Strong Bottom": "N/A"
                    })
            
            if display_data:
                df_display = pd.DataFrame(display_data)
                
                # Add search/filter
                search_term = st.text_input("🔍 Search ticker or signal type", "")
                if search_term:
                    df_display = df_display[
                        df_display["Ticker"].str.contains(search_term, case=False, na=False) |
                        df_display["Signal Type"].str.contains(search_term, case=False, na=False)
                    ]
                
                # Display table
                st.markdown("### Scan Results")
                st.dataframe(
                    df_display,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Ticker": st.column_config.TextColumn("Ticker", width="small"),
                        "Date": st.column_config.TextColumn("Date", width="small"),
                        "Price": st.column_config.TextColumn("Price", width="small"),
                        "Signal Type": st.column_config.TextColumn("Signal Type", width="medium"),
                        "Severity": st.column_config.NumberColumn("Severity", width="small"),
                        "Duration": st.column_config.TextColumn("Duration", width="small"),
                        "Strong Bottom": st.column_config.TextColumn("Strong Bottom", width="small")
                    }
                )
                
                # Download button
                csv = df_display.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name=f"falling_knife_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No results match the current filters.")
        else:
            st.info("No results to display. Adjust your filters or run a new scan.")
        
        if errors and not show_errors:
            st.warning(f"⚠️ {len(errors)} ticker(s) had errors. Enable 'Show errors' to view them.")


def scan_ticker_for_scanner(ticker: str, lookback_days: int) -> dict:
    """Scan a single ticker for signals in the last N days - for bulk scanner."""
    try:
        # Calculate start date (need extra days for rolling calculations)
        end_date = datetime.today()
        start_date = end_date - timedelta(days=lookback_days + 100)  # Extra buffer for calculations
        start_str = start_date.strftime("%Y-%m-%d")
        
        # Download data
        prices = download_price_data(ticker, start_str)
        price_series = select_price_series(prices, ticker)
        volume_series = select_volume_series(prices, ticker)
        
        df = pd.DataFrame({
            "Adj Close": price_series,
            "Volume": volume_series
        })
        
        # Calculate signals using Falling_Knife.py logic
        df = calculate_panic_signals(df)
        df = identify_panic_clusters(df)
        
        # Filter to lookback period
        cutoff_date = end_date - timedelta(days=lookback_days)
        df_recent = df[df.index >= cutoff_date]
        
        # Find recent cluster centers
        recent_clusters = df_recent[df_recent["is_cluster_center"]].copy()
        
        if len(recent_clusters) == 0:
            return {
                "ticker": ticker,
                "status": "OK",
                "signals_found": False,
                "signal_count": 0,
                "strong_bottoms": 0,
                "panic_signals": 0,
                "details": []
            }
        
        # Prepare details
        details = []
        for date in recent_clusters.sort_index(ascending=False).index:
            row = recent_clusters.loc[date]
            signal_type = "STRONG BOTTOM *" if row["long_term_bottom"] else f"Panic Signal (Severity {int(row['cluster_severity'])})"
            duration = "N/A"
            if pd.notna(row.get('cluster_start')) and pd.notna(row.get('cluster_end')):
                duration = f"{(row['cluster_end'] - row['cluster_start']).days + 1} days"
            
            details.append({
                "date": date.strftime("%Y-%m-%d"),
                "price": f"${row['Adj Close']:.2f}",
                "signal_type": signal_type,
                "severity": int(row['cluster_severity']),
                "duration": duration
            })
        
        strong_bottoms = recent_clusters['long_term_bottom'].sum()
        panic_signals = (recent_clusters['cluster_severity'] == 2).sum()
        
        return {
            "ticker": ticker,
            "status": "OK",
            "signals_found": True,
            "signal_count": len(recent_clusters),
            "strong_bottoms": int(strong_bottoms),
            "panic_signals": int(panic_signals),
            "details": details
        }
    
    except Exception as e:
        return {
            "ticker": ticker,
            "status": "ERROR",
            "error": str(e),
            "signals_found": False,
            "signal_count": 0,
            "strong_bottoms": 0,
            "panic_signals": 0,
            "details": []
        }


if __name__ == "__main__":
    main()
