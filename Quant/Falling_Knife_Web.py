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
    /* Only fix specific elements, let Replit theme handle the rest */
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
    /* Hide Streamlit menu and footer */
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


def calculate_panic_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate panic signals based on volume and price drops."""
    df = df.copy()
    
    # Calculate returns
    df["returns"] = df["Adj Close"].pct_change()
    df["log_returns"] = np.log(df["Adj Close"] / df["Adj Close"].shift(1))
    
    # Volume metrics
    if "Volume" in df.columns:
        df["volume_ma"] = df["Volume"].rolling(window=20).mean()
        df["volume_ratio"] = df["Volume"] / df["volume_ma"]
    else:
        df["volume_ratio"] = 1.0
    
    # Rolling statistics
    window = 60
    df["rolling_mean"] = df["Adj Close"].rolling(window=window).mean()
    df["rolling_std"] = df["Adj Close"].rolling(window=window).std()
    df["z_score"] = (df["Adj Close"] - df["rolling_mean"]) / df["rolling_std"]
    
    # Panic conditions
    df["panic_signal"] = (
        (df["returns"] < -0.02) &  # 2% drop
        (df["volume_ratio"] > 1.5) &  # High volume
        (df["z_score"] < -1.0)  # Below mean
    )
    
    return df


def identify_panic_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """Identify panic clusters and long-term bottoms."""
    df = df.copy()
    
    # Initialize cluster columns
    df["cluster_id"] = None
    df["is_cluster_center"] = False
    df["cluster_severity"] = 0
    df["long_term_bottom"] = False
    df["cluster_start"] = None
    df["cluster_end"] = None
    
    panic_days = df[df["panic_signal"]].index
    
    if len(panic_days) == 0:
        return df
    
    # Group consecutive panic days into clusters
    clusters = []
    current_cluster = [panic_days[0]]
    
    for i in range(1, len(panic_days)):
        if (panic_days[i] - panic_days[i-1]).days <= 5:
            current_cluster.append(panic_days[i])
        else:
            clusters.append(current_cluster)
            current_cluster = [panic_days[i]]
    clusters.append(current_cluster)
    
    # Process each cluster
    for cluster_id, cluster_days in enumerate(clusters):
        cluster_start = cluster_days[0]
        cluster_end = cluster_days[-1]
        
        # Find the lowest point in the cluster
        cluster_data = df.loc[cluster_days]
        lowest_idx = cluster_data["Adj Close"].idxmin()
        
        # Calculate cluster metrics
        price_drop = (df.loc[cluster_start, "Adj Close"] - df.loc[lowest_idx, "Adj Close"]) / df.loc[cluster_start, "Adj Close"]
        duration = (cluster_end - cluster_start).days + 1
        volume_spike = cluster_data["volume_ratio"].mean() if "volume_ratio" in cluster_data.columns else 1.0
        
        # Determine severity
        if price_drop > 0.15 and duration >= 10:
            severity = 3
        elif price_drop > 0.10 and duration >= 5:
            severity = 2
        else:
            severity = 1
        
        # Mark cluster center
        df.loc[lowest_idx, "is_cluster_center"] = True
        df.loc[lowest_idx, "cluster_id"] = cluster_id
        df.loc[lowest_idx, "cluster_severity"] = severity
        df.loc[lowest_idx, "cluster_start"] = cluster_start
        df.loc[lowest_idx, "cluster_end"] = cluster_end
        
        # Long-term bottom detection (Severity 3 with additional conditions)
        if severity == 3:
            # Check if this is a significant low relative to surrounding period
            lookback = 252  # 1 year
            lookahead = 60  # 3 months
            
            start_idx = max(0, df.index.get_loc(lowest_idx) - lookback)
            end_idx = min(len(df), df.index.get_loc(lowest_idx) + lookahead)
            
            surrounding_prices = df.iloc[start_idx:end_idx]["Adj Close"]
            if len(surrounding_prices) > 0:
                percentile = (surrounding_prices < df.loc[lowest_idx, "Adj Close"]).sum() / len(surrounding_prices)
                if percentile < 0.1:  # Bottom 10% of prices in the period
                    df.loc[lowest_idx, "long_term_bottom"] = True
        
        # Calculate future returns for long-term bottoms
        if df.loc[lowest_idx, "long_term_bottom"]:
            for days in [30, 60, 90]:
                future_idx = df.index.get_loc(lowest_idx) + days
                if future_idx < len(df):
                    future_return = (df.iloc[future_idx]["Adj Close"] - df.loc[lowest_idx, "Adj Close"]) / df.loc[lowest_idx, "Adj Close"] * 100
                    df.loc[lowest_idx, f"return_{days}d"] = future_return
    
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
    
    # Top bar with theme selector
    top_col1, top_col2 = st.columns([1, 10])
    with top_col1:
        color_theme = st.selectbox(
            "🎨 Color Theme",
            ["Dark Mode (Default)", "Blue Ocean", "Green Energy", "Purple Night", "White Line", "Black Line"],
            index=0,
            help="Choose a color theme for the chart",
            key="theme_selector"
        )
    
    theme = get_color_theme(color_theme)
    
    # Title
    st.markdown('<h1 class="main-header" style="text-align: center; margin-top: -20px; margin-bottom: 0.5rem;">📉 Falling Knife Detector</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header" style="text-align: center; margin-bottom: 1rem;">Long-Term Bottom Detection & Panic Signal Analysis</p>', unsafe_allow_html=True)
    
    with st.expander("⚙️ Settings", expanded=True):
        col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
        with col1:
            ticker = st.text_input("Ticker Symbol", value="AAPL", placeholder="Enter ticker (e.g., AAPL, TSLA, MSFT)", help="Enter any stock ticker symbol").upper().strip()
        with col2:
            start_date = st.date_input("Start Date", value=datetime(2015, 1, 1), max_value=datetime.today(), help="Start date for historical data")
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
                            if pd.notna(row.get('return_90d')):
                                st.write(f"**90d Return:** {row['return_90d']:.1f}%")
        
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
            
            df = pd.DataFrame({"Adj Close": price_series})
            if "Volume" in prices.columns:
                volume_series = prices["Volume"] if not isinstance(prices.columns, pd.MultiIndex) else prices.xs("Volume", axis=1, level=0)
                if isinstance(volume_series, pd.DataFrame):
                    df["Volume"] = volume_series.iloc[:, 0]
                else:
                    df["Volume"] = volume_series
            
            # Calculate signals
            df = calculate_panic_signals(df)
            df = identify_panic_clusters(df)
            
            # Add return z-score
            df["z_return"] = (df["returns"] - df["returns"].rolling(60).mean()) / df["returns"].rolling(60).std()
            
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
                                if pd.notna(row.get('return_90d')):
                                    st.write(f"**90d Return:** {row['return_90d']:.1f}%")
            
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


if __name__ == "__main__":
    main()
