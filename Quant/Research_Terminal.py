"""Quant Research Terminal. Run: streamlit run Quant/Research_Terminal.py"""
import sys
import re
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from research_signals import clean_ticker, normalize_prices, compute_features, backtest, event_study, breadth

st.set_page_config(page_title="Quant Research Terminal", page_icon="📉", layout="wide")
PALETTES = {
    "Light": {"paper":"#ffffff","page":"#f7f9fc","ink":"#18253b","grid":"#e7ecf3",
              "blue":"#2467cb","red":"#d85865","orange":"#e78a25","green":"#139773",
              "purple":"#845ad0","band":"rgba(216,88,101,.15)"},
    "Dark": {"paper":"#111b2b","page":"#0c1422","ink":"#eaf1fb","grid":"#2a3a50",
             "blue":"#7cafff","red":"#ff8395","orange":"#ffb866","green":"#5ce1ac",
             "purple":"#b6a0ff","band":"rgba(255,131,149,.15)"},
}

@st.cache_data(ttl=3600, show_spinner=False)
def prices(ticker, years):
    raw = yf.download(ticker, period=f"{years}y", interval="1d",
                      auto_adjust=True, progress=False, actions=False, threads=False)
    return normalize_prices(raw, ticker)

@st.cache_data(ttl=21600, show_spinner=False)
def metadata(ticker):
    info = yf.Ticker(ticker).info
    return info if isinstance(info, dict) else {}

def last(df, col):
    n = df[col].iloc[-1] if col in df and len(df) else np.nan
    return float(n) if pd.notna(n) else np.nan

def f(value, suffix=""):
    return f"{value:,.2f}{suffix}" if np.isfinite(value) else "N/A"

def compact(n):
    if n is None or not np.isfinite(n): return "N/A"
    for scale, unit in [(1e12,"T"),(1e9,"B"),(1e6,"M")]:
        if abs(n)>=scale: return "$" + f"{n/scale:.2f}{unit}"
    return "$" + f"{n:,.0f}"

def chart_style(fig, color, height=350, title=None):
    fig.update_layout(template="plotly_white" if color["paper"]=="#ffffff" else "plotly_dark",
        paper_bgcolor=color["paper"],plot_bgcolor=color["paper"],
        font=dict(color=color["ink"]),height=height,
        margin=dict(t=55 if title else 30,l=35,r=35,b=35),hovermode="x unified",
        title=title,legend=dict(orientation="h",x=0,y=1.1))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor=color["grid"])
    return fig

def lines(df, theme, specs, title, threshold=None):
    fig=go.Figure()
    for key,name,shade in specs:
        fig.add_trace(go.Scatter(x=df.index,y=df[key],name=name,
            line=dict(color=theme[shade],width=2)))
    if threshold is not None:
        fig.add_hline(y=threshold,line_dash="dash",line_color=theme["red"])
    return chart_style(fig,theme,title=title)

def price_chart(df, c):
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=.07,row_heights=[.78,.22])
    for key,name,clr in [("Close","Adjusted price","blue"),("MA50","MA50","orange"),("MA200","MA200","purple")]:
        fig.add_trace(go.Scatter(x=df.index,y=df[key],name=name,
                      line=dict(color=c[clr],width=2 if key=="Close" else 1,dash="solid" if key=="Close" else "dot")),row=1,col=1)
    for key,name,clr,shape in [("EntryCross","−σ crossing","green","triangle-up"),
                               ("ExitCross","+σ crossing","red","triangle-down"),
                               ("Capitulation","Panic cluster","orange","star")]:
        hit=df[df[key].fillna(False)]
        fig.add_trace(go.Scatter(x=hit.index,y=hit["Close"],mode="markers",name=name,
            marker=dict(color=c[clr],size=11,symbol=shape)),row=1,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=df["Drawdown52w"]*100,name="52-week drawdown",
        line=dict(color=c["red"]),fill="tozeroy",fillcolor=c["band"]),row=2,col=1)
    fig.update_yaxes(title_text="Price",row=1,col=1)
    fig.update_yaxes(title_text="DD %",row=2,col=1)
    return chart_style(fig,c,height=540)

def sharpe_chart(df,c):
    fig=make_subplots(specs=[[{"secondary_y":True}]])
    # Two traces plus tonexty reproduce the FILLED band from sharpe_chart.py.
    fig.add_trace(go.Scatter(x=df.index,y=df["SharpeLower"],
        line=dict(width=0),showlegend=False,hoverinfo="skip"),secondary_y=False)
    fig.add_trace(go.Scatter(x=df.index,y=df["SharpeUpper"],
        fill="tonexty",fillcolor=c["band"],line=dict(width=0),
        name="Filled ±σ band",hoverinfo="skip"),secondary_y=False)
    for key,desc in [("SharpeLower","−σ lower"),("SharpeUpper","+σ upper")]:
        fig.add_trace(go.Scatter(x=df.index,y=df[key],name=desc,
            line=dict(color=c["red"],dash="dash",width=1)),secondary_y=False)
    fig.add_trace(go.Scatter(x=df.index,y=df["SharpeMean"],name="Prior mean",
        line=dict(color=c["purple"],width=1,dash="dot")),secondary_y=False)
    fig.add_trace(go.Scatter(x=df.index,y=df["Sharpe"],name="Rolling Sharpe",
        line=dict(color=c["blue"],width=2.6)),secondary_y=False)
    fig.add_trace(go.Scatter(x=df.index,y=df["Close"],name="Adjusted price",
        line=dict(color=c["orange"],width=1.8)),secondary_y=True)
    fig.update_yaxes(title_text="Annualized Sharpe",secondary_y=False)
    fig.update_yaxes(title_text="Adjusted price",secondary_y=True,showgrid=False)
    return chart_style(fig,c,height=505)

def z_chart(df,c,sigma):
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=df.index,y=np.full(len(df),-sigma),
        line=dict(width=0),showlegend=False,hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=df.index,y=np.full(len(df),sigma),fill="tonexty",
        fillcolor=c["band"],line=dict(width=0),name="Threshold range",hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=df.index,y=df["SharpeZ"],name="Rolling Sharpe z",
        line=dict(color=c["blue"],width=2.5)))
    for value in [-sigma,sigma]:
        fig.add_hline(y=value,line=dict(color=c["red"],dash="dash"),
                      annotation_text=f"{value:+g}σ")
    fig.add_hline(y=0,line=dict(color=c["purple"],dash="dot"))
    return chart_style(fig,c,height=350)

def event_compare(df,threshold):
    result=[]
    for col,name in [("SharpeZ","Sharpe z"),("Return20Z","20d return z"),
                     ("MarketResidual20Z","Market residual 20d z"),
                     ("SectorResidual20Z","Sector residual 20d z")]:
        if df[col].notna().sum()<40: continue
        tmp=df.copy()
        tmp["SharpeZ"]=df[col]
        ev=event_study(tmp,entry_z=threshold)
        row={"Signal":name,"Events":len(ev)}
        for days in [20,60,120]:
            series=ev[f"{days}d forward %"].dropna() if len(ev) else pd.Series(dtype=float)
            row[f"{days}d mean %"]=series.mean() if len(series) else np.nan
            row[f"{days}d median %"]=series.median() if len(series) else np.nan
            row[f"{days}d count"]=len(series)
        result.append(row)
    return pd.DataFrame(result)

with st.sidebar:
    st.subheader("Quant / Research")
    raw_symbol=st.text_input("Ticker",value="MSFT")
    years=st.select_slider("History years",options=[3,5,8,10],value=5)
    mode=st.radio("Theme",["Light","Dark"],horizontal=True)
    st.divider()
    st.caption("SHARPE NORMALIZATION")
    sw=st.select_slider("Rolling Sharpe (days)",options=[30,45,60,90,120],value=60)
    nw=st.select_slider("Historical baseline (days)",options=[126,252,504],value=252)
    sigma=st.slider("Bands (σ)",1.0,3.0,2.0,.25)
    risk_free=st.number_input("Fixed risk-free rate (%)",min_value=0.0,max_value=20.0,value=0.0,step=.25)
    geometry=st.slider("Local geometry (days)",10,60,20,5)
    st.divider()
    st.caption("RELATIVE PERFORMANCE")
    market_input=st.text_input("Market benchmark",value="SPY")
    sector_input=st.text_input("Sector ETF (optional)",value="XLK",
        help="Choose the correct sector for the ticker. XLK is only appropriate for tech.")
    if st.button("Refresh data cache",use_container_width=True):
        prices.clear()
        metadata.clear()
        st.rerun()

c=PALETTES[mode]
st.markdown(f"""<style>.stApp {{background:{c['page']};color:{c['ink']};}}
.block-container {{max-width:1460px;padding-top:1.6rem;}}
div[data-testid="stMetric"] {{background:{c['paper']};border:1px solid {c['grid']};
border-radius:12px;padding:13px;}}
div[data-testid="stSidebar"] {{background:{c['paper']};color:{c['ink']};}}
div[data-testid="stSidebar"] > div {{background:{c['paper']};}}
[data-testid="stMarkdownContainer"] p,[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,[data-testid="stMarkdownContainer"] h3,
div[data-testid="stMetricValue"],div[data-testid="stMetricLabel"] {{color:{c['ink']};}}
button[data-baseweb="tab"] {{color:{c['ink']};}}
</style>""",unsafe_allow_html=True)

try:
    symbol=clean_ticker(raw_symbol)
    market_symbol=clean_ticker(market_input)
    sector_symbol=clean_ticker(sector_input) if sector_input.strip() else ""
except ValueError as exc:
    st.error(str(exc)); st.stop()

st.title("Quant Research Terminal")
st.caption("Falling-knife research · Rolling risk-adjusted mean reversion · Daily market data")
try:
    with st.spinner("Loading historical adjusted OHLCV…"):
        price=prices(symbol,years)
except Exception as exc:
    st.error(f"Cannot load {symbol}: {exc}")
    st.stop()

market=None
sector=None
for sym, role in [(market_symbol,"market"),(sector_symbol,"sector")]:
    if not sym or sym==symbol or (role=="sector" and sym==market_symbol): continue
    try:
        ref=prices(sym,years)
        if role=="market": market=ref
        else: sector=ref
    except Exception as exc:
        st.warning(f"Benchmark {sym} unavailable; relative signal omitted: {exc}")

df=compute_features(price,market,sector,sw,nw,geometry,risk_free/100,sigma)
z=last(df,"SharpeZ")
state=("Negative extreme" if z<=-sigma else
       "Positive extreme" if z>=sigma else
       "Inside band" if np.isfinite(z) else "Insufficient baseline")
st.markdown(f"**{symbol}** · Prices as of **{df.index[-1]:%b %d, %Y}** · **{state}**")
st.caption("Yahoo Finance adjusted daily observations, not streaming quotes. Signals formed at the daily close.")
top=st.columns(5)
top[0].metric("Adjusted close","$"+f(last(df,"Close")))
top[1].metric("Rolling Sharpe",f(last(df,"Sharpe")))
top[2].metric("Sharpe z-score",f(z,"σ"))
top[3].metric("52w drawdown",f(last(df,"Drawdown52w")*100,"%"))
top[4].metric("Market residual z",f(last(df,"MarketResidual20Z"),"σ"))

tab=st.tabs(["Overview","Sharpe & bands","Signals","Backtest","Valuation",
             "Watchlist & breadth","Notes / methodology"])
with tab[0]:
    st.subheader("Price action, reversal crossings and drawdown")
    st.plotly_chart(price_chart(df,c),use_container_width=True)
    cols=st.columns(4)
    cols[0].metric("Panic severity 0–3",str(int(last(df,"PanicSeverity"))))
    cols[1].metric("Return z",f(last(df,"ReturnZ"),"σ"))
    cols[2].metric("Volume z",f(last(df,"VolumeZ"),"σ"))
    cols[3].metric("Trend geometry",str(df["GeometryRegime"].iloc[-1]))
    st.caption("The triangles are first crosses of the ±σ thresholds, not trade executions. Stars mark trailing panic clusters.")
    with st.expander("Panic condition details"):
        q=df.iloc[-1]
        st.write({"Return shock":bool(q["ReturnPanic"]),"Volume shock":bool(q["VolumePanic"]),
                  "Volatility spike":bool(q["VolatilityPanic"]),
                  "At least 2 panic days in trailing 6":bool(q["PanicDays6"]>=2),
                  "Stabilizing (heuristic)":bool(q["Stabilizing"])})
with tab[1]:
    st.subheader("Rolling Sharpe, filled ±σ band and adjusted price")
    st.caption("Based on the preferred white + shaded channel format from your existing sharpe_chart.py; dark mode uses the same fill.")
    st.plotly_chart(sharpe_chart(df,c),use_container_width=True)
    st.subheader("Sharpe z-score and entry / exit zones")
    st.plotly_chart(z_chart(df,c,sigma),use_container_width=True)
    a,b,d=st.columns(3)
    a.metric("Prior Sharpe mean",f(last(df,"SharpeMean")))
    b.metric("Prior Sharpe standard deviation",f(last(df,"SharpeStd")))
    d.metric("Sharpe z-score",f(z,"σ"))
    st.info("The ±2σ zones apply to a rolling Sharpe distribution, NOT directly to price. Overlapping windows make Gaussian tail probabilities unreliable.")
with tab[2]:
    st.subheader("Capitulation, relative return and geometry")
    a,b=st.columns(2)
    with a:
        st.plotly_chart(lines(df,c,[("ReturnZ","Daily return z","red"),("VolumeZ","Volume z","orange")],
                              "Return and volume surprises",0),use_container_width=True)
        st.plotly_chart(lines(df,c,[("Vol20","20-day volatility","red"),("Vol100","100-day volatility","blue")],
                              "Annualized realized volatility"),use_container_width=True)
    with b:
        st.plotly_chart(lines(df,c,[("Return20Z","Raw return z","orange"),
                     ("MarketResidual20Z","Market residual z","blue"),
                     ("SectorResidual20Z","Sector residual z","purple")],
                     "20-day return / residual z-scores",-2),use_container_width=True)
        st.plotly_chart(lines(df,c,[("Velocity","Velocity","blue"),("Curvature","Curvature","green")],
                              "Trailing quadratic trend geometry",0),use_container_width=True)
    st.dataframe(pd.DataFrame([{"Metric":label,"Latest":f(last(df,key),unit)} for label,key,unit in [
        ("20d return z","Return20Z","σ"),("20d market residual z","MarketResidual20Z","σ"),
        ("20d sector residual z","SectorResidual20Z","σ"),("RSI 14","RSI14",""),
        ("Instantaneous Sharpe","InstantaneousSharpe",""),
        ("Volume z","VolumeZ","σ")]]),hide_index=True,use_container_width=True)
    st.caption("Residuals remove lagged-beta benchmark effects; overlapping daily 20-day return observations are correlated.")
with tab[3]:
    st.subheader("Sharpe z-score trading rule")
    st.caption("Long-only, enter after first negative threshold crossing, EXIT on recovery or max hold. Orders filled at the next adjusted OPEN, not on the signal bar.")
    a,b,d,e=st.columns(4)
    entry=a.number_input("Buy if z crosses below",min_value=-4.0,max_value=-.25,value=-float(sigma),step=.25)
    exit_z=b.number_input("Sell if z reaches",min_value=-1.0,max_value=4.0,value=float(sigma),step=.25)
    hold=d.number_input("Maximum holding days",min_value=5,max_value=504,value=120,step=5)
    costs=e.number_input("Total round-trip cost (bps)",min_value=0.0,max_value=500.0,value=20.0,step=5.0)
    start=st.date_input("Evaluation start (after historical feature warmup)",
             value=df.index[len(df)//2].date(),
             min_value=df.index[0].date(),max_value=df.index[-1].date())
    evaluation=df.loc[df.index.date>=start].copy()
    trades=backtest(evaluation,entry_z=entry,exit_z=exit_z,max_hold=int(hold),
                    cost_bps_per_side=costs/2)
    if len(trades):
        total=(1+trades["Return %"]/100).prod()-1
        a,b,d,e=st.columns(4)
        a.metric("Completed trades",str(len(trades)))
        b.metric("Win rate",f((trades["Return %"]>0).mean()*100,"%"))
        d.metric("Median return per trade",f(trades["Return %"].median(),"%"))
        e.metric("Compounded closed trades",f(total*100,"%"))
        realized=(1+trades["Return %"]/100).cumprod()
        fig=go.Figure(go.Scatter(x=pd.to_datetime(trades["Exit date"]),y=realized,
            mode="lines+markers",line=dict(color=c["green"],width=2),name="Realized capital"))
        st.plotly_chart(chart_style(fig,c,title="Closed-trade capital index"),use_container_width=True)
        st.dataframe(trades.sort_values("Entry date",ascending=False),
                     hide_index=True,use_container_width=True)
        st.download_button("Download trades CSV",trades.to_csv(index=False),
                           f"{symbol}_trades.csv","text/csv")
    else:
        st.warning("No closed trades in this evaluation sample with the current parameters.")
    st.subheader("Comparison: does Sharpe add predictive information?")
    st.caption("Event study (next-open to future close) versus simpler return and residual z-scores. Descriptive, NOT alpha proof.")
    comp=event_compare(evaluation,entry)
    if len(comp):
        st.dataframe(comp.round(2),hide_index=True,use_container_width=True)
        st.download_button("Download event study",comp.to_csv(index=False),
                           f"{symbol}_events.csv","text/csv")
    st.caption("No overlapping positions; simplified next-open fills and costs. Closed-trade compounding is NOT CAGR or daily portfolio performance. The modern fair-value input is never backfilled into historical trades.")
with tab[4]:
    st.subheader("Manual fundamental thesis + margin of safety")
    left,right=st.columns(2)
    with left:
        fair=st.number_input("Independently calculated fair value per share",min_value=0.0,value=0.0,step=5.0,key="fv_"+symbol)
        safety=st.slider("Required margin of safety (%)",0,60,20,5)
        if fair>0:
            st.metric("Upside to fair value",f((fair/last(df,"Close")-1)*100,"%"))
            st.metric("Discount to fair value",f((1-last(df,"Close")/fair)*100,"%"))
            st.metric("Below valuation entry threshold","Yes" if last(df,"Close")<=fair*(1-safety/100) else "No")
        else:
            st.info("Enter your researched fair value to compare current prices.")
    with right:
        checks=[st.checkbox("Operating thesis intact"),
                st.checkbox("Latest earnings and guidance reviewed"),
                st.checkbox("Balance sheet and dilution reviewed"),
                st.checkbox("Selloff catalyst explained")]
        st.text_area("Thesis notes (session only)",height=130,key="notes_"+symbol)
        if fair>0 and all(checks) and last(df,"Close")<=fair*(1-safety/100):
            st.success("Your own valuation and due-diligence checks pass. This is not a recommendation.")
    st.divider()
    if st.button("Load Yahoo company fundamentals"):
        st.session_state["info_"+symbol]=True
    if st.session_state.get("info_"+symbol):
        try:
            info=metadata(symbol)
            st.markdown("**"+str(info.get("longName",symbol))+"** · "+str(info.get("sector","Sector unavailable")))
            cards=st.columns(4)
            for cell,label,val in zip(cards,["Market cap","Forward P/E","EV/EBITDA","Free cash flow"],[
                compact(info.get("marketCap")),f(info.get("forwardPE") or np.nan),
                f(info.get("enterpriseToEbitda") or np.nan),compact(info.get("freeCashflow"))]):
                cell.metric(label,val)
            cards=st.columns(4)
            for cell,label,val in zip(cards,["Net margin","ROE","Revenue growth","Debt"],[
                f((info.get("profitMargins") or np.nan)*100,"%"),
                f((info.get("returnOnEquity") or np.nan)*100,"%"),
                f((info.get("revenueGrowth") or np.nan)*100,"%"),compact(info.get("totalDebt"))]):
                cell.metric(label,val)
            st.caption("Yahoo Finance metadata may be stale, missing or inconsistent with your estimates.")
        except Exception as exc:
            st.warning(f"Company snapshot unavailable: {exc}")
with tab[5]:
    st.subheader("Watchlist scanner & advance–decline volatility")
    watch=st.text_area("Up to 15 tickers (comma, space or newline)",
          value="MSFT, GOOGL, AMZN, META, INTC, PYPL, SPY",height=75)
    if st.button("Run watchlist scan",type="primary"):
        symbols=list(dict.fromkeys(s for s in re.split(r"[\s,;]+",watch.strip().upper()) if s))[:15]
        results=[]
        matrix={}
        with st.spinner("Calculating watchlist signals…"):
            for s in symbols:
                try:
                    s=clean_ticker(s)
                    values=prices(s,years)
                    matrix[s]=values["Close"]
                    ft=compute_features(values,market,sector,sw,nw,geometry,risk_free/100,sigma)
                    results.append({"Ticker":s,"Close":last(ft,"Close"),"Sharpe z":last(ft,"SharpeZ"),
                        "20d return z":last(ft,"Return20Z"),
                        "52w DD %":last(ft,"Drawdown52w")*100,
                        "Volume z":last(ft,"VolumeZ"),
                        "Panic severity":int(last(ft,"PanicSeverity")),
                        "Negative extreme":bool(last(ft,"SharpeZ")<=-sigma)})
                except Exception as exc:
                    results.append({"Ticker":s,"Error":str(exc)[:130]})
        st.session_state["watch_table"]=pd.DataFrame(results)
        st.session_state["watch_prices"]=pd.DataFrame(matrix)
    if "watch_table" in st.session_state:
        st.dataframe(st.session_state["watch_table"].round(2),hide_index=True,use_container_width=True)
        st.download_button("Export scan CSV",st.session_state["watch_table"].to_csv(index=False),
                           "watchlist_scan.csv","text/csv")
        matrix=st.session_state["watch_prices"]
        if len(matrix.columns)>=2:
            b=breadth(matrix)
            st.plotly_chart(lines(b,c,[("AD Ratio","Advance–decline ratio","blue")],
                                  "Breadth of selected watchlist",0),use_container_width=True)
            st.plotly_chart(lines(b,c,[("BreadthVolZ","Watchlist breadth volatility z","red")],
                                  "Unusual dispersion / breadth volatility",2),use_container_width=True)
            st.caption("Watchlist breadth is NOT market-wide breadth. The static universe can introduce survivorship and selection bias.")
    else:
        st.info("Run a scan to compare tickers.")
with tab[6]:
    st.subheader("How the model works")
    st.markdown("""
**Sharpe:** annualized excess mean return / standard deviation over a user-selected rolling window, with a fixed selected risk-free rate.

**Sharpe z-score:** current Sharpe standardized versus the PREVIOUS rolling observations. The filled red ±σ bands show prior mean ± selected sigma × prior standard deviation.

**Panic:** unusual daily returns, unusual daily volume and/or large short-term realized volatility, with clusters counting PAST days only. The existing Falling_Knife_Web.py script used a centered window that can introduce look-ahead in historical tests.

**Market and sector residual:** stock return minus benchmark return times beta estimated using historical data only; normalized over trailing windows.

**Local trend geometry:** trailing quadratic fit to log price; slope measured at end of window and curvature indicates local acceleration, not a guaranteed bottom.

**Validation:** compare Sharpe signals with simple return z-scores and sector-residual z-scores; use next-session execution, realistic cost assumptions and unseen data. Overlapping events and repeated parameters can overfit.
""")
    st.warning("An unusual risk-adjusted selloff may reflect permanent fundamental deterioration. Always reassess cash flows, competitive position and valuation.")
    st.markdown("Adapted from sharpe_chart.py, Falling_Knife_Web.py, Quant_dashboard.py, trend_geometry.py and advance_decline_vol.py in the original Quant folder.")
