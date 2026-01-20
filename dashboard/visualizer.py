import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def plot_pnl_distribution(portfolio_df):
    if portfolio_df.empty:
        return go.Figure()
    
    # Use Market Value for now since PnL % is not fully tracked
    fig = go.Figure(data=[go.Bar(
        x=portfolio_df['symbol'],
        y=portfolio_df['market_value'],
        marker_color='#00FF00'
    )])
    
    fig.update_layout(
        title="当前持仓市值分布 (USD)",
        yaxis_title="美元价值",
        template="plotly_dark",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

def plot_market_scatter(market_df):
    if market_df.empty: return go.Figure()
    
    fig = px.scatter(
        market_df,
        x="market_cap",
        y="volume",
        size="market_cap", # Bubble size by Market Cap
        color="symbol",
        hover_name="symbol",
        log_x=True,
        log_y=True,
        title="市值 vs 成交量 (气泡大小=市值)",
        template="plotly_dark"
    )
    return fig