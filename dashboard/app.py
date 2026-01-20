import streamlit as st
import pandas as pd
import time
import json
from data_service import DashboardService
from visualizer import plot_pnl_distribution, plot_market_scatter

st.set_page_config(
    page_title="AlphaGPT 美股量化终端",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card {
        background-color: #1E1E1E;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #333;
    }
    .stDataFrame { border: none; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_service():
    return DashboardService()

svc = get_service()

st.sidebar.title("AlphaGPT 美股")
st.sidebar.markdown("---")

with st.sidebar:
    st.subheader("账户状态")
    bal = svc.get_wallet_balance()
    st.metric("现金余额 (USD)", f"${bal:,.2f}")
    
    st.markdown("---")
    st.subheader("控制面板")
    if st.button("刷新数据"):
        st.rerun()

col1, col2, col3, col4 = st.columns(4)
portfolio_df = svc.load_portfolio()
market_df = svc.get_market_overview()
strategy_data = svc.load_strategy_info()

open_positions = len(portfolio_df)
total_value = portfolio_df['market_value'].sum() if not portfolio_df.empty else 0.0

with col1:
    st.metric("持仓数量", f"{open_positions}")
with col2:
    st.metric("账户总权益", f"${(bal + total_value):,.2f}")
with col3:
    st.metric("持仓市值", f"${total_value:,.2f}")
with col4:
    st.metric("策略得分", f"{strategy_data.get('score', 0):.4f}", help=str(strategy_data.get('formula')))

from controller import PersistentController

# Initialize Controller
if 'controller' not in st.session_state:
    st.session_state.controller = PersistentController()
ctl: PersistentController = st.session_state.controller

# Tab Structure
tab1, tab2, tab3, tab4, tab5 = st.tabs(["持仓组合", "市场扫描", "策略详情", "系统控制", "配置管理"])

with tab1:
    st.subheader("当前持仓")
    if not portfolio_df.empty:
        # Display Table
        display_cols = ['symbol', 'amount_held', 'current_price', 'market_value']
        # Rename cols for display
        show_df = portfolio_df[display_cols].copy()
        show_df.columns = ['代码', '持仓股数', '当前价格', '持仓市值']
        
        st.dataframe(show_df, use_container_width=True, hide_index=True)
        
        # Display Chart
        st.plotly_chart(plot_pnl_distribution(portfolio_df), use_container_width=True)
    else:
        st.info("暂无持仓，机器人正在扫描市场机会...")

with tab2:
    st.subheader("美股市场概览")
    if not market_df.empty:
        st.plotly_chart(plot_market_scatter(market_df), use_container_width=True)
        st.dataframe(market_df, use_container_width=True)
    else:
        st.warning("数据库中无市场数据，请检查数据管道是否运行？")

with tab3:
    st.subheader("策略信息")
    st.json(strategy_data)
    
    st.subheader("策略引擎日志 (Trade Loop)")
    # Tail logs from the runner process specifically
    logs = ctl.get_log_tail("trading_runner", 20)
    st.code(logs, language="text")

with tab4:
    st.subheader("系统控制面板")
    
    @st.dialog("进程详细日志", width="large")
    def show_log_modal(proc_name, display_name):
        st.write(f"正在观察: **{display_name}** (实时刷新中...)")
        
        # Container for logs
        log_container = st.empty()
        
        # Infinite loop to stream logs (stops when user closes dialog)
        while True:
            logs = ctl.get_full_log(proc_name, 2000)
            
            html_content = f"""
            <div style="
                height: 600px;
                overflow-y: auto;
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 15px;
                color: #000000;
                font-family: 'Source Code Pro', monospace;
                font-size: 12px;
                white-space: pre-wrap;
                line-height: 1.5;
            ">{logs}</div>
            """
            
            log_container.markdown(html_content, unsafe_allow_html=True)
            time.sleep(1)
            # No st.rerun() needed! Loop updates in place.

    # Remove active_modal check (using direct button calls now)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.write("#### 1. 数据更新 (Pipeline)")
        status = ctl.get_status("data_pipeline")
        st.write(f"状态: **{status}**")
        if status == "Stopped":
            if st.button("启动数据下载", key="btn_start_data"):
                ctl.run_process("data_pipeline", ".venv/bin/python -m data_pipeline.run_pipeline")
                st.rerun()
        else:
            if st.button("停止下载", key="btn_stop_data"):
                ctl.stop_process("data_pipeline")
                st.rerun()
        
        if st.button("查看完整日志", key="view_log_data"):
            show_log_modal("data_pipeline", "数据管道 (Data Pipeline)")

    with c2:
        st.write("#### 2. 策略训练 (Training)")
        status = ctl.get_status("model_training")
        st.write(f"状态: **{status}**")
        if status == "Stopped":
            if st.button("开始模型训练", key="btn_start_train"):
                ctl.run_process("model_training", ".venv/bin/python -m model_core.engine")
                st.rerun()
        else:
            if st.button("停止训练", key="btn_stop_train"):
                ctl.stop_process("model_training")
                st.rerun()
        
        if st.button("查看完整日志", key="view_log_train"):
            show_log_modal("model_training", "模型训练 (PPO Training)")

    with c3:
        st.write("#### 3. 模拟交易 (Trading)")
        status = ctl.get_status("trading_runner")
        st.write(f"状态: **{status}**")
        if status == "Stopped":
            if st.button("启动交易机器人", key="btn_start_trade"):
                ctl.run_process("trading_runner", ".venv/bin/python -m strategy_manager.runner")
                st.rerun()
        else:
            if st.button("停止机器人", key="btn_stop_trade"):
                ctl.stop_process("trading_runner")
                st.rerun()
        
        if st.button("查看完整日志", key="view_log_trade"):
            show_log_modal("trading_runner", "交易机器人 (Paper Trader)")

with tab5:
    st.subheader("全局配置管理")
    
    current_conf = ctl.load_config()
    
    with st.form("config_form"):
        st.write("### 基础设置")
        tickers_str = st.text_area("美股关注列表 (JSON 数组格式)", 
                                   value=json.dumps(current_conf.get("US_STOCKS_TICKERS", []), indent=2),
                                   height=200)
        
        c_1, c_2 = st.columns(2)
        with c_1:
            hist_days = st.number_input("历史数据回溯天数", value=current_conf.get("HISTORY_DAYS", 730))
            train_steps = st.number_input("模型训练步数", value=current_conf.get("TRAIN_STEPS", 50))
        with c_2:
            batch_size = st.number_input("Batch Size", value=current_conf.get("BATCH_SIZE", 32))
            buy_threshold = st.number_input("买入阈值 (Score)", value=current_conf.get("BUY_THRESHOLD", 0.1))
            
        submitted = st.form_submit_button("保存配置")
        if submitted:
            try:
                new_tickers = json.loads(tickers_str)
                current_conf.update({
                    "US_STOCKS_TICKERS": new_tickers,
                    "HISTORY_DAYS": hist_days,
                    "TRAIN_STEPS": train_steps,
                    "BATCH_SIZE": batch_size,
                    "BUY_THRESHOLD": buy_threshold
                })
                success, msg = ctl.save_config(current_conf)
                if success:
                    st.success("配置已保存！请重启相关模块以生效。")
                else:
                    st.error(f"保存失败: {msg}")
            except json.JSONDecodeError:
                st.error("股票列表格式错误，必须是有效的 JSON 数组。")

time.sleep(1) 
# Initialize default state
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = True

if st.checkbox("自动刷新 (1秒)", key="auto_refresh"):
    time.sleep(1)
    st.rerun()