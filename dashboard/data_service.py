import json
import os
import pandas as pd
import sqlalchemy
from dotenv import load_dotenv

load_dotenv()

class DashboardService:
    def __init__(self):
        # Use SQLite for reading data (shared with pipeline)
        self.db_path = "stock_quant.db"
        self.engine = sqlalchemy.create_engine(f"sqlite:///{self.db_path}")
        self.state_file = "paper_trading_state.json"

    def get_wallet_balance(self):
        try:
            with open(self.state_file, "r") as f:
                data = json.load(f)
                return data.get("balance", 0.0)
        except Exception:
            return 0.0

    def load_portfolio(self):
        try:
            with open(self.state_file, "r") as f:
                data = json.load(f)
                positions = data.get("positions", {})
                if not positions: return pd.DataFrame()
                
                # Convert positions dict {ticker: units} to DataFrame
                # We need current price to calc value.
                # Ideally fetch latest price from DB.
                
                df_list = []
                for ticker, units in positions.items():
                    # Get latest close from DB
                    try:
                        price = self._get_latest_price(ticker)
                    except:
                        price = 0.0
                        
                    # We don't track entry price in simple paper trader state yet
                    # So PnL might be tricky without trade history replay.
                    # For MVP, we show current Value.
                    
                    df_list.append({
                        "symbol": ticker,
                        "amount_held": units,
                        "current_price": price,
                        "market_value": units * price,
                        "entry_price": 0.0, # Placeholder
                        "pnl_pct": 0.0 # Placeholder
                    })
                
                return pd.DataFrame(df_list)
        except FileNotFoundError:
            return pd.DataFrame()

    def _get_latest_price(self, ticker):
        query = f"SELECT close FROM ohlcv WHERE ticker='{ticker}' ORDER BY time DESC LIMIT 1"
        try:
            df = pd.read_sql(query, self.engine)
            if not df.empty:
                return df.iloc[0]['close']
        except:
            pass
        return 0.0

    def load_strategy_info(self):
        try:
            with open("best_meme_strategy.json", "r") as f:
                return json.load(f)
        except:
            return {"formula": "Not Trained Yet"}

    def get_market_overview(self, limit=50):
        # Get latest snapshot for all tickers
        # SQLite dialect
        query = """
        SELECT t.symbol, o.ticker, o.close, o.volume, o.market_cap, o.time
        FROM ohlcv o
        JOIN tokens t ON o.ticker = t.ticker
        WHERE o.time = (SELECT MAX(time) FROM ohlcv)
        ORDER BY o.market_cap DESC
        LIMIT 50
        """
        try:
            return pd.read_sql(query, self.engine)
        except Exception as e:
            print(f"Market view error: {e}")
            return pd.DataFrame()
    
    def get_recent_logs(self, n=50):
        # We were logging to stderr/stdout mostly with Loguru
        # Unless we configured a file sink.
        # Let's check where loguru logs to.
        # If no file sink, this might be empty.
        # Assuming strategy.log exists or we add it.
        log_file = "strategy.log"
        if not os.path.exists(log_file): return []
        
        with open(log_file, "r") as f:
            lines = f.readlines()
            return lines[-n:]