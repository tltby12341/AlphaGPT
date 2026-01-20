import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    DB_NAME = "stock_quant.db"
    # Use SQLite URL for SQLAlchemy
    DB_URL = f"sqlite:///{DB_NAME}" 
    # For aiosqlite, we just need the filename
    DB_PATH = DB_NAME
    
    
    # Load dynamic config
    import json
    try:
        with open("config.json", "r") as f:
            _conf = json.load(f)
    except FileNotFoundError:
        _conf = {}

    TIMEFRAME = _conf.get("TIMEFRAME", "1d")
    US_STOCKS_TICKERS = _conf.get("US_STOCKS_TICKERS", [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "BRK-B", "UNH", "JNJ",
        "JPM", "XOM", "V", "PG", "MA", "HD", "CVX", "ABBV", "PEP", "MRK",
        "KO", "LLY", "BAC", "AVGO", "COST", "MCD", "TMO", "CSCO", "ACN", "WMT"
    ])
    CONCURRENCY = 5
    HISTORY_DAYS = _conf.get("HISTORY_DAYS", 730)