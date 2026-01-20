import yfinance as yf
import pandas as pd
import asyncio
from typing import List, Tuple
from loguru import logger
from .config import Config

class StockFetcher:
    def __init__(self):
        self.tickers = Config.US_STOCKS_TICKERS

    async def get_tokens_info(self) -> List[Tuple]:
        """
        Fetch basic info for tickers. 
        Returns list of tuples: (ticker, symbol, name, sector)
        """
        results = []
        logger.info(f"Fetching metadata for {len(self.tickers)} stocks...")
        
        # yfinance Info fetching can be slow sequentially, but acceptable for <100 stocks.
        # For larger lists, we might want to just mock the name/sector or use bulk endpoint if available.
        # yf.Tickers(tickers) can bulk fetch.
        
        try:
            # Use Tickers object for efficiently accessing multiple tickers
            tickers_obj = yf.Tickers(" ".join(self.tickers))
            
            for ticker_symbol in self.tickers:
                try:
                    info = tickers_obj.tickers[ticker_symbol].info
                    name = info.get('shortName', ticker_symbol)
                    sector = info.get('sector', 'Unknown')
                    results.append((ticker_symbol, ticker_symbol, name, sector))
                except Exception as e:
                    logger.warning(f"Failed to fetch info for {ticker_symbol}: {e}")
                    # Fallback
                    results.append((ticker_symbol, ticker_symbol, ticker_symbol, 'Unknown'))
                    
        except Exception as e:
            logger.error(f"Bulk fetch error: {e}")
        
        return results

    async def get_token_history(self, ticker: str, days: int = 365) -> List[Tuple]:
        """
        Fetch OHLCV history for a ticker.
        Returns list of tuples: (time, ticker, open, high, low, close, volume, market_cap)
        """
        try:
            # Run blocking yfinance call in executor
            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(None, self._fetch_history_sync, ticker, days)
            
            if df is None or df.empty:
                return None
                
            records = []
            for index, row in df.iterrows():
                # yfinance returns timestamps in index
                dt = index.to_pydatetime()
                # Market Cap approximation: Close * SharesOutstanding? 
                # yf history doesn't return share count per day. 
                # using 0.0 or fetching current shares * close for rough estimate.
                # For this simplified version, we can just use Close * 1e9 (dummy) or try to get shares.
                # Let's use Volume * Close as 'Turnover' or just store 0 for Market Cap if unknown.
                # Better yet: access info.sharesOutstanding and multiply. 
                # But querying info inside history loop is slow.
                # Let's assume Market Cap is not CRITICAL for factors except 'liquidity' check.
                # We can store Volume * Close (Dollar Volume) into Market Cap column for now,
                # OR just keep it 0.
                
                # Decision: Store Dollar Volume (Turnover) in 'market_cap' column?
                # No, that's confusing. Let's store 0 unless we have data,
                # BUT 'market_cap' column in DB is double. 
                # Let's modify logic: The model uses 'liquidity' and 'fdv'. 
                # In stock schema we mapped 'liquidity' -> removed, 'fdv' -> 'market_cap'.
                # Actually in my plan I said: "liquidity, fdv -> market_cap".
                # Let's put Dollar Volume (Close * Volume) as a proxy for Liquidity?
                # The factors.py uses 'liquidity' / 'fdv' for health.
                # Let's store Volume as is, and maybe 0 for MCap.
                
                records.append((
                    dt,
                    ticker,
                    float(row['Open']),
                    float(row['High']),
                    float(row['Low']),
                    float(row['Close']),
                    float(row['Volume']),
                    0.0 # Market Cap placeholder
                ))
            return records

        except Exception as e:
            logger.error(f"Error fetching history for {ticker}: {e}")
            return None

    def _fetch_history_sync(self, ticker, days):
        try:
            t = yf.Ticker(ticker)
            # Fetch slightly more to ensure coverage
            period = "2y" if days > 365 else "1y"
            if days > 730: period = "5y"
            
            df = t.history(period=period, interval="1d")
            # Filter by days? yf period is approx.
            # Only keep last N days
            cutoff_date = pd.Timestamp.now(tz=df.index.tz) - pd.Timedelta(days=days)
            df = df[df.index >= cutoff_date]
            
            if df.empty: return None
            return df
        except Exception as e:
            logger.error(f"yfinance sync error for {ticker}: {e}")
            return None
