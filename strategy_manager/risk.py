from .config import StrategyConfig
from loguru import logger
from datetime import datetime, time as dtime
import pytz

class RiskEngine:
    def __init__(self):
        self.config = StrategyConfig()
        # US Eastern Time for market hours
        self.market_tz = pytz.timezone('US/Eastern')

    async def check_safety(self, ticker, liquidity_usd):
        # 1. Check Market Hours (9:30 AM - 4:00 PM ET)
        if not self._is_market_open():
            # Allow scan, but maybe warn? Or just return True if we want to trade pre-market?
            # Let's enforce market hours for safety.
            logger.warning(f"Risk: Market closed. Current EST time: {datetime.now(self.market_tz)}")
            return False

        # 2. Check Liquidity (Market Cap proxy check already done via DB filter mostly)
        # But we can check dynamic liquidity if passed
        if liquidity_usd < 1000000: # $1M Daily Volume minimum
            logger.warning(f"Risk: Low liquidity proxy ({liquidity_usd}) for {ticker}")
            return False

        return True

    def _is_market_open(self):
        now = datetime.now(self.market_tz)
        # Weekends
        if now.weekday() >= 5: return False
        
        market_start = dtime(9, 30)
        market_end = dtime(16, 0)
        return market_start <= now.time() <= market_end

    def calculate_position_size(self, wallet_balance_usd):
        # Fixed amount or percentage
        # Let's use config.ENTRY_AMOUNT_SOL as USD amount
        # Assumption: Config variable reused as USD
        size = 2000.0 # Default fixed size $2000 per trade
        
        if wallet_balance_usd < size:
            return 0.0
            
        return size

    async def close(self):
        pass