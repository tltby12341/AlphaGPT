import json
import os
import time
from loguru import logger
from datetime import datetime

class PaperTrader:
    def __init__(self, initial_balance=100000.0):
        self.state_file = "paper_trading_state.json"
        self.initial_balance = initial_balance
        self.load_state()

    def load_state(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    self.state = json.load(f)
            except Exception:
                self.reset_state()
        else:
            self.reset_state()

    def reset_state(self):
        self.state = {
            "balance": self.initial_balance,
            "positions": {}, # ticker -> amount
            "history": []
        }
        self.save_state()

    def save_state(self):
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=2)

    async def get_balance(self):
        return self.state["balance"]

    async def buy(self, ticker: str, amount_usd: float, price: float):
        """
        Execute a paper buy.
        """
        balance = self.state["balance"]
        if balance < amount_usd:
            logger.warning(f"Insufficient funds for paper buy: {balance} < {amount_usd}")
            return False

        if price <= 0:
            logger.error("Invalid price for buy")
            return False

        units = amount_usd / price
        commission = min(max(1.0, 0.005 * units), amount_usd * 0.01) # Simple commission model
        cost = amount_usd + commission

        if balance < cost:
             logger.warning(f"Insufficient funds including commission: {balance} < {cost}")
             return False
        
        self.state["balance"] -= cost
        
        current_pos = self.state["positions"].get(ticker, 0.0)
        self.state["positions"][ticker] = current_pos + units
        
        trade_record = {
            "type": "BUY",
            "ticker": ticker,
            "amount_usd": amount_usd,
            "price": price,
            "units": units,
            "commission": commission,
            "time": datetime.now().isoformat()
        }
        self.state["history"].append(trade_record)
        self.save_state()
        
        logger.success(f"PAPER BUY: {ticker} | ${amount_usd:.2f} | {units:.4f} shares @ ${price:.2f}")
        return True

    async def sell(self, ticker: str, percentage: float, price: float):
        """
        Execute a paper sell.
        """
        units_held = self.state["positions"].get(ticker, 0.0)
        if units_held <= 0:
            logger.warning(f"No position in {ticker} to sell.")
            return False

        sell_units = units_held * percentage
        if sell_units <= 0: return False

        proceeds = sell_units * price
        commission = min(max(1.0, 0.005 * sell_units), proceeds * 0.01)
        net_proceeds = proceeds - commission

        self.state["balance"] += net_proceeds
        self.state["positions"][ticker] = units_held - sell_units
        
        if self.state["positions"][ticker] < 1e-6:
            del self.state["positions"][ticker]

        trade_record = {
            "type": "SELL",
            "ticker": ticker,
            "percentage": percentage,
            "price": price,
            "units": sell_units,
            "net_proceeds": net_proceeds,
            "time": datetime.now().isoformat()
        }
        self.state["history"].append(trade_record)
        self.save_state()
        
        logger.success(f"PAPER SELL: {ticker} | {percentage:.0%} | {sell_units:.4f} shares @ ${price:.2f}")
        return True

    async def close(self):
        self.save_state()
