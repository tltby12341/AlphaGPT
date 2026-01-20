import asyncio
import time
import torch
import json
import pandas as pd
from loguru import logger
from .config import StrategyConfig
from .risk import RiskEngine
from execution.paper_trader import PaperTrader
from data_pipeline.stock_fetcher import StockFetcher
from data_pipeline.db_manager import DBManager
from model_core.vm import StackVM
from model_core.data_loader import CryptoDataLoader # Renaming locally if needed, but class name is still CryptoDataLoader in file

class StrategyRunner:
    def __init__(self):
        self.config = StrategyConfig()
        self.risk = RiskEngine()
        self.trader = PaperTrader()
        self.fetcher = StockFetcher()
        self.vm = StackVM()
        self.db = DBManager()
        
        self.active_strategy = None
        self.feature_loader = CryptoDataLoader() # TODO: Rename class to StockDataLoader eventually
        self.running = True

    async def initialize(self):
        logger.info("Initializing Stock Strategy Runner...")
        await self.db.connect()
        self.load_strategy()
        # Initialize Balance Check
        bal = await self.trader.get_balance()
        logger.info(f"Paper Trading Balance: ${bal:.2f}")
        
    def load_strategy(self):
        try:
            with open("best_meme_strategy.json", "r") as f:
                data = json.load(f)
                self.active_strategy = data["formula"]
                logger.success(f"Loaded Strategy (Score: {data['score']:.4f})")
                logger.info(f"Formula: {self.active_strategy}")
        except FileNotFoundError:
            logger.warning("No strategy file found. Please run training first.")
            self.running = False # Stop if no strategy

    async def run_loop(self):
        if not self.active_strategy: 
            logger.error("No strategy loaded. Exiting loop.")
            return

        logger.info("Starting trading loop...")
        while self.running:
            try:
                loop_start = time.time()
                
                # 1. Start of Loop - Check Market Hours
                if not self.risk._is_market_open():
                    logger.info("Market Closed. Sleeping 60s...")
                    await asyncio.sleep(60)
                    continue

                # 2. Update Data (Simulation: Assume data is fresh or fetch)
                # In real usage, we might fetch latest minute bar here.
                # For now, we rely on what's in DB (daily data mostly).
                # To make this 'live', we'd need intraday fetching. 
                # For this MVP, we proceed with daily logic or assume DB has latest.
                
                # 3. Scan Opportunities
                await self.scan_market()
                
                # 4. Manage Positions
                await self.manage_positions()
                
                elapsed = time.time() - loop_start
                sleep_time = max(10, 60 - elapsed)
                logger.info(f"Cycle finished in {elapsed:.2f}s. Sleeping {sleep_time:.2f}s...")
                await asyncio.sleep(sleep_time)

            except KeyboardInterrupt:
                self.running = False
            except Exception as e:
                logger.exception(f"Loop error: {e}")
                await asyncio.sleep(10)

    async def scan_market(self):
        try:
            # Re-load data to get latest state
            self.feature_loader.load_data(limit_tokens=100)
            if self.feature_loader.feat_tensor is None: return

            feat_tensor = self.feature_loader.feat_tensor # (Time, Batch, Feat)
            
            # Use latest time step
            last_feats = feat_tensor[-1, :, :] # (Batch, Feat)
            
            # Execute Formula
            scores = self.vm.execute(self.active_strategy, last_feats.unsqueeze(0)) 
            if scores is None: return

            scores = scores.squeeze() # (Batch)
            if scores.dim() == 0: scores = scores.unsqueeze(0) # Handle single stock case
            
            # Map indices to tickers
            tickers = list(self.feature_loader.raw_data_cache['close'].columns)
            
            # Identify Top Opportunities
            # If batch size small, topk might fail if k > batch
            k = min(3, scores.shape[0])
            top_k = torch.topk(scores, k=k)
            
            balance = await self.trader.get_balance()
            
            for i in range(top_k.values.shape[0]):
                score = top_k.values[i].item()
                idx = top_k.indices[i].item()
                ticker = tickers[idx]
                
                if score > self.config.BUY_THRESHOLD:
                    if ticker in self.trader.state["positions"]:
                        continue # Already in
                    
                    curr_close = self.feature_loader.raw_data_cache['close'][-1, idx].item()
                    curr_vol = self.feature_loader.raw_data_cache['volume'][-1, idx].item()

                    # simple liquidity check
                    safe = await self.risk.check_safety(ticker, curr_vol * curr_close)
                    if safe:
                        pos_size = self.risk.calculate_position_size(balance)
                        if pos_size > 0:
                            await self.trader.buy(ticker, pos_size, curr_close)
                            
        except Exception as e:
            logger.error(f"Scan failed: {e}")

    async def manage_positions(self):
        # iterate copy of keys to avoid modification during iteration
        for ticker in list(self.trader.state["positions"].keys()):
            # Logic: Sell if price drops below stop loss or rises above take profit?
            # Or re-evaluate score?
            # For MVP: Random sell or simple hold logic?
            # Let's implement a simple Stop Loss / Take Profit based on entry price?
            # We didn't store entry price in simple positions map suitable for that.
            # We'll just skip for now or simulate a sell after some time.
            pass

    async def close(self):
        await self.db.close()
        await self.trader.close()

if __name__ == "__main__":
    runner = StrategyRunner()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(runner.initialize())
    try:
        loop.run_until_complete(runner.run_loop())
    finally:
        loop.run_until_complete(runner.close())