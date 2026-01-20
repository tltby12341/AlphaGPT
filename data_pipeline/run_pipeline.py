import asyncio
from loguru import logger
from .db_manager import DBManager
from .stock_fetcher import StockFetcher
from .config import Config

async def main():
    db = DBManager()
    fetcher = StockFetcher()
    
    try:
        await db.connect()
        await db.init_schema()
        
        # 1. Update Token Metadata
        logger.info("Updating stock metadata...")
        tokens_info = await fetcher.get_tokens_info()
        await db.upsert_tokens(tokens_info)
        
        # 2. Update OHLCV
        logger.info("Updating stock history...")
        for ticker in Config.US_STOCKS_TICKERS:
            logger.info(f"Fetching {ticker}...")
            history = await fetcher.get_token_history(ticker, days=Config.HISTORY_DAYS)
            if history:
                await db.batch_insert_ohlcv(history)
                logger.success(f"Inserted {len(history)} records for {ticker}")
            else:
                logger.warning(f"No history found for {ticker}")
                
    except Exception as e:
        logger.exception(f"Pipeline crashed: {e}")
    finally:
        await db.close()

if __name__ == "__main__":
    asyncio.run(main())