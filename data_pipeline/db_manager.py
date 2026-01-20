import aiosqlite
from loguru import logger
from .config import Config

class DBManager:
    def __init__(self):
        self.db_path = Config.DB_PATH
        self.conn = None

    async def connect(self):
        try:
            self.conn = await aiosqlite.connect(self.db_path)
            logger.info(f"Connected to SQLite: {self.db_path}")
        except Exception as e:
            logger.error(f"DB Connection failed: {e}")
            raise e

    async def close(self):
        if self.conn:
            await self.conn.close()
            logger.info("DB Connection closed")

    async def init_schema(self):
        # Tokens table
        await self.conn.execute("""
            CREATE TABLE IF NOT EXISTS tokens (
                ticker TEXT PRIMARY KEY,
                symbol TEXT,
                name TEXT,
                sector TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # OHLCV table
        # No timescaledb hypertable in SQLite, just standard table with index
        await self.conn.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv (
                time TIMESTAMP,
                ticker TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                market_cap REAL,
                PRIMARY KEY (ticker, time)
            );
        """)
        # Index on time for faster querying
        await self.conn.execute("CREATE INDEX IF NOT EXISTS idx_ohlcv_time ON ohlcv (time);")
        await self.conn.commit()
        logger.info("Schema initialized (SQLite)")

    async def upsert_tokens(self, tokens: list):
        if not tokens: return
        try:
            # SQLite upsert syntax
            await self.conn.executemany("""
                INSERT INTO tokens (ticker, symbol, name, sector)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(ticker) DO UPDATE SET
                    name = excluded.name,
                    sector = excluded.sector,
                    updated_at = CURRENT_TIMESTAMP
            """, tokens)
            await self.conn.commit()
            logger.info(f"Upserted {len(tokens)} tokens")
        except Exception as e:
            logger.error(f"Failed to upsert tokens: {e}")

    async def batch_insert_ohlcv(self, records: list):
        """
        records: list of tuples (time, ticker, open, high, low, close, volume, market_cap)
        """
        if not records: return
        try:
            await self.conn.executemany("""
                INSERT OR IGNORE INTO ohlcv (time, ticker, open, high, low, close, volume, market_cap)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, records)
            await self.conn.commit()
        except Exception as e:
            logger.error(f"Failed to insert OHLCV: {e}")