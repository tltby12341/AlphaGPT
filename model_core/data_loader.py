import pandas as pd
import torch
import sqlalchemy
from .config import ModelConfig
from .factors import FeatureEngineer

class CryptoDataLoader:
    def __init__(self):
        self.engine = sqlalchemy.create_engine(ModelConfig.DB_URL)
        self.feat_tensor = None
        self.raw_data_cache = None
        self.target_ret = None
        
    def load_data(self, limit_tokens=500):
        print("Loading data from SQL...")
        top_query = f"""
        SELECT ticker FROM tokens 
        LIMIT {limit_tokens} 
        """
        addrs = pd.read_sql(top_query, self.engine)['ticker'].tolist()
        if not addrs: raise ValueError("No tokens found.")
        addr_str = "'" + "','".join(addrs) + "'"
        data_query = f"""
        SELECT time, ticker, open, high, low, close, volume, market_cap
        FROM ohlcv
        WHERE ticker IN ({addr_str})
        ORDER BY time ASC
        """
        df = pd.read_sql(data_query, self.engine)
        def to_tensor(col):
            pivot = df.pivot(index='time', columns='ticker', values=col)
            pivot = pivot.ffill().fillna(0.0)
            return torch.tensor(pivot.values.T, dtype=torch.float32, device=ModelConfig.DEVICE)
        self.raw_data_cache = {
            'open': to_tensor('open'),
            'high': to_tensor('high'),
            'low': to_tensor('low'),
            'close': to_tensor('close'),
            'volume': to_tensor('volume'),
            'market_cap': to_tensor('market_cap')
        }
        self.feat_tensor = FeatureEngineer.compute_features(self.raw_data_cache)
        op = self.raw_data_cache['open']
        t1 = torch.roll(op, -1, dims=1)
        t2 = torch.roll(op, -2, dims=1)
        self.target_ret = torch.log(t2 / (t1 + 1e-9))
        self.target_ret[:, -2:] = 0.0
        print(f"Data Ready. Shape: {self.feat_tensor.shape}")