import torch
import torch.nn as nn


class RMSNormFactor(nn.Module):
    """RMSNorm for factor normalization"""
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight



class StockIndicators:
    @staticmethod
    def liquidity_health(volume, market_cap):
        # Turnover rate proxy: Volume / MarketCap (if MarketCap is available/nonzero)
        # Using a safe division. 
        # Note: In stock fetcher we might have put 0 for market_cap if data missing.
        # Let's assume market_cap is roughly available or we use log volume as proxy.
        turnover = volume / (market_cap + 1e-9)
        # Clamp turnover to normal range (e.g. 0% to 20%) then scale
        return torch.clamp(turnover * 100.0, 0.0, 5.0)

    @staticmethod
    def buy_sell_imbalance(close, open_, high, low):
        range_hl = high - low + 1e-9
        body = close - open_
        strength = body / range_hl
        return torch.tanh(strength * 3.0)

    @staticmethod
    def fomo_acceleration(volume, window=5):
        vol_prev = torch.roll(volume, 1, dims=1)
        vol_chg = (volume - vol_prev) / (vol_prev + 1.0)
        acc = vol_chg - torch.roll(vol_chg, 1, dims=1)
        return torch.clamp(acc, -5.0, 5.0)

    @staticmethod
    def price_deviation(close, window=20):
        pad = torch.zeros((close.shape[0], window-1), device=close.device)
        c_pad = torch.cat([pad, close], dim=1)
        ma = c_pad.unfold(1, window, 1).mean(dim=-1)
        dev = (close - ma) / (ma + 1e-9)
        return dev

    @staticmethod
    def volatility_clustering(close, window=10):
        """Detect volatility clustering patterns"""
        ret = torch.log(close / (torch.roll(close, 1, dims=1) + 1e-9))
        ret_sq = ret ** 2
        
        pad = torch.zeros((ret_sq.shape[0], window-1), device=close.device)
        ret_sq_pad = torch.cat([pad, ret_sq], dim=1)
        vol_ma = ret_sq_pad.unfold(1, window, 1).mean(dim=-1)
        
        return torch.sqrt(vol_ma + 1e-9)

    @staticmethod
    def momentum_reversal(close, window=5):
        """Capture momentum reversal signals"""
        ret = torch.log(close / (torch.roll(close, 1, dims=1) + 1e-9))
        
        pad = torch.zeros((ret.shape[0], window-1), device=close.device)
        ret_pad = torch.cat([pad, ret], dim=1)
        mom = ret_pad.unfold(1, window, 1).sum(dim=-1)
        
        # Detect reversals
        mom_prev = torch.roll(mom, 1, dims=1)
        reversal = (mom * mom_prev < 0).float()
        
        return reversal

    @staticmethod
    def relative_strength(close, high, low, window=14):
        """RSI-like indicator for strength detection"""
        ret = close - torch.roll(close, 1, dims=1)
        
        gains = torch.relu(ret)
        losses = torch.relu(-ret)
        
        pad = torch.zeros((gains.shape[0], window-1), device=close.device)
        gains_pad = torch.cat([pad, gains], dim=1)
        losses_pad = torch.cat([pad, losses], dim=1)
        
        avg_gain = gains_pad.unfold(1, window, 1).mean(dim=-1)
        avg_loss = losses_pad.unfold(1, window, 1).mean(dim=-1)
        
        rs = (avg_gain + 1e-9) / (avg_loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        
        return (rsi - 50) / 50  # Normalize


class AdvancedFactorEngineer:
    """Advanced feature engineering with multiple factor types"""
    def __init__(self):
        self.rms_norm = RMSNormFactor(1)
    
    def robust_norm(self, t):
        """Robust normalization using median absolute deviation"""
        median = torch.nanmedian(t, dim=1, keepdim=True)[0]
        mad = torch.nanmedian(torch.abs(t - median), dim=1, keepdim=True)[0] + 1e-6
        norm = (t - median) / mad
        return torch.clamp(norm, -5.0, 5.0)
    
    def compute_advanced_features(self, raw_dict):
        """Compute 12-dimensional feature space with advanced factors"""
        c = raw_dict['close']
        o = raw_dict['open']
        h = raw_dict['high']
        l = raw_dict['low']
        v = raw_dict['volume']
        mcap = raw_dict['market_cap']
        
        # Basic factors
        ret = torch.log(c / (torch.roll(c, 1, dims=1) + 1e-9))
        liq_score = StockIndicators.liquidity_health(v, mcap)
        pressure = StockIndicators.buy_sell_imbalance(c, o, h, l)
        fomo = StockIndicators.fomo_acceleration(v)
        dev = StockIndicators.price_deviation(c)
        log_vol = torch.log1p(v)
        
        # Advanced factors
        vol_cluster = StockIndicators.volatility_clustering(c)
        momentum_rev = StockIndicators.momentum_reversal(c)
        rel_strength = StockIndicators.relative_strength(c, h, l)
        
        # High-low range
        hl_range = (h - l) / (c + 1e-9)
        
        # Close position in range
        close_pos = (c - l) / (h - l + 1e-9)
        
        # Volume trend
        vol_prev = torch.roll(v, 1, dims=1)
        vol_trend = (v - vol_prev) / (vol_prev + 1.0)
        
        features = torch.stack([
            self.robust_norm(ret),
            liq_score,
            pressure,
            self.robust_norm(fomo),
            self.robust_norm(dev),
            self.robust_norm(log_vol),
            self.robust_norm(vol_cluster),
            momentum_rev,
            self.robust_norm(rel_strength),
            self.robust_norm(hl_range),
            close_pos,
            self.robust_norm(vol_trend)
        ], dim=1)
        
        return features


class FeatureEngineer:
    INPUT_DIM = 6

    @staticmethod
    def compute_features(raw_dict):
        c = raw_dict['close']
        o = raw_dict['open']
        h = raw_dict['high']
        l = raw_dict['low']
        v = raw_dict['volume']
        mcap = raw_dict['market_cap']
        
        ret = torch.log(c / (torch.roll(c, 1, dims=1) + 1e-9))
        liq_score = StockIndicators.liquidity_health(v, mcap)
        pressure = StockIndicators.buy_sell_imbalance(c, o, h, l)
        fomo = StockIndicators.fomo_acceleration(v)
        dev = StockIndicators.price_deviation(c)
        log_vol = torch.log1p(v)
        
        def robust_norm(t):
            median = torch.nanmedian(t, dim=1, keepdim=True)[0]
            mad = torch.nanmedian(torch.abs(t - median), dim=1, keepdim=True)[0] + 1e-6
            norm = (t - median) / mad
            return torch.clamp(norm, -5.0, 5.0)

        features = torch.stack([
            robust_norm(ret),
            liq_score,
            pressure,
            robust_norm(fomo),
            robust_norm(dev),
            robust_norm(log_vol)
        ], dim=1)
        
        return features