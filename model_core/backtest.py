import torch

class MemeBacktest:
    def __init__(self):
        self.trade_size = 1000.0
        self.min_liq = 500000.0
        self.base_fee = 0.0010 # 0.1% for stocks

    def evaluate(self, factors, raw_data, target_ret):
        # Use market_cap if liquidity not present (adapt for stocks)
        liquidity = raw_data.get('liquidity', raw_data.get('market_cap'))
        if liquidity is None: # Fallback
            liquidity = torch.ones_like(factors) * 1e9 

        signal = torch.sigmoid(factors)
        is_safe = (liquidity > self.min_liq).float()
        
        # Position sizing: 0 or 1 based on threshold
        position = (signal > 0.85).float() * is_safe
        
        # Impact slippage model
        impact_slippage = self.trade_size / (liquidity + 1e-9)
        impact_slippage = torch.clamp(impact_slippage, 0.0, 0.05)
        
        # Total fee: base_fee + slippage
        total_slippage_one_way = self.base_fee + impact_slippage
        
        # Turnover calculation
        prev_pos = torch.roll(position, 1, dims=1)
        prev_pos[:, 0] = 0
        turnover = torch.abs(position - prev_pos)
        
        tx_cost = turnover * total_slippage_one_way
        
        # PnL
        gross_pnl = position * target_ret
        net_pnl = gross_pnl - tx_cost
        
        # Metrics
        cum_ret = net_pnl.sum(dim=1)
        big_drawdowns = (net_pnl < -0.05).float().sum(dim=1) # Penalize big drops
        
        score = cum_ret - (big_drawdowns * 2.0)
        
        # Activity penalty: must trade at least a bit
        activity = position.sum(dim=1)
        score = torch.where(activity < 5, torch.tensor(-10.0, device=score.device), score)
        
        final_fitness = torch.median(score)
        return final_fitness, cum_ret.mean().item()