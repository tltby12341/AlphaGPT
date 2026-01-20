import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from loguru import logger
from tqdm import tqdm
import json
import os

from .alphagpt import AlphaGPT
from .vm import StackVM
from .data_loader import CryptoDataLoader
from .config import ModelConfig
from .backtest import MemeBacktest

class AlphaEngine:
    def __init__(self):
        self.device = ModelConfig.DEVICE
        self.model = AlphaGPT().to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-5)
        self.vm = StackVM()
        self.loader = CryptoDataLoader()
        self.backtest = MemeBacktest()
        
        # RL Hyperparameters
        self.clip_eps = 0.2
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.entropy_coef = 0.01
        self.value_coef = 0.5
        
    def train(self):
        logger.info("Loading Data...")
        try:
            self.loader.load_data()
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return

        if self.loader.feat_tensor is None:
            logger.error("No features loaded.")
            return

        feat_tensor = self.loader.feat_tensor # [N_stocks, N_features, Time] ? No, check loader.
        # Loader: to_tensor is [Time, Tickers]. FeatureEngineer adds dim.
        # Let's assume feat_tensor shape is [Time, Tickers, Features] or [Tickers, Features, Time].
        # StackVM expects: feat_tensor[:, token, :] -> [Time, Tickers] (if token < offset)
        # Check vm.py: stack.append(feat_tensor[:, token, :])
        # So feat_tensor must be [Time, Feat_Dim, Tickers] or something similar?
        # Actually in loader: pivot is [Time, Tickers]. values.T is [Tickers, Time].
        # FeatureEngineer likely stacks them.
        # Start training loop
        
        self.model.train()
        best_reward = -float('inf')
        best_formula = None
        
        pbar = tqdm(range(ModelConfig.TRAIN_STEPS))
        for step in pbar:
            # 1. Rollout / Sampling
            batch_size = ModelConfig.BATCH_SIZE
            
            # Start tokens: we can use a fixed start token or just empty.
            # Model pos_emb handles positions.
            # We don't have a BOS token in vocab (just features + ops).
            # Let's randomly pick valid feature tokens as start or use model to predict first.
            # We can start with a dummy input (e.g. 0) or handling empty seq in model (if supported).
            # Simplest: Random valid feature as start token.
            
            # States: [B, T]
            # LogProbs: [B, T]
            # Actions: [B, T]
            # Values: [B, T]
            
            # Generate formulas autoregressively
            generated_seqs, log_probs, values, entropies = self.generate_batch(batch_size)
            
            # 2. Evaluation / Reward
            rewards = []
            valid_formulas = []
            
            for i in range(batch_size):
                formula = generated_seqs[i].tolist()
                # Execute on VM
                # VM execute needs: formula_tokens, feat_tensor
                # feat_tensor shape: [Tickers, Features, Time] ?
                # Let's verify data shape from logs later if needed.
                signal = self.vm.execute(formula, feat_tensor)
                
                if signal is None:
                    # Invalid formula (syntax error or stack underflow)
                    r = -0.05 # Small penalty
                    rewards.append(r)
                else:
                    # Backtest
                    # signal: [Tickers, Time]
                    # raw_data: dict of [Tickers, Time] tensors
                    # target_ret: [Tickers, Time]
                    fit, ret = self.backtest.evaluate(signal, self.loader.raw_data_cache, self.loader.target_ret)
                    rewards.append(fit.item())
                    
                    if fit.item() > best_reward:
                        best_reward = fit.item()
                        best_formula = formula
                        self.save_best_strategy(formula, best_reward)
            
            rewards = torch.tensor(rewards, device=self.device)
            
            # 3. PPO Update
            # We treat the whole formula generation as one "episode" or step-wise?
            # Usually for text gen PPO, we reward the end or dense.
            # Here reward is only at the end.
            
            # PPO Loss Calculation
            # Simplified: Use rewards as return for all steps (with decay if needed, but here instant)
            # Normalize rewards
            rewards_norm = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            
            agg_loss = 0
            # For each step in the generated sequence (excluding start token)
            num_steps = values.shape[1]
            for t in range(num_steps):
                # Advantage: R - V(s)
                # Here R is the final reward for all steps (masked by gamma?)
                # Simplified: advantage = reward - value
                adv = rewards_norm - values[:, t].detach().squeeze()
                
                # Actor Loss
                ratio = torch.exp(log_probs[:, t] - log_probs[:, t].detach())
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Critic Loss
                critic_loss = 0.5 * (rewards_norm - values[:, t].squeeze()).pow(2).mean()
                
                # Entropy
                entropy_loss = -entropies[:, t].mean()
                
                loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss
                agg_loss += loss

            agg_loss = agg_loss / num_steps
            
            self.optimizer.zero_grad()
            agg_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
            
            pbar.set_description(f"Best: {best_reward:.4f} Loss: {agg_loss.item():.4f}")

    def generate_batch(self, batch_size):
        # Autoregressive generation
        seq_len = ModelConfig.MAX_FORMULA_LEN
        
        # Start with random valid features (tokens 0 to INPUT_DIM-1)
        # vocab size check
        start_tokens = torch.randint(0, ModelConfig.INPUT_DIM, (batch_size, 1), device=self.device)
        
        curr_seq = start_tokens
        all_log_probs = []
        all_values = []
        all_entropies = []
        
        for _ in range(seq_len - 1):
            logits, value, _ = self.model(curr_seq)
            # Logits: [B, vocab_size] (from last step)
            # Value: [B, 1]
            
            # Should be last step only?
            # AlphaGPT forward returns full sequence logits if checking code?
            # Actually typically transformers return [B, T, V]
            # MTPHead stacks outputs. Check alphagpt.py forward.
            # x = self.blocks(x...) -> x is [B, T, D]
            # last_emb = x[:, -1, :] -> It selects ONLY the last step embedding!
            # So forward(idx) returns predictions for the NEXT token based on idx.
            
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
            
            all_log_probs.append(log_prob)
            all_values.append(value)
            all_entropies.append(entropy)
            
            curr_seq = torch.cat([curr_seq, action.unsqueeze(1)], dim=1)
            
        return curr_seq, torch.stack(all_log_probs, dim=1), torch.stack(all_values, dim=1), torch.stack(all_entropies, dim=1)

    def save_best_strategy(self, formula, score):
        try:
            with open("best_meme_strategy.json", "w") as f:
                json.dump({
                    "formula": formula,
                    "score": float(score)
                }, f)
        except Exception as e:
            logger.error(f"Save failed: {e}")

if __name__ == "__main__":
    engine = AlphaEngine()
    engine.train()