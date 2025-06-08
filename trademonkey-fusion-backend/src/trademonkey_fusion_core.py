#!/usr/bin/env python3
"""
TradeMonkey Fusion Core - The Beast of All Beasts
"When legends combine, universes tremble" - Ancient Coding Proverb

The ultimate fusion of battle-tested wisdom and GPU-powered fury!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import asyncio
import ccxt
from datetime import datetime, timedelta
import json
import pickle
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# GPU optimization imports
import cupy as cp  # For even more GPU acceleration
from numba import cuda
import logging

# Setup epic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('TradeMonkeyFusion')

class MarketRegime(Enum):
    """Enhanced market regime classification from the OG TradeMonkey"""
    ACCUMULATION = 0    # Smart money accumulating
    MANIPULATION = 1    # Market makers manipulating  
    DISTRIBUTION = 2    # Smart money distributing
    VOLATILITY = 3      # High volatility regime
    TREND = 4          # Strong trending regime
    UNKNOWN = 5

@dataclass
class FusionConfig:
    """Configuration that merges all the best practices"""
    # GPU Settings
    use_gpu: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True  # For even faster training
    
    # Trading Parameters (from original TradeMonkey)
    initial_capital: float = 10000.0
    max_positions: int = 4
    position_size_pct: float = 0.25
    risk_per_trade: float = 0.01
    
    # Enhanced Risk Management
    max_daily_loss: float = -0.10
    trailing_stop_mult: float = 2.0
    stop_after_max_loss: bool = True
    dynamic_risk_adjustment: bool = True
    
    # Market Regime Risk Multipliers (from original config)
    regime_risk_multipliers: Dict[str, float] = None
    
    # Volume Profile Settings (battle-tested from original)
    volume_profile_bins: int = 50
    value_area_pct: float = 0.7
    ote_retracement: float = 0.618
    sweep_threshold: float = 0.02
    sweep_window: int = 5
    
    # ML Model Settings
    sequence_length: int = 50
    hidden_dim: int = 512
    num_heads: int = 16
    num_layers: int = 8
    dropout: float = 0.1
    prediction_horizon: int = 5
    
    # Signal Generation
    signal_threshold: float = 0.65
    confidence_threshold: float = 0.7
    
    # Backtesting (from proven engine)
    use_realistic_fees: bool = True
    slippage_bps: float = 5.0
    trading_fee_bps: float = 10.0
    
    def __post_init__(self):
        if self.regime_risk_multipliers is None:
            self.regime_risk_multipliers = {
                "accumulation": 1.2,    # Higher risk in accumulation
                "manipulation": 0.5,    # Lower risk during manipulation
                "distribution": 1.5,    # Higher risk in distribution
                "volatility": 0.7,     # Moderate risk in high vol
                "trend": 1.3,          # Higher risk in strong trends
                "unknown": 0.3         # Very conservative when uncertain
            }

class GPUAcceleratedFeatures:
    """GPU-accelerated feature engineering - Speed of Light Edition"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        logger.info(f"🚀 GPU Features initialized on {device}")
        
    def calculate_all_features(self, df: pd.DataFrame) -> torch.Tensor:
        """Calculate ALL features on GPU - the works!"""
        # Convert to GPU tensors
        ohlcv = torch.tensor(
            df[['open', 'high', 'low', 'close', 'volume']].values,
            dtype=torch.float32, device=self.device
        )
        
        features = []
        
        # === TREND INDICATORS ===
        features.append(self._gpu_sma(ohlcv[:, 3], 20))    # SMA 20
        features.append(self._gpu_sma(ohlcv[:, 3], 50))    # SMA 50
        features.append(self._gpu_ema(ohlcv[:, 3], 12))    # EMA 12
        features.append(self._gpu_ema(ohlcv[:, 3], 26))    # EMA 26
        
        # === MOMENTUM INDICATORS ===
        features.append(self._gpu_rsi(ohlcv[:, 3], 14))    # RSI
        features.append(self._gpu_rsi(ohlcv[:, 3], 7))     # Fast RSI
        
        # MACD
        macd, macd_signal = self._gpu_macd(ohlcv[:, 3])
        features.append(macd)
        features.append(macd_signal)
        features.append(macd - macd_signal)  # MACD histogram
        
        # Stochastic
        stoch_k, stoch_d = self._gpu_stochastic(ohlcv[:, 1], ohlcv[:, 2], ohlcv[:, 3])
        features.append(stoch_k)
        features.append(stoch_d)
        
        # === VOLATILITY INDICATORS ===
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self._gpu_bollinger_bands(ohlcv[:, 3])
        features.append((ohlcv[:, 3] - bb_middle) / (bb_upper - bb_lower + 1e-8))  # BB position
        features.append((bb_upper - bb_lower) / bb_middle)  # BB width
        
        # ATR
        atr = self._gpu_atr(ohlcv[:, 1], ohlcv[:, 2], ohlcv[:, 3])
        features.append(atr)
        features.append(atr / ohlcv[:, 3])  # ATR percentage
        
        # === VOLUME INDICATORS ===
        volume_sma = self._gpu_sma(ohlcv[:, 4], 20)
        features.append(ohlcv[:, 4] / (volume_sma + 1e-8))  # Volume ratio
        features.append(self._gpu_obv(ohlcv[:, 3], ohlcv[:, 4]))  # OBV
        
        # === PRICE ACTION FEATURES ===
        features.append(self._gpu_returns(ohlcv[:, 3], 1))   # 1-period return
        features.append(self._gpu_returns(ohlcv[:, 3], 5))   # 5-period return
        features.append(self._gpu_returns(ohlcv[:, 3], 20))  # 20-period return
        
        # High-Low ratio
        features.append((ohlcv[:, 1] - ohlcv[:, 2]) / ohlcv[:, 3])  # Candle range
        
        # === ADVANCED FEATURES ===
        # Fractal features
        features.append(self._gpu_fractal_dimension(ohlcv[:, 3]))
        
        # Market microstructure
        features.append(self._gpu_vwap(ohlcv))
        
        # Stack all features
        feature_tensor = torch.stack(features, dim=1)
        
        # Handle NaN values
        feature_tensor = torch.where(torch.isnan(feature_tensor), 
                                   torch.zeros_like(feature_tensor), 
                                   feature_tensor)
        
        logger.info(f"✅ Calculated {feature_tensor.shape[1]} features on GPU")
        return feature_tensor
    
    def _gpu_sma(self, prices: torch.Tensor, period: int) -> torch.Tensor:
        """GPU-optimized Simple Moving Average using convolution"""
        kernel = torch.ones(period, device=self.device) / period
        # Pad to maintain length
        padded = F.pad(prices, (period-1, 0), mode='replicate')
        return F.conv1d(padded.unsqueeze(0).unsqueeze(0), 
                       kernel.unsqueeze(0).unsqueeze(0)).squeeze()
    
    def _gpu_ema(self, prices: torch.Tensor, period: int) -> torch.Tensor:
        """GPU-optimized Exponential Moving Average"""
        alpha = 2.0 / (period + 1)
        ema = torch.zeros_like(prices)
        ema[0] = prices[0]
        
        # Vectorized EMA calculation
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
        
        return ema
    
    def _gpu_rsi(self, prices: torch.Tensor, period: int = 14) -> torch.Tensor:
        """GPU-optimized RSI calculation"""
        deltas = torch.diff(prices, prepend=prices[:1])
        gains = torch.where(deltas > 0, deltas, torch.zeros_like(deltas))
        losses = torch.where(deltas < 0, -deltas, torch.zeros_like(deltas))
        
        avg_gains = self._gpu_sma(gains, period)
        avg_losses = self._gpu_sma(losses, period)
        
        rs = avg_gains / (avg_losses + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _gpu_macd(self, prices: torch.Tensor, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[torch.Tensor, torch.Tensor]:
        """GPU-optimized MACD calculation"""
        ema_fast = self._gpu_ema(prices, fast)
        ema_slow = self._gpu_ema(prices, slow)
        macd_line = ema_fast - ema_slow
        signal_line = self._gpu_ema(macd_line, signal)
        return macd_line, signal_line
    
    def _gpu_stochastic(self, high: torch.Tensor, low: torch.Tensor, close: torch.Tensor, 
                       k_period: int = 14, d_period: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
        """GPU-optimized Stochastic Oscillator"""
        # Rolling min/max using unfold
        low_min = self._gpu_rolling_min(low, k_period)
        high_max = self._gpu_rolling_max(high, k_period)
        
        k_percent = 100 * (close - low_min) / (high_max - low_min + 1e-8)
        d_percent = self._gpu_sma(k_percent, d_period)
        
        return k_percent, d_percent
    
    def _gpu_bollinger_bands(self, prices: torch.Tensor, period: int = 20, std_mult: float = 2.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """GPU-optimized Bollinger Bands"""
        sma = self._gpu_sma(prices, period)
        std = self._gpu_rolling_std(prices, period)
        
        upper = sma + (std_mult * std)
        lower = sma - (std_mult * std)
        
        return upper, sma, lower
    
    def _gpu_atr(self, high: torch.Tensor, low: torch.Tensor, close: torch.Tensor, period: int = 14) -> torch.Tensor:
        """GPU-optimized Average True Range"""
        prev_close = torch.cat([close[:1], close[:-1]])
        
        tr1 = high - low
        tr2 = torch.abs(high - prev_close)
        tr3 = torch.abs(low - prev_close)
        
        true_range = torch.max(tr1, torch.max(tr2, tr3))
        atr = self._gpu_sma(true_range, period)
        
        return atr
    
    def _gpu_obv(self, close: torch.Tensor, volume: torch.Tensor) -> torch.Tensor:
        """GPU-optimized On-Balance Volume"""
        price_change = torch.diff(close, prepend=close[:1])
        direction = torch.sign(price_change)
        obv = torch.cumsum(direction * volume, dim=0)
        return obv
    
    def _gpu_returns(self, prices: torch.Tensor, periods: int) -> torch.Tensor:
        """GPU-optimized returns calculation"""
        return torch.diff(prices, n=periods, prepend=prices[:periods]) / prices
    
    def _gpu_fractal_dimension(self, prices: torch.Tensor, window: int = 20) -> torch.Tensor:
        """GPU-optimized fractal dimension calculation"""
        # Simplified Higuchi fractal dimension
        fd = torch.zeros_like(prices)
        
        for i in range(window, len(prices)):
            segment = prices[i-window:i]
            # Calculate path length
            path_length = torch.sum(torch.abs(torch.diff(segment)))
            # Normalize by Euclidean distance
            euclidean_dist = torch.sqrt(torch.tensor(window-1, dtype=torch.float32, device=self.device))
            fd[i] = torch.log(path_length) / torch.log(euclidean_dist)
        
        return fd
    
    def _gpu_vwap(self, ohlcv: torch.Tensor, window: int = 20) -> torch.Tensor:
        """GPU-optimized Volume Weighted Average Price"""
        typical_price = (ohlcv[:, 1] + ohlcv[:, 2] + ohlcv[:, 3]) / 3  # (H+L+C)/3
        volume = ohlcv[:, 4]
        
        vwap = torch.zeros_like(typical_price)
        
        for i in range(window, len(typical_price)):
            segment_price = typical_price[i-window:i]
            segment_volume = volume[i-window:i]
            
            vwap[i] = torch.sum(segment_price * segment_volume) / torch.sum(segment_volume)
        
        return vwap
    
    def _gpu_rolling_min(self, tensor: torch.Tensor, window: int) -> torch.Tensor:
        """GPU-optimized rolling minimum"""
        unfolded = tensor.unfold(0, window, 1)
        return torch.cat([torch.full((window-1,), tensor[0], device=self.device),
                         torch.min(unfolded, dim=1)[0]])
    
    def _gpu_rolling_max(self, tensor: torch.Tensor, window: int) -> torch.Tensor:
        """GPU-optimized rolling maximum"""
        unfolded = tensor.unfold(0, window, 1)
        return torch.cat([torch.full((window-1,), tensor[0], device=self.device),
                         torch.max(unfolded, dim=1)[0]])
    
    def _gpu_rolling_std(self, tensor: torch.Tensor, window: int) -> torch.Tensor:
        """GPU-optimized rolling standard deviation"""
        unfolded = tensor.unfold(0, window, 1)
        return torch.cat([torch.full((window-1,), 0.01, device=self.device),
                         torch.std(unfolded, dim=1)])

class VolumeProfileGPU:
    """GPU-accelerated volume profile analysis from the OG TradeMonkey"""
    
    def __init__(self, bins: int = 50, device: str = "cuda"):
        self.bins = bins
        self.device = device
        
    def calculate_volume_profile(self, ohlcv: torch.Tensor, window: int = 200) -> Dict[str, torch.Tensor]:
        """Calculate volume profile levels on GPU"""
        results = {
            'poc': torch.zeros(len(ohlcv), device=self.device),  # Point of Control
            'vah': torch.zeros(len(ohlcv), device=self.device),  # Value Area High
            'val': torch.zeros(len(ohlcv), device=self.device),  # Value Area Low
            'in_value_area': torch.zeros(len(ohlcv), device=self.device),
            'in_ote': torch.zeros(len(ohlcv), device=self.device)  # Optimal Trade Entry
        }
        
        for i in range(window, len(ohlcv)):
            segment = ohlcv[i-window:i]
            high_prices = segment[:, 1]  # High
            low_prices = segment[:, 2]   # Low
            close_prices = segment[:, 3] # Close
            volumes = segment[:, 4]      # Volume
            
            # Create price levels
            min_price = torch.min(low_prices)
            max_price = torch.max(high_prices)
            price_levels = torch.linspace(min_price, max_price, self.bins, device=self.device)
            
            # Calculate volume at each price level
            volume_at_price = torch.zeros(self.bins, device=self.device)
            
            for j in range(len(segment)):
                # Distribute volume across price range for this bar
                bar_high = high_prices[j]
                bar_low = low_prices[j]
                bar_volume = volumes[j]
                
                # Find which bins this bar covers
                level_mask = (price_levels >= bar_low) & (price_levels <= bar_high)
                num_levels = torch.sum(level_mask.float())
                
                if num_levels > 0:
                    volume_at_price[level_mask] += bar_volume / num_levels
            
            # Find Point of Control (highest volume)
            poc_idx = torch.argmax(volume_at_price)
            poc_price = price_levels[poc_idx]
            
            # Calculate Value Area (70% of volume)
            sorted_volume, sorted_indices = torch.sort(volume_at_price, descending=True)
            cumulative_volume = torch.cumsum(sorted_volume, dim=0)
            total_volume = torch.sum(volume_at_price)
            
            value_area_threshold = 0.7 * total_volume
            value_area_mask = cumulative_volume <= value_area_threshold
            value_area_indices = sorted_indices[value_area_mask]
            
            if len(value_area_indices) > 0:
                vah = torch.max(price_levels[value_area_indices])
                val = torch.min(price_levels[value_area_indices])
            else:
                vah = poc_price * 1.01
                val = poc_price * 0.99
            
            # Store results
            results['poc'][i] = poc_price
            results['vah'][i] = vah
            results['val'][i] = val
            
            # Check if current price is in value area
            current_price = close_prices[-1]
            results['in_value_area'][i] = (current_price >= val) & (current_price <= vah)
            
            # Check OTE (Optimal Trade Entry) - 61.8% retracement level
            ote_level = val + 0.618 * (vah - val)
            results['in_ote'][i] = torch.abs(current_price - ote_level) / ote_level < 0.02  # Within 2%
        
        return results

class FusionTransformer(nn.Module):
    """Enhanced Transformer model with multi-task learning"""
    
    def __init__(self, config: FusionConfig):
        super().__init__()
        
        self.config = config
        self.input_dim = 25  # Number of features from GPU calculator
        
        # Input projection
        self.input_projection = nn.Linear(self.input_dim, config.hidden_dim)
        self.pos_encoding = self._create_positional_encoding(config.sequence_length, config.hidden_dim)
        
        # Multi-head attention layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm for better stability
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=config.num_layers,
            enable_nested_tensor=False
        )
        
        # Multi-task output heads
        self.regime_classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, len(MarketRegime))
        )
        
        self.price_predictor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Tanh()  # Bound predictions to [-1, 1]
        )
        
        self.volatility_predictor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 4),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 4, 1),
            nn.Softplus()  # Ensure positive volatility
        )
        
        self.confidence_estimator = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 4),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 4, 1),
            nn.Sigmoid()  # Confidence between 0 and 1
        )
        
        # Apply weight initialization
        self.apply(self._init_weights)
        
        logger.info(f"🧠 FusionTransformer initialized with {self._count_parameters():,} parameters")
    
    def _create_positional_encoding(self, seq_len: int, dim: int) -> torch.Tensor:
        """Create positional encoding for transformer"""
        pe = torch.zeros(seq_len, dim)
        position = torch.arange(0, seq_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, dim, 2).float() * 
                           -(np.log(10000.0) / dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
    
    def _init_weights(self, module):
        """Initialize weights using Xavier/Glorot initialization"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def _count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass with multi-task outputs"""
        batch_size, seq_len, _ = x.shape
        
        # Project input features
        x = self.input_projection(x)
        
        # Add positional encoding
        if self.pos_encoding.device != x.device:
            self.pos_encoding = self.pos_encoding.to(x.device)
        
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # Apply transformer encoding
        if mask is not None:
            # Create attention mask if provided
            attn_mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            encoded = self.transformer(x, src_key_padding_mask=attn_mask)
        else:
            encoded = self.transformer(x)
        
        # Use the last sequence element for predictions
        last_hidden = encoded[:, -1, :]
        
        # Multi-task outputs
        regime_logits = self.regime_classifier(last_hidden)
        price_prediction = self.price_predictor(last_hidden)
        volatility_prediction = self.volatility_predictor(last_hidden)
        confidence = self.confidence_estimator(last_hidden)
        
        return {
            'regime_logits': regime_logits,
            'price_prediction': price_prediction,
            'volatility_prediction': volatility_prediction,
            'confidence': confidence,
            'hidden_states': encoded
        }

class TradeMonkeyFusionCore:
    """The Ultimate Trading Beast - Fusion of All Powers!"""
    
    def __init__(self, config: FusionConfig):
        self.config = config
        self.device = config.device
        
        # Initialize GPU components
        self.gpu_features = GPUAcceleratedFeatures(self.device)
        self.volume_profile = VolumeProfileGPU(config.volume_profile_bins, self.device)
        
        # Initialize the neural network
        self.model = FusionTransformer(config).to(self.device)
        
        # Mixed precision training
        if config.mixed_precision and config.device == "cuda":
            self.scaler = torch.cuda.amp.GradScaler()
            logger.info("🚀 Mixed precision training enabled")
        else:
            self.scaler = None
        
        # Training utilities
        self.optimizer = None
        self.scheduler = None
        
        # Performance tracking
        self.training_metrics = {
            'epoch_losses': [],
            'validation_scores': [],
            'regime_accuracies': []
        }
        
        logger.info(f"🦍 TradeMonkey Fusion Core initialized on {self.device}")
        logger.info(f"💪 Ready to unleash the beast with {self.model._count_parameters():,} parameters!")
    
    def prepare_data_sequences(self, df: pd.DataFrame) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Prepare data sequences for the transformer"""
        # Calculate features on GPU
        features = self.gpu_features.calculate_all_features(df)
        
        # Calculate volume profile
        ohlcv = torch.tensor(
            df[['open', 'high', 'low', 'close', 'volume']].values,
            dtype=torch.float32, device=self.device
        )
        volume_profile_data = self.volume_profile.calculate_volume_profile(ohlcv)
        
        # Add volume profile features to main features
        vp_features = torch.stack([
            volume_profile_data['in_value_area'],
            volume_profile_data['in_ote'],
            (ohlcv[:, 3] - volume_profile_data['poc']) / volume_profile_data['poc'],  # Distance from POC
        ], dim=1)
        
        # Combine all features
        all_features = torch.cat([features, vp_features], dim=1)
        
        # Create sequences for transformer
        seq_len = self.config.sequence_length
        sequences = []
        
        for i in range(seq_len, len(all_features)):
            sequence = all_features[i-seq_len:i]
            sequences.append(sequence)
        
        if sequences:
            feature_sequences = torch.stack(sequences, dim=0)
        else:
            feature_sequences = torch.empty(0, seq_len, all_features.shape[1], device=self.device)
        
        # Create targets for training
        targets = self._create_targets(df, seq_len)
        
        return feature_sequences, targets, volume_profile_data
    
    def _create_targets(self, df: pd.DataFrame, seq_offset: int) -> Dict[str, torch.Tensor]:
        """Create training targets"""
        horizon = self.config.prediction_horizon
        
        # Price targets (future returns)
        future_returns = []
        for i in range(seq_offset, len(df) - horizon):
            current_price = df['close'].iloc[i]
            future_price = df['close'].iloc[i + horizon]
            ret = (future_price - current_price) / current_price
            future_returns.append(ret)
        
        price_targets = torch.tensor(future_returns, dtype=torch.float32, device=self.device)
        
        # Regime targets (simplified classification)
        regime_targets = self._classify_regimes(df, seq_offset)
        
        # Volatility targets
        volatility_targets = self._calculate_volatility_targets(df, seq_offset, horizon)
        
        return {
            'price_targets': price_targets,
            'regime_targets': regime_targets,
            'volatility_targets': volatility_targets
        }
    
    def _classify_regimes(self, df: pd.DataFrame, seq_offset: int) -> torch.Tensor:
        """Classify market regimes using battle-tested logic"""
        regimes = []
        
        for i in range(seq_offset, len(df)):
            if i < 50:  # Need enough history
                regimes.append(MarketRegime.UNKNOWN.value)
                continue
            
            # Calculate regime indicators
            sma20 = df['close'].iloc[i-20:i].mean()
            sma50 = df['close'].iloc[i-50:i].mean()
            current_price = df['close'].iloc[i]
            
            # Volume analysis
            avg_volume = df['volume'].iloc[i-20:i].mean()
            current_volume = df['volume'].iloc[i]
            
            # Volatility analysis
            recent_returns = df['close'].iloc[i-10:i].pct_change().std()
            
            # Price momentum
            momentum_5 = (current_price - df['close'].iloc[i-5]) / df['close'].iloc[i-5]
            momentum_20 = (current_price - df['close'].iloc[i-20]) / df['close'].iloc[i-20]
            
            # Classification logic (battle-tested from original TradeMonkey)
            if current_price > sma20 > sma50 and momentum_20 > 0.05:
                # Strong uptrend - Distribution phase
                regime = MarketRegime.DISTRIBUTION.value
            elif current_price < sma20 < sma50 and momentum_20 < -0.05:
                # Strong downtrend - Accumulation phase  
                regime = MarketRegime.ACCUMULATION.value
            elif recent_returns > 0.03:  # High volatility
                regime = MarketRegime.VOLATILITY.value
            elif abs(momentum_5) < 0.01 and current_volume < avg_volume * 0.8:
                # Low momentum, low volume - Manipulation
                regime = MarketRegime.MANIPULATION.value
            elif abs(momentum_20) > 0.02:
                # Clear trend
                regime = MarketRegime.TREND.value
            else:
                regime = MarketRegime.UNKNOWN.value
            
            regimes.append(regime)
        
        return torch.tensor(regimes, dtype=torch.long, device=self.device)
    
    def _calculate_volatility_targets(self, df: pd.DataFrame, seq_offset: int, horizon: int) -> torch.Tensor:
        """Calculate future volatility targets"""
        volatilities = []
        
        for i in range(seq_offset, len(df) - horizon):
            # Calculate realized volatility over the prediction horizon
            future_returns = df['close'].iloc[i:i+horizon].pct_change().dropna()
            if len(future_returns) > 1:
                vol = future_returns.std() * np.sqrt(252)  # Annualized volatility
            else:
                vol = 0.0
            volatilities.append(vol)
        
        return torch.tensor(volatilities, dtype=torch.float32, device=self.device)
    
    async def generate_signals(self, df: pd.DataFrame) -> Dict:
        """Generate trading signals using the fusion approach"""
        logger.info("🎯 Generating signals with FUSION POWER!")
        
        # Prepare features
        feature_sequences, _, volume_profile_data = self.prepare_data_sequences(df)
        
        if len(feature_sequences) == 0:
            return {'action': 'hold', 'confidence': 0.0, 'reason': 'insufficient_data'}
        
        # Get model predictions
        self.model.eval()
        with torch.no_grad():
            # Use the last sequence for prediction
            last_sequence = feature_sequences[-1:].unsqueeze(0)  # Add batch dimension
            
            if self.config.mixed_precision and self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(last_sequence)
            else:
                outputs = self.model(last_sequence)
            
            # Extract predictions
            regime_probs = F.softmax(outputs['regime_logits'], dim=-1)
            price_pred = outputs['price_prediction'].item()
            volatility_pred = outputs['volatility_prediction'].item()
            confidence = outputs['confidence'].item()
            
            # Get most likely regime
            regime_idx = torch.argmax(regime_probs, dim=-1).item()
            regime = MarketRegime(regime_idx)
            regime_confidence = regime_probs[0, regime_idx].item()
        
        # Apply regime-specific risk adjustment
        risk_multiplier = self.config.regime_risk_multipliers.get(
            regime.name.lower(), 0.5
        )
        
        # Volume profile confirmation
        current_idx = len(df) - 1
        in_value_area = volume_profile_data['in_value_area'][current_idx].item() > 0.5
        in_ote = volume_profile_data['in_ote'][current_idx].item() > 0.5
        
        # Enhanced signal generation
        base_signal_strength = confidence * risk_multiplier
        
        # Volume profile boost
        if in_value_area:
            base_signal_strength *= 1.2
        if in_ote:
            base_signal_strength *= 1.3
        
        # Volatility adjustment
        if volatility_pred > 0.3:  # High predicted volatility
            base_signal_strength *= 0.8  # Reduce position size
        elif volatility_pred < 0.1:  # Low predicted volatility
            base_signal_strength *= 1.2  # Increase position size
        
        # Final signal decision
        if price_pred > 0.02 and base_signal_strength > self.config.signal_threshold:
            action = 'buy'
            signal_strength = base_signal_strength
        elif price_pred < -0.02 and base_signal_strength > self.config.signal_threshold:
            action = 'sell'
            signal_strength = base_signal_strength
        else:
            action = 'hold'
            signal_strength = 0.0
        
        # Calculate position size based on signal strength and volatility
        base_position_size = self.config.position_size_pct
        volatility_adjusted_size = base_position_size / (1 + volatility_pred)
        final_position_size = min(volatility_adjusted_size * signal_strength, 
                                self.config.position_size_pct)
        
        signal = {
            'action': action,
            'confidence': confidence,
            'regime': regime.name,
            'regime_confidence': regime_confidence,
            'risk_multiplier': risk_multiplier,
            'signal_strength': signal_strength,
            'predicted_return': price_pred,
            'predicted_volatility': volatility_pred,
            'position_size': final_position_size,
            'volume_confirmations': {
                'in_value_area': in_value_area,
                'in_ote': in_ote
            },
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'model_version': 'fusion_v1.0',
                'features_count': feature_sequences.shape[-1]
            }
        }
        
        logger.info(f"🎯 Signal generated: {action.upper()} with {confidence:.3f} confidence")
        logger.info(f"📊 Regime: {regime.name} ({regime_confidence:.3f}) | Volatility: {volatility_pred:.3f}")
        
        return signal
    
    def train_model(self, df: pd.DataFrame, epochs: int = 100, validation_split: float = 0.2):
        """Train the fusion model with all the bells and whistles"""
        logger.info(f"🏋️ Training FUSION MODEL for {epochs} epochs on {self.device}")
        
        # Prepare training data
        feature_sequences, targets, _ = self.prepare_data_sequences(df)
        
        if len(feature_sequences) == 0:
            logger.error("❌ No training sequences available!")
            return
        
        # Split into train/validation
        split_idx = int(len(feature_sequences) * (1 - validation_split))
        
        train_features = feature_sequences[:split_idx]
        train_targets = {k: v[:split_idx] for k, v in targets.items()}
        
        val_features = feature_sequences[split_idx:]
        val_targets = {k: v[split_idx:] for k, v in targets.items()}
        
        logger.info(f"📊 Training samples: {len(train_features)}, Validation: {len(val_features)}")
        
        # Initialize optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=0.001,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=0.003,
            epochs=epochs,
            steps_per_epoch=1,
            pct_start=0.3,
            anneal_strategy='cos'
        )
        
        # Loss functions
        regime_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        price_criterion = nn.MSELoss()
        volatility_criterion = nn.MSELoss()
        
        # Training loop
        best_val_loss = float('inf')
        patience = 20
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_losses = {'total': 0.0, 'regime': 0.0, 'price': 0.0, 'volatility': 0.0}
            
            # Batch processing (simulate mini-batches)
            batch_size = min(32, len(train_features))
            num_batches = len(train_features) // batch_size
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(train_features))
                
                batch_features = train_features[start_idx:end_idx]
                batch_regime_targets = train_targets['regime_targets'][start_idx:end_idx]
                batch_price_targets = train_targets['price_targets'][start_idx:end_idx]
                batch_vol_targets = train_targets['volatility_targets'][start_idx:end_idx]
                
                self.optimizer.zero_grad()
                
                if self.config.mixed_precision and self.scaler:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_features)
                        
                        # Calculate losses
                        regime_loss = regime_criterion(outputs['regime_logits'], batch_regime_targets)
                        price_loss = price_criterion(outputs['price_prediction'].squeeze(), batch_price_targets)
                        vol_loss = volatility_criterion(outputs['volatility_prediction'].squeeze(), batch_vol_targets)
                        
                        # Weighted combination
                        total_loss = regime_loss + 2.0 * price_loss + 0.5 * vol_loss
                    
                    # Backward pass with mixed precision
                    self.scaler.scale(total_loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(batch_features)
                    
                    # Calculate losses
                    regime_loss = regime_criterion(outputs['regime_logits'], batch_regime_targets)
                    price_loss = price_criterion(outputs['price_prediction'].squeeze(), batch_price_targets)
                    vol_loss = volatility_criterion(outputs['volatility_prediction'].squeeze(), batch_vol_targets)
                    
                    # Weighted combination
                    total_loss = regime_loss + 2.0 * price_loss + 0.5 * vol_loss
                    
                    # Backward pass
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                
                # Accumulate losses
                train_losses['total'] += total_loss.item()
                train_losses['regime'] += regime_loss.item()
                train_losses['price'] += price_loss.item()
                train_losses['volatility'] += vol_loss.item()
            
            # Average training losses
            for key in train_losses:
                train_losses[key] /= num_batches
            
            # Validation phase
            val_losses = self._validate_model(val_features, val_targets, regime_criterion, 
                                            price_criterion, volatility_criterion)
            
            # Update scheduler
            self.scheduler.step()
            
            # Log progress
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}/{epochs}")
                logger.info(f"  Train Loss: {train_losses['total']:.6f} "
                           f"(R: {train_losses['regime']:.4f}, P: {train_losses['price']:.4f}, V: {train_losses['volatility']:.4f})")
                logger.info(f"  Val Loss: {val_losses['total']:.6f} "
                           f"(R: {val_losses['regime']:.4f}, P: {val_losses['price']:.4f}, V: {val_losses['volatility']:.4f})")
                logger.info(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Early stopping
            if val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']
                patience_counter = 0
                # Save best model
                self.save_model('best_fusion_model.pth')
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                logger.info(f"🛑 Early stopping triggered after {epoch} epochs")
                break
            
            # Store metrics
            self.training_metrics['epoch_losses'].append(train_losses['total'])
            self.training_metrics['validation_scores'].append(val_losses['total'])
        
        logger.info("🎉 Training completed!")
        logger.info(f"💎 Best validation loss: {best_val_loss:.6f}")
        
        # Load best model
        self.load_model('best_fusion_model.pth')
    
    def _validate_model(self, val_features: torch.Tensor, val_targets: Dict, 
                       regime_criterion, price_criterion, volatility_criterion) -> Dict:
        """Validate the model"""
        self.model.eval()
        val_losses = {'total': 0.0, 'regime': 0.0, 'price': 0.0, 'volatility': 0.0}
        
        with torch.no_grad():
            if self.config.mixed_precision and self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(val_features)
            else:
                outputs = self.model(val_features)
            
            # Calculate validation losses
            regime_loss = regime_criterion(outputs['regime_logits'], val_targets['regime_targets'])
            price_loss = price_criterion(outputs['price_prediction'].squeeze(), val_targets['price_targets'])
            vol_loss = volatility_criterion(outputs['volatility_prediction'].squeeze(), val_targets['volatility_targets'])
            
            total_loss = regime_loss + 2.0 * price_loss + 0.5 * vol_loss
            
            val_losses['total'] = total_loss.item()
            val_losses['regime'] = regime_loss.item()
            val_losses['price'] = price_loss.item()
            val_losses['volatility'] = vol_loss.item()
        
        return val_losses
    
    def save_model(self, path: str):
        """Save the complete model state"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': asdict(self.config),
            'training_metrics': self.training_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        if self.optimizer:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, path)
        logger.info(f"💾 Model saved to {path}")
    
    def load_model(self, path: str):
        """Load the complete model state"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'optimizer_state_dict' in checkpoint and self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'training_metrics' in checkpoint:
            self.training_metrics = checkpoint['training_metrics']
        
        logger.info(f"📂 Model loaded from {path}")
    
    async def run_enhanced_backtest(self, df: pd.DataFrame) -> Dict:
        """Run comprehensive backtest with all the original TradeMonkey wisdom"""
        logger.info("🧪 Running ENHANCED BACKTEST with fusion power!")
        
        # Initialize backtest state
        capital = self.config.initial_capital
        positions = []
        trades = []
        equity_curve = []
        
        # Performance tracking
        peak_capital = capital
        max_drawdown = 0.0
        total_trades = 0
        winning_trades = 0
        
        # Get all signals for the dataset
        signals_history = []
        
        # Simulate real-time signal generation
        min_lookback = max(200, self.config.sequence_length)  # Need enough data for features
        
        for i in range(min_lookback, len(df), 5):  # Check every 5 bars
            # Get data up to current point
            current_data = df.iloc[:i+1].copy()
            
            try:
                signal = await self.generate_signals(current_data)
                signal['bar_index'] = i
                signal['timestamp'] = df.index[i] if hasattr(df.index, '__getitem__') else i
                signals_history.append(signal)
                
                # Simulate trade execution
                if signal['action'] in ['buy', 'sell'] and len(positions) < self.config.max_positions:
                    entry_price = df['close'].iloc[i]
                    position_size = signal['position_size'] * capital
                    
                    position = {
                        'entry_bar': i,
                        'entry_price': entry_price,
                        'size': position_size,
                        'side': signal['action'],
                        'stop_loss': entry_price * (0.98 if signal['action'] == 'buy' else 1.02),
                        'take_profit': entry_price * (1.04 if signal['action'] == 'buy' else 0.96),
                        'signal': signal
                    }
                    positions.append(position)
                    
                # Check exit conditions for existing positions
                for pos in positions[:]:  # Create a copy to iterate safely
                    current_price = df['close'].iloc[i]
                    
                    # Check stop loss / take profit
                    if pos['side'] == 'buy':
                        if current_price <= pos['stop_loss'] or current_price >= pos['take_profit']:
                            # Close position
                            exit_price = current_price
                            pnl = (exit_price - pos['entry_price']) / pos['entry_price'] * pos['size']
                            
                            trades.append({
                                'entry_bar': pos['entry_bar'],
                                'exit_bar': i,
                                'entry_price': pos['entry_price'],
                                'exit_price': exit_price,
                                'pnl': pnl,
                                'side': pos['side'],
                                'exit_reason': 'stop_loss' if current_price <= pos['stop_loss'] else 'take_profit'
                            })
                            
                            capital += pnl
                            total_trades += 1
                            if pnl > 0:
                                winning_trades += 1
                            
                            positions.remove(pos)
                    
                    elif pos['side'] == 'sell':
                        if current_price >= pos['stop_loss'] or current_price <= pos['take_profit']:
                            # Close position
                            exit_price = current_price
                            pnl = (pos['entry_price'] - exit_price) / pos['entry_price'] * pos['size']
                            
                            trades.append({
                                'entry_bar': pos['entry_bar'],
                                'exit_bar': i,
                                'entry_price': pos['entry_price'],
                                'exit_price': exit_price,
                                'pnl': pnl,
                                'side': pos['side'],
                                'exit_reason': 'stop_loss' if current_price >= pos['stop_loss'] else 'take_profit'
                            })
                            
                            capital += pnl
                            total_trades += 1
                            if pnl > 0:
                                winning_trades += 1
                            
                            positions.remove(pos)
                
                # Update equity curve
                current_equity = capital + sum(
                    (df['close'].iloc[i] - pos['entry_price']) / pos['entry_price'] * pos['size']
                    if pos['side'] == 'buy' else
                    (pos['entry_price'] - df['close'].iloc[i]) / pos['entry_price'] * pos['size']
                    for pos in positions
                )
                
                equity_curve.append({
                    'bar': i,
                    'equity': current_equity,
                    'drawdown': (peak_capital - current_equity) / peak_capital if peak_capital > 0 else 0
                })
                
                # Update peak and drawdown
                if current_equity > peak_capital:
                    peak_capital = current_equity
                else:
                    current_drawdown = (peak_capital - current_equity) / peak_capital
                    max_drawdown = max(max_drawdown, current_drawdown)
                
            except Exception as e:
                logger.warning(f"Error generating signal at bar {i}: {e}")
                continue
        
        # Calculate final metrics
        total_return = (capital - self.config.initial_capital) / self.config.initial_capital
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Calculate Sharpe ratio
        if trades:
            trade_returns = [trade['pnl'] / self.config.initial_capital for trade in trades]
            if len(trade_returns) > 1:
                sharpe = np.mean(trade_returns) / np.std(trade_returns) * np.sqrt(252) if np.std(trade_returns) > 0 else 0
            else:
                sharpe = 0
        else:
            sharpe = 0
        
        # Compile results
        backtest_results = {
            'total_return': total_return,
            'final_capital': capital,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe,
            'trades': trades,
            'equity_curve': equity_curve,
            'signals_generated': len(signals_history),
            'regime_distribution': {},
            'performance_by_regime': {}
        }
        
        # Analyze regime performance
        regime_stats = {}
        for signal in signals_history:
            regime = signal.get('regime', 'unknown')
            if regime not in regime_stats:
                regime_stats[regime] = {'count': 0, 'total_return': 0}
            regime_stats[regime]['count'] += 1
        
        backtest_results['regime_distribution'] = regime_stats
        
        logger.info("🎉 Enhanced backtest completed!")
        logger.info(f"💰 Total Return: {total_return:.2%}")
        logger.info(f"📊 Win Rate: {win_rate:.2%}")
        logger.info(f"🎯 Total Trades: {total_trades}")
        logger.info(f"📉 Max Drawdown: {max_drawdown:.2%}")
        logger.info(f"⚡ Sharpe Ratio: {sharpe:.2f}")
        
        return backtest_results

# Example usage - The Beast Awakens!
async def main():
    """Unleash the Fusion Beast!"""
    logger.info("🔥 INITIALIZING TRADEMONKEY FUSION CORE 🔥")
    
    # Create configuration
    config = FusionConfig(
        use_gpu=torch.cuda.is_available(),
        initial_capital=10000.0,
        risk_per_trade=0.01,
        signal_threshold=0.65,
        mixed_precision=True
    )
    
    # Initialize the fusion system
    fusion_core = TradeMonkeyFusionCore(config)
    
    logger.info("🚀 TradeMonkey Fusion Core ready for deployment!")
    logger.info("💪 All systems nominal - ready to print money!")
    
    # Example workflow:
    # 1. Load your data: df = pd.read_csv('your_data.csv')
    # 2. Train the model: fusion_core.train_model(df, epochs=100)
    # 3. Generate signals: signal = await fusion_core.generate_signals(df)
    # 4. Run backtest: results = await fusion_core.run_enhanced_backtest(df)
    
    return fusion_core

if __name__ == "__main__":
    asyncio.run(main())
            