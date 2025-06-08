#!/usr/bin/env python3
"""
TradeMonkey Lite - Enhanced Indicators with Stochastic Oscillator
"When momentum meets oscillation, profits are born!" - Quantum Trading Wisdom
"""

import pandas as pd
import numpy as np
import ta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger('EnhancedIndicators')

class SignalStrength(Enum):
    """Enhanced signal strength with scoring"""
    LEGENDARY = 100   # 90-100 points
    STRONG = 85       # 80-89 points  
    MEDIUM = 65       # 60-79 points
    WEAK = 45         # 40-59 points
    NONE = 0          # <40 points

@dataclass
class IndicatorSignal:
    """Individual indicator signal with strength"""
    name: str
    signal: str  # 'long', 'short', 'neutral'
    strength: float  # 0-100 score
    value: float
    description: str

@dataclass 
class CompositeSignal:
    """Combined signal from all indicators"""
    direction: str  # 'long', 'short', 'neutral'
    total_score: float  # 0-100
    strength: SignalStrength
    individual_signals: List[IndicatorSignal]
    confidence: float  # 0-1 based on agreement
    timestamp: pd.Timestamp

class SuperchargedIndicators:
    """Enhanced indicators with Stochastic Oscillator and scoring system"""
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        
    def _default_config(self) -> Dict:
        """Default indicator configuration"""
        return {
            # Moving Averages
            'sma_short': 20,
            'sma_long': 50,
            'ema_fast': 12,
            'ema_slow': 26,
            
            # RSI
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'rsi_neutral_low': 45,
            'rsi_neutral_high': 55,
            
            # MACD
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            
            # Bollinger Bands
            'bb_period': 20,
            'bb_std': 2,
            
            # Stochastic Oscillator - THE NEW HOTNESS! 🔥
            'stoch_k_period': 14,
            'stoch_d_period': 3,
            'stoch_oversold': 20,
            'stoch_overbought': 80,
            'stoch_neutral_low': 40,
            'stoch_neutral_high': 60,
            
            # ATR
            'atr_period': 14,
            'atr_stop_multiplier': 2.0,
            
            # Volume
            'volume_ma_period': 20,
            'volume_threshold': 1.5,
            
            # Signal weights (how much each indicator contributes to final score)
            'weights': {
                'trend': 0.25,      # MA trend alignment
                'momentum': 0.25,   # RSI + Stochastic
                'macd': 0.20,       # MACD signals
                'bollinger': 0.15,  # BB position
                'volume': 0.15      # Volume confirmation
            }
        }
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate ALL indicators including our new Stochastic beast! 🐙"""
        
        # === TREND INDICATORS ===
        df['sma_20'] = ta.trend.sma_indicator(df['close'], window=self.config['sma_short'])
        df['sma_50'] = ta.trend.sma_indicator(df['close'], window=self.config['sma_long'])
        df['ema_12'] = ta.trend.ema_indicator(df['close'], window=self.config['ema_fast'])
        df['ema_26'] = ta.trend.ema_indicator(df['close'], window=self.config['ema_slow'])
        
        # === MOMENTUM INDICATORS ===
        # RSI
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=self.config['rsi_period']).rsi()
        
        # 🚀 STOCHASTIC OSCILLATOR - THE STAR OF THE SHOW! 🚀
        stoch = ta.momentum.StochasticOscillator(
            df['high'], 
            df['low'], 
            df['close'], 
            window=self.config['stoch_k_period'],
            smooth_window=self.config['stoch_d_period']
        )
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # Stochastic signals
        df['stoch_oversold'] = df['stoch_k'] < self.config['stoch_oversold']
        df['stoch_overbought'] = df['stoch_k'] > self.config['stoch_overbought']
        df['stoch_bullish_cross'] = (df['stoch_k'] > df['stoch_d']) & (df['stoch_k'].shift(1) <= df['stoch_d'].shift(1))
        df['stoch_bearish_cross'] = (df['stoch_k'] < df['stoch_d']) & (df['stoch_k'].shift(1) >= df['stoch_d'].shift(1))
        
        # === MACD ===
        macd = ta.trend.MACD(df['close'], window_fast=self.config['macd_fast'], 
                            window_slow=self.config['macd_slow'], window_sign=self.config['macd_signal'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        df['macd_bullish'] = df['macd'] > df['macd_signal']
        df['macd_cross_up'] = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        df['macd_cross_down'] = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
        
        # === BOLLINGER BANDS ===
        bb = ta.volatility.BollingerBands(df['close'], window=self.config['bb_period'], window_dev=self.config['bb_std'])
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # === VOLUME ===
        df['volume_sma'] = df['volume'].rolling(window=self.config['volume_ma_period']).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # === ATR for stops ===
        df['atr'] = ta.volatility.AverageTrueRange(
            df['high'], df['low'], df['close'], window=self.config['atr_period']
        ).average_true_range()
        
        return df
    
    def analyze_trend_signals(self, row: pd.Series) -> IndicatorSignal:
        """Analyze trend alignment signals"""
        score = 0
        signals = []
        
        # Price vs MAs
        if row['close'] > row['sma_20']:
            score += 25
            signals.append("Price > SMA20")
        elif row['close'] < row['sma_20']:
            score -= 25
            signals.append("Price < SMA20")
            
        # MA alignment
        if row['sma_20'] > row['sma_50']:
            score += 25
            signals.append("SMA20 > SMA50")
        elif row['sma_20'] < row['sma_50']:
            score -= 25
            signals.append("SMA20 < SMA50")
            
        # EMA momentum
        if row['ema_12'] > row['ema_26']:
            score += 25
            signals.append("EMA bullish")
        elif row['ema_12'] < row['ema_26']:
            score -= 25
            signals.append("EMA bearish")
            
        # Trend strength
        ma_separation = abs(row['sma_20'] - row['sma_50']) / row['close']
        if ma_separation > 0.02:  # 2% separation = strong trend
            score += 25 if score > 0 else -25
            signals.append("Strong trend")
        
        # Convert to 0-100 scale and determine direction
        final_score = max(0, min(100, 50 + score))
        direction = 'long' if score > 0 else 'short' if score < 0 else 'neutral'
        
        return IndicatorSignal(
            name="Trend",
            signal=direction,
            strength=final_score,
            value=score,
            description=" | ".join(signals) if signals else "No clear trend"
        )
    
    def analyze_momentum_signals(self, row: pd.Series) -> IndicatorSignal:
        """Analyze RSI + Stochastic momentum signals - THE DYNAMIC DUO! 💪"""
        score = 0
        signals = []
        
        # === RSI Analysis ===
        rsi = row['rsi']
        if rsi > self.config['rsi_neutral_high']:
            if rsi < self.config['rsi_overbought']:
                score += 20  # Bullish but not overbought
                signals.append(f"RSI bullish ({rsi:.1f})")
            else:
                score += 5   # Overbought territory
                signals.append(f"RSI overbought ({rsi:.1f})")
        elif rsi < self.config['rsi_neutral_low']:
            if rsi > self.config['rsi_oversold']:
                score -= 20  # Bearish but not oversold
                signals.append(f"RSI bearish ({rsi:.1f})")
            else:
                score -= 5   # Oversold - potential reversal
                signals.append(f"RSI oversold ({rsi:.1f})")
        
        # === 🚀 STOCHASTIC OSCILLATOR MAGIC! 🚀 ===
        stoch_k = row['stoch_k']
        stoch_d = row['stoch_d']
        
        # Stochastic position scoring
        if stoch_k > self.config['stoch_neutral_high']:
            if stoch_k < self.config['stoch_overbought']:
                score += 20  # Bullish momentum
                signals.append(f"Stoch bullish ({stoch_k:.1f})")
            else:
                score += 5   # Overbought
                signals.append(f"Stoch overbought ({stoch_k:.1f})")
        elif stoch_k < self.config['stoch_neutral_low']:
            if stoch_k > self.config['stoch_oversold']:
                score -= 20  # Bearish momentum
                signals.append(f"Stoch bearish ({stoch_k:.1f})")
            else:
                score -= 5   # Oversold
                signals.append(f"Stoch oversold ({stoch_k:.1f})")
        
        # Stochastic crossover signals - PURE GOLD! ✨
        if row['stoch_bullish_cross']:
            score += 15
            signals.append("Stoch bullish cross! 🚀")
        elif row['stoch_bearish_cross']:
            score -= 15
            signals.append("Stoch bearish cross 📉")
        
        # K vs D relationship
        if stoch_k > stoch_d:
            score += 10
            signals.append("Stoch %K > %D")
        else:
            score -= 10
            signals.append("Stoch %K < %D")
        
        # RSI-Stochastic divergence detection
        rsi_bullish = rsi > 50
        stoch_bullish = stoch_k > 50
        if rsi_bullish == stoch_bullish:
            score += 10  # Agreement bonus
            signals.append("RSI-Stoch agreement")
        else:
            signals.append("RSI-Stoch divergence")
        
        # Convert to 0-100 scale
        final_score = max(0, min(100, 50 + score))
        direction = 'long' if score > 10 else 'short' if score < -10 else 'neutral'
        
        return IndicatorSignal(
            name="Momentum",
            signal=direction,
            strength=final_score,
            value=score,
            description=" | ".join(signals) if signals else "Neutral momentum"
        )
    
    def analyze_macd_signals(self, row: pd.Series) -> IndicatorSignal:
        """Analyze MACD signals"""
        score = 0
        signals = []
        
        # MACD vs Signal line
        if row['macd_bullish']:
            score += 30
            signals.append("MACD > Signal")
        else:
            score -= 30
            signals.append("MACD < Signal")
        
        # MACD crossovers
        if row['macd_cross_up']:
            score += 20
            signals.append("MACD bullish cross! 🎯")
        elif row['macd_cross_down']:
            score -= 20
            signals.append("MACD bearish cross")
        
        # MACD histogram
        if row['macd_diff'] > 0:
            score += 15
            signals.append("MACD histogram positive")
        else:
            score -= 15
            signals.append("MACD histogram negative")
        
        # MACD momentum
        macd_momentum = row['macd'] - row.get('macd_prev', row['macd'])
        if abs(macd_momentum) > 0:
            if macd_momentum > 0:
                score += 10
                signals.append("MACD accelerating")
            else:
                score -= 10
                signals.append("MACD decelerating")
        
        final_score = max(0, min(100, 50 + score))
        direction = 'long' if score > 0 else 'short' if score < 0 else 'neutral'
        
        return IndicatorSignal(
            name="MACD",
            signal=direction,
            strength=final_score,
            value=row['macd'],
            description=" | ".join(signals)
        )
    
    def analyze_bollinger_signals(self, row: pd.Series) -> IndicatorSignal:
        """Analyze Bollinger Band signals"""
        score = 0
        signals = []
        
        bb_pos = row['bb_position']
        
        if bb_pos > 0.8:
            score -= 15  # Near upper band - potential reversal
            signals.append("Near BB upper")
        elif bb_pos > 0.6:
            score += 20  # Strong but not extreme
            signals.append("BB bullish zone")
        elif bb_pos > 0.4:
            score += 5   # Neutral
            signals.append("BB neutral")
        elif bb_pos > 0.2:
            score -= 20  # Bearish zone
            signals.append("BB bearish zone")
        else:
            score += 15  # Near lower band - potential bounce
            signals.append("Near BB lower")
        
        # BB squeeze detection
        bb_width = (row['bb_upper'] - row['bb_lower']) / row['bb_middle']
        if bb_width < 0.04:  # Tight bands = volatility breakout coming
            score += 10
            signals.append("BB squeeze - breakout pending! ⚡")
        
        final_score = max(0, min(100, 50 + score))
        direction = 'long' if score > 0 else 'short' if score < 0 else 'neutral'
        
        return IndicatorSignal(
            name="Bollinger",
            signal=direction,
            strength=final_score,
            value=bb_pos,
            description=" | ".join(signals)
        )
    
    def analyze_volume_signals(self, row: pd.Series) -> IndicatorSignal:
        """Analyze volume confirmation"""
        score = 0
        signals = []
        
        vol_ratio = row['volume_ratio']
        
        if vol_ratio > 2.0:
            score += 30  # Massive volume
            signals.append(f"Explosive volume ({vol_ratio:.1f}x)")
        elif vol_ratio > self.config['volume_threshold']:
            score += 20  # Above average volume
            signals.append(f"High volume ({vol_ratio:.1f}x)")
        elif vol_ratio > 1.0:
            score += 10  # Normal volume
            signals.append("Normal volume")
        else:
            score -= 10  # Low volume
            signals.append("Low volume")
        
        final_score = max(0, min(100, 50 + score))
        
        return IndicatorSignal(
            name="Volume",
            signal='long' if score > 0 else 'neutral',  # Volume doesn't give direction, just confirmation
            strength=final_score,
            value=vol_ratio,
            description=" | ".join(signals)
        )
    
    def generate_composite_signal(self, row: pd.Series) -> CompositeSignal:
        """Generate the ULTIMATE composite signal! 🏆"""
        
        # Get individual signals
        trend_sig = self.analyze_trend_signals(row)
        momentum_sig = self.analyze_momentum_signals(row)
        macd_sig = self.analyze_macd_signals(row)
        bb_sig = self.analyze_bollinger_signals(row)
        volume_sig = self.analyze_volume_signals(row)
        
        signals = [trend_sig, momentum_sig, macd_sig, bb_sig, volume_sig]
        
        # Calculate weighted score
        weights = self.config['weights']
        total_score = (
            trend_sig.strength * weights['trend'] +
            momentum_sig.strength * weights['momentum'] +
            macd_sig.strength * weights['macd'] +
            bb_sig.strength * weights['bollinger'] +
            volume_sig.strength * weights['volume']
        )
        
        # Determine direction based on individual signals
        long_votes = sum(1 for sig in signals[:4] if sig.signal == 'long')  # Exclude volume from direction
        short_votes = sum(1 for sig in signals[:4] if sig.signal == 'short')
        
        if long_votes >= 3:
            direction = 'long'
        elif short_votes >= 3:
            direction = 'short'
        else:
            direction = 'neutral'
        
        # Calculate confidence based on agreement
        max_votes = max(long_votes, short_votes)
        confidence = max_votes / 4  # 4 directional indicators
        
        # Determine signal strength
        if total_score >= 90:
            strength = SignalStrength.LEGENDARY
        elif total_score >= 80:
            strength = SignalStrength.STRONG
        elif total_score >= 60:
            strength = SignalStrength.MEDIUM
        elif total_score >= 40:
            strength = SignalStrength.WEAK
        else:
            strength = SignalStrength.NONE
        
        return CompositeSignal(
            direction=direction,
            total_score=total_score,
            strength=strength,
            individual_signals=signals,
            confidence=confidence,
            timestamp=row.name
        )
    
    def analyze_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze entire dataframe and add signal columns"""
        
        # Calculate all indicators first
        df = self.calculate_all_indicators(df)
        
        # Initialize signal columns
        df['signal_direction'] = 'neutral'
        df['signal_score'] = 0.0
        df['signal_strength'] = 'NONE'
        df['signal_confidence'] = 0.0
        
        # Generate signals for each row (skip first 50 for indicator warmup)
        for i in range(50, len(df)):
            row = df.iloc[i]
            composite = self.generate_composite_signal(row)
            
            df.iloc[i, df.columns.get_loc('signal_direction')] = composite.direction
            df.iloc[i, df.columns.get_loc('signal_score')] = composite.total_score
            df.iloc[i, df.columns.get_loc('signal_strength')] = composite.strength.name
            df.iloc[i, df.columns.get_loc('signal_confidence')] = composite.confidence
        
        return df
    
    def get_latest_signal(self, df: pd.DataFrame) -> Optional[CompositeSignal]:
        """Get the latest composite signal from dataframe"""
        if len(df) < 51:  # Need enough data for indicators
            return None
            
        latest_row = df.iloc[-1]
        return self.generate_composite_signal(latest_row)
    
    def print_signal_breakdown(self, signal: CompositeSignal):
        """Print a beautiful breakdown of the signal"""
        print(f"\n🎯 SIGNAL BREAKDOWN - {signal.timestamp}")
        print("=" * 60)
        print(f"🚀 Direction: {signal.direction.upper()}")
        print(f"📊 Total Score: {signal.total_score:.1f}/100")
        print(f"💪 Strength: {signal.strength.name}")
        print(f"🎯 Confidence: {signal.confidence:.1%}")
        print("\n📈 Individual Indicators:")
        print("-" * 40)
        
        for sig in signal.individual_signals:
            emoji = "🟢" if sig.signal == 'long' else "🔴" if sig.signal == 'short' else "⚪"
            print(f"{emoji} {sig.name:<12} {sig.strength:>5.1f}/100  {sig.description}")
        
        print("=" * 60)


# Quick test function
def test_enhanced_indicators():
    """Test our enhanced indicators with sample data"""
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=100, freq='1H')
    np.random.seed(42)
    
    # Generate realistic price data
    base_price = 50000
    returns = np.random.normal(0, 0.02, 100)
    prices = [base_price]
    
    for r in returns[1:]:
        prices.append(prices[-1] * (1 + r))
    
    df = pd.DataFrame({
        'open': [p * 0.999 for p in prices],
        'high': [p * 1.002 for p in prices],
        'low': [p * 0.998 for p in prices],
        'close': prices,
        'volume': np.random.uniform(1000, 5000, 100)
    }, index=dates)
    
    # Test our enhanced indicators
    indicators = SuperchargedIndicators()
    df = indicators.analyze_dataframe(df)
    
    # Get latest signal
    latest_signal = indicators.get_latest_signal(df)
    if latest_signal:
        indicators.print_signal_breakdown(latest_signal)
    
    return df, latest_signal

if __name__ == "__main__":
    print("🐙 Testing Enhanced Indicators with Stochastic Power!")
    test_enhanced_indicators()