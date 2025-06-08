#!/usr/bin/env python3
"""
TradeMonkey Lite - Enhanced Backtesting Framework
"In backtesting we trust, in live trading we profit!" - Ancient Trader Wisdom
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import asyncio
import ccxt
from dataclasses import dataclass
import ta

@dataclass
class BacktestResult:
    """Results from a backtest run"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    profit_factor: float
    avg_trade_return: float
    best_trade: float
    worst_trade: float
    start_date: datetime
    end_date: datetime
    strategy_name: str


class EnhancedBacktester:
    """Fast backtesting engine for rapid indicator evaluation"""
    
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.commission = 0.001  # 0.1% commission
        self.slippage = 0.0005   # 0.05% slippage
        
    async def fetch_historical_data(self, symbol: str, timeframe: str, 
                                  start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch historical OHLCV data"""
        try:
            # Use paper trading exchange for data fetching
            exchange = ccxt.kraken({
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'}
            })
            
            # Convert dates to timestamps
            since = int(start_date.timestamp() * 1000)
            
            all_data = []
            current_since = since
            end_timestamp = int(end_date.timestamp() * 1000)
            
            while current_since < end_timestamp:
                ohlcv = await asyncio.to_thread(
                    exchange.fetch_ohlcv, 
                    symbol, 
                    timeframe, 
                    since=current_since,
                    limit=1000
                )
                
                if not ohlcv:
                    break
                    
                all_data.extend(ohlcv)
                current_since = ohlcv[-1][0] + 1  # Next candle
                
                # Prevent rate limiting
                await asyncio.sleep(0.1)
            
            # Convert to DataFrame
            df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            
            return df
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            # Return sample data for testing
            return self._generate_sample_data(start_date, end_date)
    
    def _generate_sample_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Generate sample OHLCV data for testing"""
        periods = int((end_date - start_date).total_seconds() / 3600)  # Hourly data
        dates = pd.date_range(start=start_date, end=end_date, periods=periods)
        
        # Generate realistic crypto price action
        np.random.seed(42)
        base_price = 50000
        returns = np.random.normal(0, 0.02, len(dates))  # 2% volatility
        
        prices = [base_price]
        for r in returns[1:]:
            prices.append(prices[-1] * (1 + r))
        
        # Create OHLCV from price series
        data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            volatility = abs(np.random.normal(0, 0.01))
            high = price * (1 + volatility)
            low = price * (1 - volatility)
            open_price = prices[i-1] if i > 0 else price
            volume = np.random.uniform(1000, 10000)
            
            data.append({
                'open': open_price,
                'high': high,
                'low': low,
                'close': price,
                'volume': volume
            })
        
        df = pd.DataFrame(data, index=dates)
        return df
    
    def calculate_base_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate our existing indicators"""
        # Moving averages
        df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
        df['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
        df['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)
        
        # RSI
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['close'], window=20)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()
        
        # Volume
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # ATR
        df['atr'] = ta.volatility.AverageTrueRange(
            df['high'], df['low'], df['close'], window=14
        ).average_true_range()
        
        return df
    
    def calculate_stochastic(self, df: pd.DataFrame, k_period: int = 14, 
                           d_period: int = 3) -> pd.DataFrame:
        """Calculate Stochastic Oscillator"""
        stoch = ta.momentum.StochasticOscillator(
            df['high'], df['low'], df['close'], 
            window=k_period, smooth_window=d_period
        )
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        return df
    
    def calculate_williams_r(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Williams %R"""
        df['williams_r'] = ta.momentum.WilliamsRIndicator(
            df['high'], df['low'], df['close'], lbp=period
        ).williams_r()
        return df
    
    def generate_base_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate signals using our current logic"""
        df['signal'] = 0  # 0 = hold, 1 = long, -1 = short
        
        for i in range(50, len(df)):  # Start after indicators are valid
            row = df.iloc[i]
            
            # Long conditions
            long_conditions = [
                row['close'] > row['sma_20'],
                row['sma_20'] > row['sma_50'], 
                row['rsi'] > 50 and row['rsi'] < 70,
                row['macd'] > row['macd_signal'],
                row['close'] > row['bb_middle'],
                row['volume_ratio'] > 1.5  # Volume confirmation
            ]
            
            # Short conditions  
            short_conditions = [
                row['close'] < row['sma_20'],
                row['sma_20'] < row['sma_50'],
                row['rsi'] < 50 and row['rsi'] > 30,
                row['macd'] < row['macd_signal'],
                row['close'] < row['bb_middle'],
                row['volume_ratio'] > 1.5  # Volume confirmation
            ]
            
            if sum(long_conditions) >= 5:
                df.iloc[i, df.columns.get_loc('signal')] = 1
            elif sum(short_conditions) >= 5:
                df.iloc[i, df.columns.get_loc('signal')] = -1
        
        return df
    
    def generate_enhanced_signals(self, df: pd.DataFrame, use_stochastic: bool = False,
                                use_williams_r: bool = False) -> pd.DataFrame:
        """Generate enhanced signals with new indicators"""
        df['enhanced_signal'] = 0
        
        for i in range(50, len(df)):
            row = df.iloc[i]
            
            # Base long conditions
            long_conditions = [
                row['close'] > row['sma_20'],
                row['sma_20'] > row['sma_50'],
                row['rsi'] > 50 and row['rsi'] < 70,
                row['macd'] > row['macd_signal'],
                row['close'] > row['bb_middle'],
                row['volume_ratio'] > 1.5
            ]
            
            # Enhanced conditions
            if use_stochastic and 'stoch_k' in df.columns:
                # Stochastic oversold bounce
                stoch_long = (row['stoch_k'] > 20 and row['stoch_k'] > row['stoch_d'])
                long_conditions.append(stoch_long)
            
            if use_williams_r and 'williams_r' in df.columns:
                # Williams %R oversold bounce
                williams_long = row['williams_r'] > -80 and row['williams_r'] < -20
                long_conditions.append(williams_long)
            
            # Base short conditions
            short_conditions = [
                row['close'] < row['sma_20'],
                row['sma_20'] < row['sma_50'],
                row['rsi'] < 50 and row['rsi'] > 30,
                row['macd'] < row['macd_signal'],
                row['close'] < row['bb_middle'],
                row['volume_ratio'] > 1.5
            ]
            
            # Enhanced short conditions
            if use_stochastic and 'stoch_k' in df.columns:
                stoch_short = (row['stoch_k'] < 80 and row['stoch_k'] < row['stoch_d'])
                short_conditions.append(stoch_short)
            
            if use_williams_r and 'williams_r' in df.columns:
                williams_short = row['williams_r'] < -20 and row['williams_r'] > -80
                short_conditions.append(williams_short)
            
            # Signal generation
            required_conditions = 5 + (1 if use_stochastic else 0) + (1 if use_williams_r else 0)
            
            if sum(long_conditions) >= required_conditions:
                df.iloc[i, df.columns.get_loc('enhanced_signal')] = 1
            elif sum(short_conditions) >= required_conditions:
                df.iloc[i, df.columns.get_loc('enhanced_signal')] = -1
        
        return df
    
    def generate_supercharged_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate signals using our LEGENDARY composite scoring system! 🚀"""
        from enhanced_indicators_stochastic import SuperchargedIndicators
        
        # Initialize our enhanced indicators
        indicators = SuperchargedIndicators()
        
        # Calculate all indicators and signals
        df = indicators.analyze_dataframe(df)
        
        # Convert to our signal format
        df['supercharged_signal'] = 0
        
        # Use the composite signals - only trade STRONG+ signals with high confidence
        for i in range(len(df)):
            direction = df.iloc[i]['signal_direction']
            score = df.iloc[i]['signal_score']
            confidence = df.iloc[i]['signal_confidence']
            
            # Only take trades with:
            # - Score >= 75 (STRONG signals)
            # - Confidence >= 0.6 (60%+ indicator agreement)
            if score >= 75 and confidence >= 0.6:
                if direction == 'long':
                    df.iloc[i, df.columns.get_loc('supercharged_signal')] = 1
                elif direction == 'short':
                    df.iloc[i, df.columns.get_loc('supercharged_signal')] = -1
        
        return df
    
    def simulate_trades(self, df: pd.DataFrame, signal_column: str = 'signal') -> BacktestResult:
        """Simulate trading based on signals"""
        capital = self.initial_capital
        position = 0  # 0 = no position, 1 = long, -1 = short
        entry_price = 0
        trades = []
        equity_curve = [capital]
        
        for i in range(len(df)):
            row = df.iloc[i]
            current_price = row['close']
            signal = row[signal_column]
            
            # Close existing position if signal changes
            if position != 0 and signal != position:
                # Calculate trade result
                if position == 1:  # Close long
                    trade_return = (current_price - entry_price) / entry_price
                else:  # Close short
                    trade_return = (entry_price - current_price) / entry_price
                
                # Apply costs
                trade_return -= (self.commission + self.slippage)
                
                # Update capital
                trade_size = capital * 0.25  # 25% position size
                profit = trade_size * trade_return
                capital += profit
                
                trades.append({
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'side': 'long' if position == 1 else 'short',
                    'return': trade_return,
                    'profit': profit,
                    'timestamp': row.name
                })
                
                position = 0
            
            # Open new position
            if position == 0 and signal != 0:
                position = signal
                entry_price = current_price
            
            equity_curve.append(capital)
        
        # Calculate performance metrics
        if not trades:
            return BacktestResult(
                total_trades=0, winning_trades=0, losing_trades=0,
                win_rate=0, total_return=0, max_drawdown=0,
                sharpe_ratio=0, profit_factor=0, avg_trade_return=0,
                best_trade=0, worst_trade=0,
                start_date=df.index[0], end_date=df.index[-1],
                strategy_name=signal_column
            )
        
        winning_trades = len([t for t in trades if t['return'] > 0])
        losing_trades = len([t for t in trades if t['return'] <= 0])
        win_rate = winning_trades / len(trades) if trades else 0
        
        total_return = (capital - self.initial_capital) / self.initial_capital
        
        # Calculate max drawdown
        peak = self.initial_capital
        max_dd = 0
        for equity in equity_curve:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak
            max_dd = max(max_dd, dd)
        
        # Calculate other metrics
        returns = [t['return'] for t in trades]
        avg_return = np.mean(returns) if returns else 0
        best_trade = max(returns) if returns else 0
        worst_trade = min(returns) if returns else 0
        
        # Sharpe ratio (simplified)
        sharpe = np.mean(returns) / np.std(returns) if len(returns) > 1 and np.std(returns) > 0 else 0
        
        # Profit factor
        gross_profit = sum([t['return'] for t in trades if t['return'] > 0])
        gross_loss = abs(sum([t['return'] for t in trades if t['return'] < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return BacktestResult(
            total_trades=len(trades),
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_return=total_return,
            max_drawdown=max_dd,
            sharpe_ratio=sharpe,
            profit_factor=profit_factor,
            avg_trade_return=avg_return,
            best_trade=best_trade,
            worst_trade=worst_trade,
            start_date=df.index[0],
            end_date=df.index[-1],
            strategy_name=signal_column
        )
    
    async def compare_strategies(self, symbol: str = "BTC/USD", 
                               days_back: int = 90) -> Dict[str, BacktestResult]:
        """Compare base strategy vs enhanced strategies"""
        
        # Fetch data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        print(f"📊 Fetching {days_back} days of {symbol} data...")
        df = await self.fetch_historical_data(symbol, '1h', start_date, end_date)
        
        # Calculate all indicators
        print("🔧 Calculating indicators...")
        df = self.calculate_base_indicators(df)
        df = self.calculate_stochastic(df)
        df = self.calculate_williams_r(df)
        
        # Generate different signal types
        print("🎯 Generating signals...")
        df = self.generate_base_signals(df)
        
        # Enhanced signals with individual indicators
        df = self.generate_enhanced_signals(df, use_stochastic=True, use_williams_r=False)
        df['stoch_signals'] = df['enhanced_signal']
        
        df = self.generate_enhanced_signals(df, use_stochastic=False, use_williams_r=True)
        df['williams_signals'] = df['enhanced_signal']
        
        df = self.generate_enhanced_signals(df, use_stochastic=True, use_williams_r=True)
        df['combined_signals'] = df['enhanced_signal']
        
        # 🚀 THE SUPERCHARGED SIGNALS - OUR SECRET WEAPON! 🚀
        df = self.generate_supercharged_signals(df)
        
        # Run backtests
        print("🚀 Running backtests...")
        results = {}
        
        strategies = {
            'Base Strategy': 'signal',
            'Base + Stochastic': 'stoch_signals',
            'Base + Williams %R': 'williams_signals',
            'Base + Both': 'combined_signals',
            '🚀 SUPERCHARGED 🚀': 'supercharged_signal'  # THE BEAST!
        }
        
        for name, column in strategies.items():
            if column in df.columns:
                result = self.simulate_trades(df, column)
                result.strategy_name = name
                results[name] = result
        
        return results
    
    def print_comparison(self, results: Dict[str, BacktestResult]):
        """Print a beautiful comparison of strategies"""
        print("\n" + "🎯" * 25)
        print("📊 STOCHASTIC OSCILLATOR BATTLE ROYALE! 📊")
        print("🎯" * 25)
        
        print(f"{'Strategy':<25} {'Trades':<8} {'Win Rate':<10} {'Return':<10} {'Sharpe':<8} {'Max DD':<8}")
        print("-" * 80)
        
        best_return = -float('inf')
        best_strategy = ""
        
        # Sort by return for dramatic effect
        sorted_results = sorted(results.items(), key=lambda x: x[1].total_return, reverse=True)
        
        for name, result in sorted_results:
            win_rate_str = f"{result.win_rate:.1%}"
            return_str = f"{result.total_return:.1%}"
            sharpe_str = f"{result.sharpe_ratio:.2f}"
            dd_str = f"{result.max_drawdown:.1%}"
            
            # Add emoji based on performance
            emoji = "🏆" if name == sorted_results[0][0] else "🥈" if name == sorted_results[1][0] else "📈" if result.total_return > 0 else "📉"
            
            print(f"{emoji} {name:<23} {result.total_trades:<8} {win_rate_str:<10} {return_str:<10} {sharpe_str:<8} {dd_str:<8}")
            
            if result.total_return > best_return:
                best_return = result.total_return
                best_strategy = name
        
        print("-" * 80)
        print(f"🏆 CHAMPION: {best_strategy} with {best_return:.1%} return!")
        
        if '🚀 SUPERCHARGED 🚀' in results:
            supercharged = results['🚀 SUPERCHARGED 🚀']
            print(f"🚀 SUPERCHARGED STATS:")
            print(f"   💎 Total Trades: {supercharged.total_trades}")
            print(f"   🎯 Win Rate: {supercharged.win_rate:.1%}")
            print(f"   📈 Best Trade: {supercharged.best_trade:.1%}")
            print(f"   📉 Worst Trade: {supercharged.worst_trade:.1%}")
            print(f"   ⚖️ Profit Factor: {supercharged.profit_factor:.2f}")
        
        print("🎯" * 25)


# Quick usage example
async def quick_test():
    """Quick test of the backtesting framework"""
    backtester = EnhancedBacktester(initial_capital=10000)
    results = await backtester.compare_strategies(symbol="BTC/USD", days_back=30)
    backtester.print_comparison(results)

if __name__ == "__main__":
    print("🐙 TradeMonkey Backtesting Engine - Ready for SPEED TESTING!")
    asyncio.run(quick_test())