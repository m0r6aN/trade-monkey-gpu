#!/usr/bin/env python3
"""
TradeMonkey Lite - Backtesting Engine
"One engine to rule them all, and in the caching bind them!" 💍⚡

Just pure, concentrated backtesting power with:
- Smart data caching (memory + disk)
- Kraken-friendly rate limiting
- Proper metric calculations
- Clean, simple architecture
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import ta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import logging
from enum import Enum
import warnings
import pickle
import hashlib
from pathlib import Path
warnings.filterwarnings('ignore')

# Setup logging with style
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('BacktestingEngine')

class MarketCondition(Enum):
    """Market condition classification"""
    BULL_RUN = "Bull Run"
    BEAR_MARKET = "Bear Market" 
    SIDEWAYS = "Sideways Chop"
    VOLATILE = "High Volatility"
    UNKNOWN = "Unknown"

@dataclass
class TestPeriod:
    """Define a testing period with context"""
    name: str
    start_date: str
    end_date: str
    expected_condition: MarketCondition
    description: str

@dataclass
class BacktestConfig:
    """Configuration for a single backtest run"""
    symbol: str
    timeframe: str
    test_period: TestPeriod
    strategy_name: str
    signal_threshold: int = 75
    position_size_pct: float = 0.25
    use_stop_loss: bool = True
    use_take_profit: bool = True
    leverage: float = 2.0
    atr_stop_multiplier: float = 2.0
    atr_profit_multiplier: float = 3.0

@dataclass
class Trade:
    """Individual trade record"""
    entry_time: datetime
    exit_time: datetime
    side: str  # 'long' or 'short'
    entry_price: float
    exit_price: float
    size: float
    leverage: float
    gross_profit: float
    fees_paid: float
    net_profit: float
    return_pct: float
    duration_hours: float
    exit_reason: str

@dataclass
class EquityPoint:
    """Point in equity curve"""
    timestamp: datetime
    capital: float
    drawdown_pct: float
    peak_capital: float

@dataclass
class BacktestResult:
    """Complete backtest results with proper metrics"""
    config: BacktestConfig
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    
    # Return metrics
    total_return_pct: float
    total_profit_usd: float
    cagr: float
    
    # Risk metrics - PROPERLY CALCULATED!
    max_drawdown_pct: float
    max_drawdown_duration_days: float
    volatility_annualized: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Performance metrics
    profit_factor: float
    avg_trade_return_pct: float
    best_trade_pct: float
    worst_trade_pct: float
    avg_win_pct: float
    avg_loss_pct: float
    largest_winning_streak: int
    largest_losing_streak: int
    
    # Trading frequency
    trades_per_week: float
    avg_trade_duration_hours: float
    
    # Benchmark comparison
    market_return_pct: float
    alpha: float
    beta: float
    
    # Cost analysis
    total_fees_paid: float
    fees_as_pct_of_profit: float
    
    # Timing
    start_date: datetime
    end_date: datetime
    trading_days: int
    
    # Data for visualization
    equity_curve: List[EquityPoint]
    trades: List[Trade]

class DataCache:
    """Smart data caching system - stores RAW OHLCV data only"""
    
    def __init__(self, cache_dir: str = "data_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.memory_cache: Dict[str, pd.DataFrame] = {}
        
        logger.info(f"🗄️ Data cache initialized at {self.cache_dir}")
    
    def _get_cache_key(self, symbol: str, timeframe: str, test_period: TestPeriod) -> str:
        """Generate a unique cache key"""
        key_str = f"{symbol}_{timeframe}_{test_period.name}_{test_period.start_date}_{test_period.end_date}"
        return hashlib.md5(key_str.encode()).hexdigest()[:16]
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get the disk cache file path"""
        return self.cache_dir / f"{cache_key}.parquet"
    
    async def get_raw_data(self, symbol: str, timeframe: str, test_period: TestPeriod, 
                          fetcher_func) -> Optional[pd.DataFrame]:
        """Get RAW OHLCV data from cache or fetch if not available"""
        
        cache_key = self._get_cache_key(symbol, timeframe, test_period)
        
        # Try memory cache first
        if cache_key in self.memory_cache:
            logger.info(f"🎯 MEMORY CACHE HIT: {symbol} {timeframe} {test_period.name}")
            return self.memory_cache[cache_key].copy()
        
        # Try disk cache
        cache_path = self._get_cache_path(cache_key)
        if cache_path.exists():
            try:
                df = pd.read_parquet(cache_path)
                # Restore datetime index
                if 'timestamp' in df.columns:
                    df = df.set_index('timestamp')
                elif not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                
                # Store in memory cache for faster access
                self.memory_cache[cache_key] = df.copy()
                logger.info(f"💾 DISK CACHE HIT: {symbol} {timeframe} {test_period.name}")
                return df.copy()
            except Exception as e:
                logger.warning(f"Failed to load from disk cache: {e}")
        
        # Cache miss - need to fetch
        logger.info(f"🌐 CACHE MISS: Fetching {symbol} {timeframe} {test_period.name}")
        df = await fetcher_func(symbol, timeframe, test_period)
        
        if df is not None and not df.empty:
            # Store in both caches (RAW data only)
            self.memory_cache[cache_key] = df.copy()
            
            try:
                # Reset index for parquet compatibility
                df_to_save = df.reset_index()
                df_to_save.to_parquet(cache_path, compression='snappy')
                logger.info(f"💾 Saved RAW data to disk cache: {cache_path}")
            except Exception as e:
                logger.warning(f"Failed to save to disk cache: {e}")
        
        return df.copy() if df is not None else None
    
    def clear_cache(self):
        """Clear all cached data"""
        self.memory_cache.clear()
        if self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("*.parquet"):
                cache_file.unlink()
        logger.info("🗑️ Cache cleared")

class BacktestingEngine:
    """THE backtesting engine with all the power! 🚀"""
    
    def __init__(self, initial_capital: float = 10000, use_futures: bool = False):
        self.initial_capital = initial_capital
        self.use_futures = use_futures
        
        # Real Kraken fees
        if use_futures:
            self.taker_fee = 0.0005   # 0.05% Kraken futures taker
            self.maker_fee = 0.0002   # 0.02% Kraken futures maker  
        else:
            self.taker_fee = 0.0026   # 0.26% Kraken spot taker
            self.maker_fee = 0.0016   # 0.16% Kraken spot maker
            
        self.slippage_rate = 0.0005   # 0.05% realistic slippage
        self.risk_free_rate = 0.0     # Crypto assumption
        
        # Initialize exchange and cache
        self.exchange = self._initialize_exchange()
        self.data_cache = DataCache()
        
        # Rate limiting state (BacktestingEngine owns this!)
        self.api_call_count = 0
        self.last_api_call = datetime.now()
        self.min_api_interval = 2.0  # Minimum seconds between API calls
        
        # Define test periods
        self.test_periods = self._define_test_periods()
        
        logger.info("🚀 BacktestingEngine initialized!")
        logger.info(f"💰 Capital: ${initial_capital:,.2f}")
        logger.info(f"📊 Mode: {'Futures' if use_futures else 'Spot'}")
    
    def _initialize_exchange(self):
        """Initialize exchange with conservative rate limiting"""
        try:
            exchange = ccxt.kraken({
                'enableRateLimit': True,
                'rateLimit': 3000,  # Conservative 3 seconds
                'timeout': 30000,   # 30 second timeout (was missing!)
                'options': {'defaultType': 'spot'}
            })
            logger.info("✅ Initialized Kraken exchange")
            return exchange
        except Exception as e:
            logger.error(f"Failed to initialize exchange: {e}")
            raise
    
    def _define_test_periods(self) -> List[TestPeriod]:
        """Define test periods with ACTUAL available market data"""
        return [
            TestPeriod(
                name="Recent_3_Weeks",
                start_date="2025-05-01",
                end_date="2025-05-22",  # Data we know exists
                expected_condition=MarketCondition.VOLATILE,
                description="Recent market data (3 weeks)"
            ),
            TestPeriod(
                name="Last_Week",
                start_date="2025-05-24",
                end_date="2025-05-31",  # Very recent data
                expected_condition=MarketCondition.VOLATILE,
                description="Last week of data"
            ),
            TestPeriod(
                name="Mid_Month",
                start_date="2025-05-10",
                end_date="2025-05-20",  # Subset of available data
                expected_condition=MarketCondition.VOLATILE,
                description="Mid-month period"
            )
        ]
    
    async def _rate_limited_api_call(self, func, *args, **kwargs):
        """Make API call with proper rate limiting - BacktestingEngine manages this!"""
        # Ensure minimum time between calls
        time_since_last = (datetime.now() - self.last_api_call).total_seconds()
        if time_since_last < self.min_api_interval:
            sleep_time = self.min_api_interval - time_since_last
            logger.debug(f"⏱️ Rate limiting: sleeping {sleep_time:.2f}s")
            await asyncio.sleep(sleep_time)
        
        self.api_call_count += 1
        self.last_api_call = datetime.now()
        
        try:
            result = await asyncio.to_thread(func, *args, **kwargs)
            return result
        except Exception as e:
            if "Too many requests" in str(e):
                logger.warning("🚦 Rate limit hit, backing off...")
                await asyncio.sleep(30)  # Long backoff
                raise
            else:
                raise
    
    async def _fetch_raw_data_from_api(self, symbol: str, timeframe: str, test_period: TestPeriod) -> Optional[pd.DataFrame]:
        """Fetch RAW OHLCV data from API with rate limiting"""
        
        try:
            await self._rate_limited_api_call(self.exchange.load_markets)
            
            if symbol not in self.exchange.markets:
                logger.error(f"Symbol {symbol} not found")
                return None
            
            # Parse dates
            start_date = datetime.strptime(test_period.start_date, "%Y-%m-%d")
            end_date = datetime.strptime(test_period.end_date, "%Y-%m-%d")
            
            # Check for future dates
            if start_date > datetime.now():
                logger.warning(f"Period {test_period.name} is in future - skipping")
                return None
            
            # Fetch data conservatively
            since = int(start_date.timestamp() * 1000)
            end_timestamp = int(end_date.timestamp() * 1000)
            
            all_candles = []
            current_since = since
            max_requests = 15  # Very conservative
            request_count = 0
            
            while current_since < end_timestamp and request_count < max_requests:
                try:
                    ohlcv = await self._rate_limited_api_call(
                        self.exchange.fetch_ohlcv,
                        symbol,
                        timeframe,
                        since=current_since,
                        limit=500
                    )
                    
                    if not ohlcv:
                        break
                    
                    # Filter to date range
                    period_candles = [c for c in ohlcv if since <= c[0] <= end_timestamp]
                    all_candles.extend(period_candles)
                    
                    # Move to next batch
                    if ohlcv:
                        timeframe_ms = self._timeframe_to_milliseconds(timeframe)
                        current_since = ohlcv[-1][0] + timeframe_ms
                        request_count += 1
                        
                        # Extra polite to the API
                        await asyncio.sleep(2.0)
                    else:
                        break
                        
                except Exception as e:
                    if "Too many requests" in str(e):
                        logger.warning(f"Rate limit during batch {request_count}, long backoff...")
                        await asyncio.sleep(60)  # Very long backoff
                        continue
                    else:
                        logger.error(f"Error fetching batch {request_count}: {e}")
                        break
            
            if not all_candles:
                logger.error(f"No data available for {test_period.name}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            df = df[~df.index.duplicated(keep='first')].sort_index()
            
            logger.info(f"✅ Fetched {len(df)} RAW candles for {test_period.name}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch RAW data for {test_period.name}: {e}")
            return None
    
    def _timeframe_to_milliseconds(self, timeframe: str) -> int:
        """Convert timeframe string to milliseconds"""
        multipliers = {
            '1m': 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '30m': 30 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000
        }
        return multipliers.get(timeframe, 60 * 60 * 1000)
    
    async def get_data_with_indicators(self, symbol: str, timeframe: str, test_period: TestPeriod) -> Optional[pd.DataFrame]:
        """Get data with indicators calculated - the main data pipeline entry point"""
        
        # Get RAW data (cached or fetched)
        raw_df = await self.data_cache.get_raw_data(
            symbol, timeframe, test_period, self._fetch_raw_data_from_api
        )
        
        if raw_df is None or len(raw_df) < 100:
            logger.warning(f"Insufficient RAW data for {symbol} {timeframe} {test_period.name}")
            return None
        
        # Calculate indicators on the RAW data
        try:
            df_with_indicators = self.calculate_indicators(raw_df.copy())
            logger.info(f"✅ Calculated indicators for {symbol} {timeframe} {test_period.name}")
            return df_with_indicators
        except ValueError as e:
            logger.error(f"Indicator calculation failed: {e}")
            return None
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators - GOLD STANDARD implementation"""
        
        if len(df) < 100:
            raise ValueError(f"Need at least 100 candles, got {len(df)}")
        
        try:
            # Make a copy to avoid modifying original
            df = df.copy()
            
            # Trend indicators
            df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
            df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
            df['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
            df['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)
            
            # Momentum indicators
            df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
            
            # Stochastic Oscillator
            stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=14, smooth_window=3)
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()
            
            # MACD
            macd = ta.trend.MACD(df['close'], window_fast=12, window_slow=26, window_sign=9)
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(df['close'], window=20)
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_middle'] = bb.bollinger_mavg()
            df['bb_lower'] = bb.bollinger_lband()
            
            # Volume indicators
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # ATR for stop loss calculation
            df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
            
            # Remove NaN values
            df = df.dropna()
            
            logger.debug(f"✅ Calculated indicators. Clean data: {len(df)} candles")
            return df
            
        except Exception as e:
            logger.error(f"Indicator calculation failed: {e}")
            raise
    
    def generate_signals(self, df: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
        """Generate trading signals with configurable parameters"""
        
        # Make a copy to avoid modifying original
        df = df.copy()
        
        df['signal'] = 0
        df['signal_score'] = 0
        df['stop_loss'] = np.nan
        df['take_profit'] = np.nan
        
        for i in range(len(df)):
            row = df.iloc[i]
            score = 0
            
            # Handle NaN ATR values
            atr = row['atr']
            if pd.isna(atr) or atr <= 0:
                continue
            
            # Trend alignment (30 points)
            if row['close'] > row['sma_20'] > row['sma_50']:
                score += 30
            elif row['close'] < row['sma_20'] < row['sma_50']:
                score -= 30
            
            # RSI momentum (25 points)
            if 55 < row['rsi'] < 70:
                score += 25
            elif 30 < row['rsi'] < 45:
                score -= 25
            
            # Stochastic momentum (25 points)
            if 20 < row['stoch_k'] < 80:
                if row['stoch_k'] > row['stoch_d']:
                    score += 25 if row['stoch_k'] > 50 else 15
                else:
                    score -= 25 if row['stoch_k'] < 50 else 15
            
            # MACD confirmation (15 points)
            score += 15 if row['macd'] > row['macd_signal'] else -15
            
            # Volume confirmation (5 points)
            if row['volume_ratio'] > 2.0:
                score += 5
            elif row['volume_ratio'] > 1.5:
                score += 3
            
            df.iloc[i, df.columns.get_loc('signal_score')] = score
            
            # Generate signals based on threshold
            if score >= config.signal_threshold:
                df.iloc[i, df.columns.get_loc('signal')] = 1
                stop_loss = row['close'] - (atr * config.atr_stop_multiplier)
                take_profit = row['close'] + (atr * config.atr_profit_multiplier)
                df.iloc[i, df.columns.get_loc('stop_loss')] = stop_loss
                df.iloc[i, df.columns.get_loc('take_profit')] = take_profit
                
            elif score <= -config.signal_threshold:
                df.iloc[i, df.columns.get_loc('signal')] = -1
                stop_loss = row['close'] + (atr * config.atr_stop_multiplier)
                take_profit = row['close'] - (atr * config.atr_profit_multiplier)
                df.iloc[i, df.columns.get_loc('stop_loss')] = stop_loss
                df.iloc[i, df.columns.get_loc('take_profit')] = take_profit
        
        return df
    
    def simulate_trading(self, df: pd.DataFrame, config: BacktestConfig) -> BacktestResult:
        """PROPERLY simulate trading with all fixes applied - THE CROWN JEWEL! 👑"""
        
        logger.debug(f"💰 Simulating {config.strategy_name} with PROPER calculations...")
        
        # Initialize tracking variables
        capital = self.initial_capital
        equity_curve = [EquityPoint(df.index[0], capital, 0.0, capital)]
        trades = []
        
        # Position tracking
        in_position = False
        position_side = 0
        position_entry_price = 0
        position_entry_time = None
        position_size = 0
        position_leverage = config.leverage
        stop_loss_price = 0
        take_profit_price = 0
        
        # Peak tracking for drawdown
        peak_capital = capital
        
        for i in range(len(df)):
            row = df.iloc[i]
            current_price = row['close']
            current_time = row.name
            signal = row['signal']
            
            # Check exit conditions if in position
            if in_position:
                exit_reason = None
                exit_price = current_price
                
                # Check stop loss
                if config.use_stop_loss:
                    if (position_side == 1 and current_price <= stop_loss_price) or \
                       (position_side == -1 and current_price >= stop_loss_price):
                        exit_reason = 'stop_loss'
                        exit_price = stop_loss_price
                
                # Check take profit
                if not exit_reason and config.use_take_profit:
                    if (position_side == 1 and current_price >= take_profit_price) or \
                       (position_side == -1 and current_price <= take_profit_price):
                        exit_reason = 'take_profit'
                        exit_price = take_profit_price
                
                # Check signal exit
                if not exit_reason and (signal != position_side or signal == 0):
                    exit_reason = 'signal'
                    exit_price = current_price * (1 - self.slippage_rate if position_side == 1 else 1 + self.slippage_rate)
                
                # Execute exit
                if exit_reason:
                    # Calculate trade results
                    if position_side == 1:  # Long position
                        raw_return = (exit_price - position_entry_price) / position_entry_price
                    else:  # Short position  
                        raw_return = (position_entry_price - exit_price) / position_entry_price
                    
                    # Calculate profits with PROPER leverage
                    allocated_capital = capital * config.position_size_pct
                    
                    if self.use_futures:
                        # Futures: Apply leverage to P&L
                        gross_profit = allocated_capital * raw_return * position_leverage
                    else:
                        # Spot: No leverage
                        gross_profit = allocated_capital * raw_return
                    
                    # Calculate fees (entry + exit)
                    notional_value = allocated_capital * position_leverage if self.use_futures else allocated_capital
                    fees = notional_value * self.taker_fee * 2  # Entry + exit
                    net_profit = gross_profit - fees
                    
                    # Update capital
                    capital += net_profit
                    
                    # Create trade record
                    duration_hours = (current_time - position_entry_time).total_seconds() / 3600
                    trade = Trade(
                        entry_time=position_entry_time,
                        exit_time=current_time,
                        side='long' if position_side == 1 else 'short',
                        entry_price=position_entry_price,
                        exit_price=exit_price,
                        size=position_size,
                        leverage=position_leverage,
                        gross_profit=gross_profit,
                        fees_paid=fees,
                        net_profit=net_profit,
                        return_pct=raw_return,
                        duration_hours=duration_hours,
                        exit_reason=exit_reason
                    )
                    trades.append(trade)
                    
                    # Reset position
                    in_position = False
                    position_side = 0
            
            # Check entry conditions
            if not in_position and signal != 0:
                position_side = signal
                position_entry_price = current_price * (1 + self.slippage_rate if signal == 1 else 1 - self.slippage_rate)
                position_entry_time = current_time
                
                # Calculate position size
                allocated_capital = capital * config.position_size_pct
                if self.use_futures:
                    position_size = (allocated_capital * position_leverage) / position_entry_price
                else:
                    position_size = allocated_capital / position_entry_price
                
                # Set stop loss and take profit
                if config.use_stop_loss and not pd.isna(row['stop_loss']):
                    stop_loss_price = row['stop_loss']
                if config.use_take_profit and not pd.isna(row['take_profit']):
                    take_profit_price = row['take_profit']
                
                in_position = True
            
            # Update equity curve and drawdown
            if capital > peak_capital:
                peak_capital = capital
            
            drawdown_pct = (peak_capital - capital) / peak_capital if peak_capital > 0 else 0
            
            equity_point = EquityPoint(
                timestamp=current_time,
                capital=capital,
                drawdown_pct=drawdown_pct,
                peak_capital=peak_capital
            )
            equity_curve.append(equity_point)
        
        # Calculate all metrics using the CROWN JEWEL method
        return self._calculate_proper_metrics(trades, equity_curve, config, df)
    
    def _calculate_proper_metrics(self, trades: List[Trade], equity_curve: List[EquityPoint], 
                                config: BacktestConfig, df: pd.DataFrame) -> BacktestResult:
        """Calculate ALL metrics properly - THE CROWN JEWEL! 👑"""
        
        if not trades:
            logger.warning("No trades executed - returning empty result")
            return self._empty_result(config, df)
        
        # Basic trade statistics
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t.net_profit > 0])
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Return calculations
        final_capital = equity_curve[-1].capital
        total_return_pct = (final_capital - self.initial_capital) / self.initial_capital
        total_profit_usd = final_capital - self.initial_capital
        
        # Time calculations
        start_date = equity_curve[0].timestamp
        end_date = equity_curve[-1].timestamp
        trading_days = (end_date - start_date).days
        years = trading_days / 365.25
        
        # CAGR calculation
        if years > 0 and final_capital > 0:
            cagr = (final_capital / self.initial_capital) ** (1/years) - 1
        else:
            cagr = 0
        
        # PROPER Max Drawdown calculation
        max_drawdown_pct = max([ep.drawdown_pct for ep in equity_curve]) if equity_curve else 0
        
        # Max drawdown duration (FIXED edge case handling)
        in_drawdown = False
        drawdown_start = None
        max_dd_duration = 0
        
        for ep in equity_curve:
            if ep.drawdown_pct > 0.01:  # In drawdown (>1%)
                if not in_drawdown:
                    drawdown_start = ep.timestamp
                    in_drawdown = True
            else:  # Out of drawdown
                if in_drawdown and drawdown_start:
                    duration = (ep.timestamp - drawdown_start).days
                    max_dd_duration = max(max_dd_duration, duration)
                    in_drawdown = False
        
        # EDGE CASE: Check if we end in drawdown
        if in_drawdown and drawdown_start:
            duration = (end_date - drawdown_start).days
            max_dd_duration = max(max_dd_duration, duration)
        
        # 🚀 GOLD STANDARD: Daily Equity Returns for Sharpe/Sortino/Volatility
        equity_df = pd.DataFrame([
            {'timestamp': ep.timestamp, 'capital': ep.capital} 
            for ep in equity_curve
        ])
        equity_df = equity_df.set_index('timestamp')
        
        # Resample to daily and calculate returns
        daily_equity = equity_df['capital'].resample('D').last().dropna()
        daily_returns = daily_equity.pct_change().dropna()
        
        # Market returns (daily resampled)
        market_df = df[['close']].copy()
        daily_market = market_df['close'].resample('D').last().dropna()
        market_returns = daily_market.pct_change().dropna()
        
        # Align dates for proper comparison
        common_dates = daily_returns.index.intersection(market_returns.index)
        aligned_portfolio_returns = daily_returns.reindex(common_dates)
        aligned_market_returns = market_returns.reindex(common_dates)
        
        # Remove any NaN values after alignment
        valid_mask = ~(aligned_portfolio_returns.isna() | aligned_market_returns.isna())
        aligned_portfolio_returns = aligned_portfolio_returns[valid_mask]
        aligned_market_returns = aligned_market_returns[valid_mask]
        
        # 🎯 GOLD STANDARD: Consistent annualization (crypto 24/7/365)
        ANNUALIZATION_FACTOR = 365
        
        # 🎯 PROPER Sharpe ratio calculation (daily returns, annualized)
        if len(aligned_portfolio_returns) > 1 and aligned_portfolio_returns.std() > 0:
            daily_risk_free = self.risk_free_rate / ANNUALIZATION_FACTOR
            excess_returns = aligned_portfolio_returns - daily_risk_free
            sharpe_ratio = excess_returns.mean() / aligned_portfolio_returns.std() * np.sqrt(ANNUALIZATION_FACTOR)
        else:
            sharpe_ratio = np.nan
        
        # 🎯 PROPER Sortino ratio (downside deviation)
        if len(aligned_portfolio_returns) > 1:
            negative_returns = aligned_portfolio_returns[aligned_portfolio_returns < 0]
            if len(negative_returns) > 0:
                downside_deviation = negative_returns.std() * np.sqrt(ANNUALIZATION_FACTOR)
                daily_risk_free = self.risk_free_rate / ANNUALIZATION_FACTOR
                excess_return_annual = (aligned_portfolio_returns.mean() - daily_risk_free) * ANNUALIZATION_FACTOR
                sortino_ratio = excess_return_annual / downside_deviation
            else:
                sortino_ratio = float('inf') if aligned_portfolio_returns.mean() > 0 else 0
        else:
            sortino_ratio = np.nan
        
        # 🎯 PROPER Annualized Volatility
        if len(aligned_portfolio_returns) > 1:
            volatility_annualized = aligned_portfolio_returns.std() * np.sqrt(ANNUALIZATION_FACTOR)
        else:
            volatility_annualized = np.nan
        
        # Calmar ratio
        calmar_ratio = cagr / max_drawdown_pct if max_drawdown_pct > 0 else float('inf')
        
        # PROPER Profit Factor calculation
        gross_profits = [t.gross_profit for t in trades if t.gross_profit > 0]
        gross_losses = [abs(t.gross_profit) for t in trades if t.gross_profit < 0]
        
        total_gross_profit = sum(gross_profits)
        total_gross_loss = sum(gross_losses)
        profit_factor = total_gross_profit / total_gross_loss if total_gross_loss > 0 else float('inf')
        
        # Trade statistics
        trade_returns = [t.return_pct for t in trades]
        avg_trade_return = np.mean(trade_returns)
        best_trade = max(trade_returns) if trade_returns else 0
        worst_trade = min(trade_returns) if trade_returns else 0
        
        winning_returns = [t.return_pct for t in trades if t.net_profit > 0]
        losing_returns = [t.return_pct for t in trades if t.net_profit < 0]
        
        avg_win = np.mean(winning_returns) if winning_returns else 0
        avg_loss = np.mean(losing_returns) if losing_returns else 0
        
        # Winning/losing streaks
        streaks = []
        current_streak = 0
        current_type = None
        
        for trade in trades:
            trade_type = 'win' if trade.net_profit > 0 else 'loss'
            if trade_type == current_type:
                current_streak += 1
            else:
                if current_streak > 0:
                    streaks.append((current_type, current_streak))
                current_streak = 1
                current_type = trade_type
        
        if current_streak > 0:
            streaks.append((current_type, current_streak))
        
        max_winning_streak = max([s[1] for s in streaks if s[0] == 'win'], default=0)
        max_losing_streak = max([s[1] for s in streaks if s[0] == 'loss'], default=0)
        
        # Trading frequency
        trades_per_week = total_trades / (trading_days / 7) if trading_days > 0 else 0
        avg_duration = np.mean([t.duration_hours for t in trades]) if trades else 0
        
        # Market comparison
        if len(daily_market) > 0:
            market_return = (daily_market.iloc[-1] - daily_market.iloc[0]) / daily_market.iloc[0]
        else:
            market_return = 0
        
        alpha = total_return_pct - market_return
        
        # 🚀 GOLD STANDARD: Beta calculation with perfect alignment
        if len(aligned_portfolio_returns) > 1 and len(aligned_market_returns) > 1:
            if aligned_market_returns.var() > 1e-10:  # Avoid division by near-zero variance
                covariance = np.cov(aligned_portfolio_returns, aligned_market_returns)[0, 1]
                market_variance = aligned_market_returns.var()
                beta = covariance / market_variance
            else:
                beta = np.nan
        else:
            beta = np.nan
        
        # Cost analysis
        total_fees_paid = sum([t.fees_paid for t in trades])
        fees_as_pct_of_profit = (total_fees_paid / total_profit_usd * 100) if total_profit_usd > 0 else 0
        
        # 🎉 ASSEMBLE THE FINAL RESULT
        return BacktestResult(
            config=config,
            
            # Trade statistics
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            
            # Return metrics
            total_return_pct=total_return_pct,
            total_profit_usd=total_profit_usd,
            cagr=cagr,
            
            # Risk metrics - PROPERLY CALCULATED!
            max_drawdown_pct=max_drawdown_pct,
            max_drawdown_duration_days=max_dd_duration,
            volatility_annualized=volatility_annualized,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            
            # Performance metrics
            profit_factor=profit_factor,
            avg_trade_return_pct=avg_trade_return,
            best_trade_pct=best_trade,
            worst_trade_pct=worst_trade,
            avg_win_pct=avg_win,
            avg_loss_pct=avg_loss,
            largest_winning_streak=max_winning_streak,
            largest_losing_streak=max_losing_streak,
            
            # Trading frequency
            trades_per_week=trades_per_week,
            avg_trade_duration_hours=avg_duration,
            
            # Benchmark comparison
            market_return_pct=market_return,
            alpha=alpha,
            beta=beta,
            
            # Cost analysis
            total_fees_paid=total_fees_paid,
            fees_as_pct_of_profit=fees_as_pct_of_profit,
            
            # Timing
            start_date=start_date,
            end_date=end_date,
            trading_days=trading_days,
            
            # Data for visualization
            equity_curve=equity_curve,
            trades=trades
        )
    
    def _empty_result(self, config: BacktestConfig, df: pd.DataFrame) -> BacktestResult:
        """Return properly initialized empty result when no trades are executed"""
        
        start_date = df.index[0] if len(df) > 0 else datetime.now()
        end_date = df.index[-1] if len(df) > 0 else datetime.now()
        trading_days = (end_date - start_date).days if end_date > start_date else 1
        
        # Market return for comparison
        market_return = 0
        if len(df) > 0:
            market_return = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]
        
        # Empty equity curve (just starting capital)
        equity_curve = [EquityPoint(start_date, self.initial_capital, 0.0, self.initial_capital)]
        
        return BacktestResult(
            config=config,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            total_return_pct=0.0,
            total_profit_usd=0.0,
            cagr=0.0,
            max_drawdown_pct=0.0,
            max_drawdown_duration_days=0.0,
            volatility_annualized=0.0,
            sharpe_ratio=np.nan,
            sortino_ratio=np.nan,
            calmar_ratio=0.0,
            profit_factor=0.0,
            avg_trade_return_pct=0.0,
            best_trade_pct=0.0,
            worst_trade_pct=0.0,
            avg_win_pct=0.0,
            avg_loss_pct=0.0,
            largest_winning_streak=0,
            largest_losing_streak=0,
            trades_per_week=0.0,
            avg_trade_duration_hours=0.0,
            market_return_pct=market_return,
            alpha=-market_return,
            beta=np.nan,
            total_fees_paid=0.0,
            fees_as_pct_of_profit=0.0,
            start_date=start_date,
            end_date=end_date,
            trading_days=trading_days,
            equity_curve=equity_curve,
            trades=[]
        )
    
    async def run_campaign(self, test_matrix: Dict) -> List[BacktestResult]:
        """Run the parameter matrix campaign - THE MAIN EVENT! 🎪"""
        
        logger.info("🏰 Starting Grand Campaign!")
        logger.info(f"🎯 Test matrix: {test_matrix}")
        
        all_results = []
        
        # Calculate total combinations for progress tracking
        total_combinations = (
            len(test_matrix['symbols']) *
            len(test_matrix['timeframes']) *
            len(self.test_periods) *
            len(test_matrix['signal_thresholds']) *
            len(test_matrix['position_sizes']) *
            len(test_matrix['atr_stops']) *
            len(test_matrix['atr_profits']) *
            len(test_matrix['leverage'])
        )
        
        logger.info(f"📊 Total combinations: {total_combinations:,}")
        completed_configs = 0
        
        # OUTER LOOPS: Data-defining parameters (fetch once per unique combination)
        for symbol in test_matrix['symbols']:
            for timeframe in test_matrix['timeframes']:
                for test_period in self.test_periods:
                    
                    logger.info(f"🌐 Processing: {symbol} {timeframe} {test_period.name}")
                    
                    # Get data with indicators (cached automatically)
                    df_with_indicators = await self.get_data_with_indicators(symbol, timeframe, test_period)
                    
                    if df_with_indicators is None or len(df_with_indicators) < 100:
                        logger.warning(f"Insufficient data for {symbol} {timeframe} {test_period.name}")
                        continue
                    
                    # INNER LOOPS: Simulation parameters (reuse data)
                    for threshold in test_matrix['signal_thresholds']:
                        for pos_size in test_matrix['position_sizes']:
                            for atr_stop in test_matrix['atr_stops']:
                                for atr_profit in test_matrix['atr_profits']:
                                    for leverage in test_matrix['leverage']:
                                        
                                        completed_configs += 1
                                        progress = (completed_configs / total_combinations) * 100
                                        
                                        config = BacktestConfig(
                                            symbol=symbol,
                                            timeframe=timeframe,
                                            test_period=test_period,
                                            strategy_name="Supercharged",
                                            signal_threshold=threshold,
                                            position_size_pct=pos_size,
                                            leverage=leverage,
                                            atr_stop_multiplier=atr_stop,
                                            atr_profit_multiplier=atr_profit
                                        )
                                        
                                        logger.info(f"🚀 [{progress:.1f}%] Config {completed_configs}/{total_combinations}: "
                                                   f"{symbol} {timeframe} {test_period.name} "
                                                   f"(T:{threshold}, PS:{pos_size}, SL:{atr_stop}, TP:{atr_profit}, L:{leverage})")
                                        
                                        try:
                                            # Generate signals (fast, no API calls)
                                            df_with_signals = self.generate_signals(df_with_indicators.copy(), config)
                                            
                                            # Simulate trading (fast, no API calls)
                                            result = self.simulate_trading(df_with_signals, config)
                                            all_results.append(result)
                                            
                                        except Exception as e:
                                            logger.error(f"Simulation failed for config {completed_configs}: {e}")
                                            continue
        
        logger.info(f"🎉 Campaign complete! {len(all_results)} successful backtests")
        
        # Save results
        self.save_results(all_results)
        
        return all_results
    
    def save_results(self, results: List[BacktestResult], filename: str = None):
        """Save results to disk for later analysis"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_results_{timestamp}.pkl"
        
        save_path = Path("backtest_results")
        save_path.mkdir(exist_ok=True)
        
        filepath = save_path / filename
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(results, f)
            
            logger.info(f"💾 Results saved to {filepath}")
            logger.info(f"📊 Saved {len(results)} backtest results")
            
            # Also save a summary CSV for quick inspection
            summary_data = []
            for result in results:
                summary_data.append({
                    'symbol': result.config.symbol,
                    'timeframe': result.config.timeframe,
                    'test_period': result.config.test_period.name,
                    'market_condition': result.config.test_period.expected_condition.value,
                    'signal_threshold': result.config.signal_threshold,
                    'atr_stop_mult': result.config.atr_stop_multiplier,
                    'atr_profit_mult': result.config.atr_profit_multiplier,
                    'position_size': result.config.position_size_pct,
                    'leverage': result.config.leverage,
                    'total_trades': result.total_trades,
                    'win_rate': result.win_rate,
                    'total_return_pct': result.total_return_pct,
                    'cagr': result.cagr,
                    'max_drawdown_pct': result.max_drawdown_pct,
                    'sharpe_ratio': result.sharpe_ratio,
                    'sortino_ratio': result.sortino_ratio,
                    'profit_factor': result.profit_factor,
                    'alpha': result.alpha,
                    'beta': result.beta
                })
            
            summary_df = pd.DataFrame(summary_data)
            csv_path = save_path / filename.replace('.pkl', '_summary.csv')
            summary_df.to_csv(csv_path, index=False)
            
            logger.info(f"📊 Summary CSV saved to {csv_path}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    @staticmethod
    def load_results(filename: str) -> List[BacktestResult]:
        """Load previously saved results"""
        filepath = Path("backtest_results") / filename
        
        try:
            with open(filepath, 'rb') as f:
                results = pickle.load(f)
            
            logger.info(f"📂 Loaded {len(results)} results from {filepath}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to load results: {e}")
            return []


# THE EXECUTION FUNCTION! 🚀
async def run_grand_campaign():
    """Execute the parameter matrix testing campaign!"""
    
    # Initialize the backtesting engine
    backtester = BacktestingEngine(initial_capital=10000, use_futures=False)
    
    # Conservative test matrix to start (scale up after confirming it works)
    test_matrix = {
        'symbols': ['BTC/USD'],  # Start with 1 symbol
        'timeframes': ['1h'],    # 1 timeframe
        'signal_thresholds': [60, 75, 90],  # 3 thresholds
        'position_sizes': [0.25],  # 1 position size
        'atr_stops': [1.5, 2.0, 2.5],  # 3 stop multipliers
        'atr_profits': [3.0, 4.0],  # 2 profit multipliers
        'leverage': [1.0, 2.0]  # 2 leverage levels
    }
    
    # This gives us: 1×1×3×3×1×3×2×2 = 36 configurations (perfect for testing!)
    
    logger.info("🏰 THE GRAND CAMPAIGN BEGINS!")
    logger.info(f"🎯 Conservative test matrix: {test_matrix}")
    
    # Execute the campaign
    results = await backtester.run_campaign(test_matrix)
    
    logger.info(f"🎉 Campaign complete! {len(results)} results generated")
    
    return results


# Example usage
if __name__ == "__main__":
    async def main():
        # Run the campaign
        results = await run_grand_campaign()
        
        if results:
            logger.info("🧠 Results ready for analysis!")
            logger.info("💡 API calls minimized, Kraken appeased, metrics perfected!")
            
            # Quick preview of top performers
            valid_results = [r for r in results if r.total_trades > 0]
            
            if valid_results:
                sorted_results = sorted(
                    valid_results, 
                    key=lambda x: x.sharpe_ratio if not pd.isna(x.sharpe_ratio) else -999, 
                    reverse=True
                )
                
                logger.info("\n🏆 TOP 5 PERFORMERS BY SHARPE RATIO:")
                for i, result in enumerate(sorted_results[:5], 1):
                    logger.info(f"{i}. {result.config.symbol} {result.config.timeframe} "
                               f"{result.config.test_period.name} "
                               f"(Thresh:{result.config.signal_threshold}, "
                               f"ATR:{result.config.atr_stop_multiplier}/{result.config.atr_profit_multiplier}, "
                               f"Lev:{result.config.leverage}) "
                               f"- Sharpe: {result.sharpe_ratio:.3f}, "
                               f"Return: {result.total_return_pct:.2%}, "
                               f"Trades: {result.total_trades}")
            else:
                logger.warning("No valid results with trades found")
        
        return results
    
    # Run it with proper error handling
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("👋 Campaign interrupted by user")
    except Exception as e:
        logger.error(f"💥 Campaign failed: {e}")
        logger.error("🔍 Check API credentials and network connection")