#!/usr/bin/env python3
"""
TradeMonkey Lite - Enhanced Edition (Kraken Powered!)
"With great leverage comes great responsibility, and with Kraken comes great profits!"
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time
import logging
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncio
import aiohttp
from dotenv import load_dotenv
import coloredlogs
import ta

# Load our configuration
from config.settings import config

# Load environment variables
load_dotenv()

# Setup fancy logging
coloredlogs.install(level=config.LOG_LEVEL, fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('TradeMonkeyLite')


class SignalStrength(Enum):
    """Signal strength for multi-timeframe confirmation"""
    STRONG = 3
    MEDIUM = 2
    WEAK = 1
    NONE = 0


@dataclass
class TradingSignal:
    """Represents a trading signal with metadata"""
    symbol: str
    side: str  # 'long' or 'short'
    strength: SignalStrength
    entry_price: float
    stop_loss: float
    take_profits: List[float]
    timeframes_confirmed: List[str]
    volume_confirmed: bool
    timestamp: datetime


class NotificationManager:
    """Handles Discord and Telegram notifications - Now with more emoji! 🎉"""
    
    def __init__(self):
        self.discord_webhook = config.DISCORD_WEBHOOK_URL
        self.telegram_token = config.TELEGRAM_BOT_TOKEN
        self.telegram_chat_id = config.TELEGRAM_CHAT_ID
        
    async def send_discord(self, message: str, color: int = 0x00ff00):
        """Send Discord notification"""
        if not self.discord_webhook:
            return
            
        embed = {
            "title": "🐙 TradeMonkey Lite (Kraken Edition)",
            "description": message,
            "color": color,
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {"text": "Powered by the power of the Kraken! 🐙⚡"}
        }
        
        payload = {"embeds": [embed]}
        
        async with aiohttp.ClientSession() as session:
            try:
                await session.post(self.discord_webhook, json=payload)
            except Exception as e:
                logger.error(f"Discord notification failed: {e}")
                
    async def send_telegram(self, message: str):
        """Send Telegram notification"""
        if not self.telegram_token or not self.telegram_chat_id:
            return
            
        url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
        payload = {
            "chat_id": self.telegram_chat_id,
            "text": f"🐙 TradeMonkey Lite (Kraken Edition)\\n\\n{message}",
            "parse_mode": "Markdown"
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                await session.post(url, json=payload)
            except Exception as e:
                logger.error(f"Telegram notification failed: {e}")
                
    async def notify(self, message: str, alert_type: str = "info"):
        """Send notification to all channels"""
        colors = {
            "info": 0x00ff00,    # Green
            "warning": 0xffff00,  # Yellow
            "error": 0xff0000,    # Red
            "profit": 0x00ffff    # Cyan
        }
        
        color = colors.get(alert_type, 0x00ff00)
        
        # Send to both platforms
        await asyncio.gather(
            self.send_discord(message, color),
            self.send_telegram(message),
            return_exceptions=True
        )


class EnhancedIndicators:
    """Technical indicators with multi-timeframe analysis - Kraken optimized! 🐙"""
    
    @staticmethod
    def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators"""
        # Moving averages
        df['sma_20'] = ta.trend.sma_indicator(df['close'], window=config.INDICATORS['sma_short'])
        df['sma_50'] = ta.trend.sma_indicator(df['close'], window=config.INDICATORS['sma_long'])
        df['ema_12'] = ta.trend.ema_indicator(df['close'], window=config.INDICATORS['ema_fast'])
        df['ema_26'] = ta.trend.ema_indicator(df['close'], window=config.INDICATORS['ema_slow'])
        
        # RSI
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=config.INDICATORS['rsi_period']).rsi()
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['close'], window=config.INDICATORS['bb_period'])
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=config.INDICATORS['volume_ma_period']).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # ATR for stop loss
        df['atr'] = ta.volatility.AverageTrueRange(
            df['high'], 
            df['low'], 
            df['close'], 
            window=config.INDICATORS['atr_period']
        ).average_true_range()
        
        return df
        
    @staticmethod
    def analyze_volume(df: pd.DataFrame) -> bool:
        """Check if volume confirms the move"""
        latest = df.iloc[-1]
        return latest['volume_ratio'] > config.INDICATORS['volume_threshold']


class EnhancedStrategy:
    """Multi-timeframe momentum strategy with volume confirmation - Kraken Edition! 🐙"""
    
    def __init__(self, exchange: ccxt.Exchange):
        self.exchange = exchange
        self.indicators = EnhancedIndicators()
        self.timeframes = config.TIMEFRAMES
        self.min_confirmations = config.MIN_TIMEFRAME_CONFIRMATIONS
        
    async def analyze_timeframe(self, symbol: str, timeframe: str) -> Optional[str]:
        """Analyze a single timeframe"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=100)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df = self.indicators.calculate_indicators(df)
            
            if len(df) < 50:
                return None
                
            latest = df.iloc[-1]
            
            # Long signals (bullish conditions)
            long_conditions = [
                latest['close'] > latest['sma_20'],
                latest['sma_20'] > latest['sma_50'],
                latest['rsi'] > 50 and latest['rsi'] < config.INDICATORS['rsi_overbought'],
                latest['macd'] > latest['macd_signal'],
                latest['close'] > latest['bb_middle']
            ]
            
            # Short signals (bearish conditions)
            short_conditions = [
                latest['close'] < latest['sma_20'],
                latest['sma_20'] < latest['sma_50'],
                latest['rsi'] < 50 and latest['rsi'] > config.INDICATORS['rsi_oversold'],
                latest['macd'] < latest['macd_signal'],
                latest['close'] < latest['bb_middle']
            ]
            
            if sum(long_conditions) >= 4:
                return 'long'
            elif sum(short_conditions) >= 4:
                return 'short'
                
            return None
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol} on {timeframe}: {e}")
            return None
            
    async def get_signal(self, symbol: str) -> Optional[TradingSignal]:
        """Get trading signal with multi-timeframe confirmation"""
        signals = {}
        
        # Analyze all timeframes
        tasks = [self.analyze_timeframe(symbol, tf) for tf in self.timeframes]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count confirmations
        for tf, signal in zip(self.timeframes, results):
            if isinstance(signal, str):  # Valid signal, not an exception
                signals[tf] = signal
                
        # Check if we have enough confirmations
        long_count = sum(1 for s in signals.values() if s == 'long')
        short_count = sum(1 for s in signals.values() if s == 'short')
        
        if long_count >= self.min_confirmations:
            side = 'long'
            strength = SignalStrength.STRONG if long_count == 4 else SignalStrength.MEDIUM
        elif short_count >= self.min_confirmations:
            side = 'short'
            strength = SignalStrength.STRONG if short_count == 4 else SignalStrength.MEDIUM
        else:
            return None
            
        # Get current price and calculate levels
        ticker = self.exchange.fetch_ticker(symbol)
        current_price = ticker['last']
        
        # Fetch 1h data for ATR calculation
        ohlcv = self.exchange.fetch_ohlcv(symbol, '1h', limit=50)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df = self.indicators.calculate_indicators(df)
        
        # Check volume confirmation
        volume_confirmed = self.indicators.analyze_volume(df)
        
        # Calculate stop loss using ATR
        atr = df['atr'].iloc[-1]
        stop_distance = atr * config.INDICATORS['atr_stop_multiplier']
        
        if side == 'long':
            stop_loss = current_price - stop_distance
            take_profits = [
                current_price * (1 + tp) for tp in config.TAKE_PROFIT_LEVELS
            ]
        else:
            stop_loss = current_price + stop_distance
            take_profits = [
                current_price * (1 - tp) for tp in config.TAKE_PROFIT_LEVELS
            ]
            
        return TradingSignal(
            symbol=symbol,
            side=side,
            strength=strength,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profits=take_profits,
            timeframes_confirmed=list(signals.keys()),
            volume_confirmed=volume_confirmed,
            timestamp=datetime.now()
        )


class AdvancedPosition:
    """Enhanced position with partial profit taking - Kraken optimized! 🐙"""
    
    def __init__(self, signal: TradingSignal, size: float, leverage: float):
        self.signal = signal
        self.initial_size = size
        self.remaining_size = size
        self.leverage = leverage
        self.entry_price = signal.entry_price
        self.realized_pnl = 0
        self.take_profit_levels = signal.take_profits.copy()
        self.partial_closes = [0.25, 0.25, 0.25, 0.25]  # Take 25% at each TP
        self.trailing_stop_pct = config.TRAILING_STOP_PCT
        self.highest_price = signal.entry_price if signal.side == 'long' else float('inf')
        self.lowest_price = signal.entry_price if signal.side == 'short' else 0
        
    def update(self, current_price: float) -> List[Dict]:
        """Update position and return list of actions to take"""
        actions = []
        
        # Update trailing stop
        if self.signal.side == 'long':
            if current_price > self.highest_price:
                self.highest_price = current_price
                new_stop = max(
                    self.signal.stop_loss,
                    self.highest_price * (1 - self.trailing_stop_pct)
                )
                if new_stop > self.signal.stop_loss:
                    self.signal.stop_loss = new_stop
                    
            # Check take profits
            for i, tp in enumerate(self.take_profit_levels):
                if current_price >= tp and self.partial_closes[i] > 0:
                    close_size = self.initial_size * self.partial_closes[i]
                    actions.append({
                        'type': 'partial_close',
                        'size': close_size,
                        'price': current_price,
                        'level': i + 1
                    })
                    self.remaining_size -= close_size
                    self.partial_closes[i] = 0
                    self.realized_pnl += close_size * (current_price - self.entry_price)
                    
        else:  # Short position
            if current_price < self.lowest_price:
                self.lowest_price = current_price
                new_stop = min(
                    self.signal.stop_loss,
                    self.lowest_price * (1 + self.trailing_stop_pct)
                )
                if new_stop < self.signal.stop_loss:
                    self.signal.stop_loss = new_stop
                    
            # Check take profits
            for i, tp in enumerate(self.take_profit_levels):
                if current_price <= tp and self.partial_closes[i] > 0:
                    close_size = self.initial_size * self.partial_closes[i]
                    actions.append({
                        'type': 'partial_close',
                        'size': close_size,
                        'price': current_price,
                        'level': i + 1
                    })
                    self.remaining_size -= close_size
                    self.partial_closes[i] = 0
                    self.realized_pnl += close_size * (self.entry_price - current_price)
                    
        # Check stop loss
        if self.signal.side == 'long' and current_price <= self.signal.stop_loss:
            actions.append({'type': 'stop_loss', 'size': self.remaining_size})
        elif self.signal.side == 'short' and current_price >= self.signal.stop_loss:
            actions.append({'type': 'stop_loss', 'size': self.remaining_size})
            
        return actions
        
    def get_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized PnL"""
        if self.signal.side == 'long':
            return self.remaining_size * (current_price - self.entry_price)
        else:
            return self.remaining_size * (self.entry_price - current_price)


class TradeMonkeyLiteEnhanced:
    """Enhanced main bot with all the features - Now with 100% more Kraken! 🐙"""
    
    def __init__(self):
        # Use our configuration system
        exchange_config = config.get_exchange_config()
        
        self.exchange = ccxt.kraken(exchange_config)
        
        self.strategy = EnhancedStrategy(self.exchange)
        self.notifier = NotificationManager()
        self.positions: Dict[str, AdvancedPosition] = {}
        
        # Load settings from config
        self.max_positions = config.MAX_POSITIONS
        self.position_size_pct = config.POSITION_SIZE_PCT
        self.initial_leverage = config.INITIAL_LEVERAGE
        self.max_leverage = config.MAX_LEVERAGE
        
        # Simulation mode settings
        self.dry_run_mode = config.DRY_RUN_MODE
        self.simulated_balance = config.STARTING_CAPITAL if self.dry_run_mode else 0
        
        # Load target symbols from config
        self.target_symbols = config.TARGET_SYMBOLS
        
        logger.info(f"🐙 Kraken TradeMonkey initialized!")
        logger.info(f"Mode: {'DRY RUN Simulation' if self.dry_run_mode else 'Paper Trading'}")
        if self.dry_run_mode:
            logger.info(f"Simulated Capital: ${self.simulated_balance:,.2f}")
        logger.info(f"Symbols: {', '.join(self.target_symbols)}")
        
    def get_available_balance(self) -> float:
        """Get available balance (real or simulated)"""
        if self.dry_run_mode:
            return self.simulated_balance
        else:
            try:
                balance = self.exchange.fetch_balance()
                return balance.get('USDT', {}).get('free', 0) or balance.get('USD', {}).get('free', 0)
            except Exception as e:
                logger.error(f"Error fetching balance: {e}")
                return 0
        
    async def open_position(self, signal: TradingSignal):
        """Open a new position based on signal"""
        try:
            # Get available balance (real or simulated)
            usdt_balance = self.get_available_balance()
            
            if usdt_balance < 10:  # Minimum balance check
                logger.warning(f"Insufficient balance: ${usdt_balance:.2f}")
                return
            
            # Calculate position size
            position_value = usdt_balance * self.position_size_pct
            
            # For spot trading, we don't use leverage, so position_value is the actual amount
            leverage = 1.0  # No leverage in spot trading
            if signal.strength == SignalStrength.STRONG and signal.volume_confirmed:
                leverage = 1.0  # Still no leverage in spot, but we could buy more
                
            # Calculate order size (amount of base currency to buy/sell)
            contract_size = position_value / signal.entry_price
            
            if self.dry_run_mode:
                # SIMULATE the order
                logger.info(f"🎮 SIMULATED ORDER: {signal.side} {contract_size:.6f} {signal.symbol} at ${signal.entry_price:.4f}")
                
                # Deduct from simulated balance
                self.simulated_balance -= position_value
                
                # Create simulated position
                position = AdvancedPosition(signal, contract_size, leverage)
                self.positions[signal.symbol] = position
                
                # Notify
                message = (
                    f"🎮 **SIMULATED {signal.side.upper()} Position**\\n"
                    f"Symbol: {signal.symbol}\\n"
                    f"Entry: ${signal.entry_price:.4f}\\n"
                    f"Amount: {contract_size:.6f}\\n"
                    f"Value: ${position_value:.2f}\\n"
                    f"Stop Loss: ${signal.stop_loss:.4f}\\n"
                    f"Remaining Balance: ${self.simulated_balance:.2f}\\n"
                    f"Timeframes: {', '.join(signal.timeframes_confirmed)}\\n"
                    f"Volume: {'✅' if signal.volume_confirmed else '❌'}\\n"
                    f"Signal Strength: {signal.strength.name}"
                )
                await self.notifier.notify(message, "info")
                
                logger.info(f"🎮 Simulated {signal.side} position on {signal.symbol} at ${signal.entry_price:.4f}")
                
            else:
                # REAL order
                side = 'buy' if signal.side == 'long' else 'sell'
                order = await asyncio.to_thread(
                    self.exchange.create_order,
                    signal.symbol,
                    'market',
                    side,
                    contract_size
                )
                
                # Create position
                position = AdvancedPosition(signal, contract_size, leverage)
                self.positions[signal.symbol] = position
                
                # Notify
                message = (
                    f"📈 **New {signal.side.upper()} Position** (SPOT)\\n"
                    f"Symbol: {signal.symbol}\\n"
                    f"Entry: ${signal.entry_price:.4f}\\n"
                    f"Amount: {contract_size:.6f}\\n"
                    f"Value: ${position_value:.2f}\\n"
                    f"Stop Loss: ${signal.stop_loss:.4f}\\n"
                    f"Timeframes: {', '.join(signal.timeframes_confirmed)}\\n"
                    f"Volume: {'✅' if signal.volume_confirmed else '❌'}\\n"
                    f"Signal Strength: {signal.strength.name}"
                )
                await self.notifier.notify(message, "info")
                
                logger.info(f"Opened {signal.side} SPOT position on {signal.symbol} at ${signal.entry_price}")
            
        except Exception as e:
            logger.error(f"Error opening position: {e}")
            await self.notifier.notify(f"❌ Failed to open position: {str(e)}", "error")
            
    async def execute_actions(self, symbol: str, actions: List[Dict]):
        """Execute position management actions"""
        position = self.positions[symbol]
        
        for action in actions:
            try:
                if action['type'] == 'partial_close':
                    # Close partial position
                    side = 'sell' if position.signal.side == 'long' else 'buy'
                    order = await asyncio.to_thread(
                        self.exchange.create_order,
                        symbol,
                        'market',
                        side,
                        action['size']
                    )
                    
                    profit_pct = ((action['price'] - position.entry_price) / position.entry_price) * 100
                    if position.signal.side == 'short':
                        profit_pct = -profit_pct
                        
                    message = (
                        f"💰 **Partial Close - TP{action['level']}**\\n"
                        f"Symbol: {symbol}\\n"
                        f"Size: {action['size']:.4f}\\n"
                        f"Exit: ${action['price']:.4f}\\n"
                        f"Profit: {profit_pct:.2f}%\\n"
                        f"Remaining: {position.remaining_size:.4f}"
                    )
                    await self.notifier.notify(message, "profit")
                    
                elif action['type'] == 'stop_loss':
                    # Close entire remaining position
                    side = 'sell' if position.signal.side == 'long' else 'buy'
                    order = await asyncio.to_thread(
                        self.exchange.create_order,
                        symbol,
                        'market',
                        side,
                        action['size']
                    )
                    
                    # Calculate total PnL
                    ticker = self.exchange.fetch_ticker(symbol)
                    exit_price = ticker['last']
                    total_pnl = position.realized_pnl + position.get_unrealized_pnl(exit_price)
                    
                    message = (
                        f"🛑 **Stop Loss Hit**\\n"
                        f"Symbol: {symbol}\\n"
                        f"Exit: ${exit_price:.4f}\\n"
                        f"Total PnL: ${total_pnl:.2f}\\n"
                        f"Position closed completely"
                    )
                    await self.notifier.notify(message, "warning")
                    
                    # Remove position
                    del self.positions[symbol]
                    
            except Exception as e:
                logger.error(f"Error executing action {action['type']}: {e}")
                
    async def manage_positions(self):
        """Manage all open positions"""
        for symbol, position in list(self.positions.items()):
            try:
                ticker = self.exchange.fetch_ticker(symbol)
                current_price = ticker['last']
                
                # Get actions from position update
                actions = position.update(current_price)
                
                if actions:
                    await self.execute_actions(symbol, actions)
                    
                # Note: No leverage adjustment for spot trading
                # In spot, we just hold the position and manage exits
                        
            except Exception as e:
                logger.error(f"Error managing position {symbol}: {e}")
                
    async def run(self):
        """Main bot loop - The heart of the Kraken! 🐙"""
        logger.info("🚀 TradeMonkey Lite Enhanced starting up!")
        await self.notifier.notify("🐙 TradeMonkey Lite is ONLINE! Kraken has been released! Let's get this bread!", "info")
        
        # Load markets first with better error handling
        try:
            logger.info("🔄 Loading markets from Kraken...")
            await asyncio.to_thread(self.exchange.load_markets)
            
            # Count different market types
            futures_count = sum(1 for m in self.exchange.markets.values() if m.get('type') == 'future')
            spot_count = sum(1 for m in self.exchange.markets.values() if m.get('type') == 'spot')
            
            logger.info(f"✅ Loaded {len(self.exchange.markets)} total markets from Kraken")
            logger.info(f"   📈 Futures markets: {futures_count}")
            logger.info(f"   💰 Spot markets: {spot_count}")
            
            # Verify our target symbols exist
            available_symbols = []
            missing_symbols = []
            
            for symbol in self.target_symbols:
                if symbol in self.exchange.markets:
                    market = self.exchange.markets[symbol]
                    market_type = market.get('type', 'unknown')
                    available_symbols.append(f"{symbol} ({market_type})")
                else:
                    missing_symbols.append(symbol)
            
            if available_symbols:
                logger.info(f"🎯 Target symbols found: {', '.join(available_symbols)}")
            
            if missing_symbols:
                logger.warning(f"⚠️  Missing symbols: {', '.join(missing_symbols)}")
                logger.info("💡 Run 'python main.py --symbols' to see available markets")
                
        except Exception as e:
            logger.error(f"Failed to load markets: {e}")
            logger.error("🚨 Cannot continue without market data!")
            return
        
        while True:
            try:
                # Check for new signals
                if len(self.positions) < self.max_positions:
                    tasks = [self.strategy.get_signal(symbol) for symbol in self.target_symbols
                            if symbol not in self.positions]
                    signals = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # Open positions for valid signals
                    for signal in signals:
                        if isinstance(signal, TradingSignal) and len(self.positions) < self.max_positions:
                            await self.open_position(signal)
                            
                # Manage existing positions
                await self.manage_positions()
                
                # Log status
                try:
                    balance = self.exchange.fetch_balance()
                    usdt_balance = balance.get('USDT', {}).get('free', 0) or balance.get('USD', {}).get('free', 0)
                    total_pnl = sum(p.realized_pnl + p.get_unrealized_pnl(
                        self.exchange.fetch_ticker(s)['last']
                    ) for s, p in self.positions.items())
                    
                    logger.info(
                        f"💰 Balance: ${usdt_balance:.2f} | "
                        f"🎯 Positions: {len(self.positions)}/{self.max_positions} | "
                        f"📈 Total PnL: ${total_pnl:.2f}"
                    )
                except Exception as e:
                    logger.warning(f"Error fetching status: {e}")
                
                # Sleep between cycles
                await asyncio.sleep(config.SIGNAL_CHECK_INTERVAL)
                
            except KeyboardInterrupt:
                logger.info("Shutting down...")
                await self.notifier.notify("🛑 TradeMonkey Lite shutting down gracefully", "warning")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(60)


if __name__ == "__main__":
    import sys
    
    # Print configuration summary
    config.print_config_summary()
    
    if not config.validate_config():
        logger.error("❌ Configuration validation failed!")
        sys.exit(1)
        
    # Create and run bot
    bot = TradeMonkeyLiteEnhanced()
    
    # Run the async bot
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        logger.info("👋 Shutdown requested by user")
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        sys.exit(1)
