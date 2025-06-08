#!/usr/bin/env python3
"""
TradeMonkey Lite Configuration Settings
"Configuration is the foundation of automation" - Ancient DevOps wisdom
"""

import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class TradingConfig:
    """Configuration manager for TradeMonkey Lite"""
    
    def __init__(self):
        self.load_config()
        
    def load_config(self):
        """Load configuration from environment variables and files"""
        
        # =============================================================================
        # API CREDENTIALS
        # =============================================================================
        self.KRAKEN_API_KEY = os.getenv('KRAKEN_API_KEY', '')
        self.KRAKEN_API_SECRET = os.getenv('KRAKEN_API_SECRET', '')
        
        if not self.KRAKEN_API_KEY or not self.KRAKEN_API_SECRET:
            raise ValueError("Missing Kraken API credentials! Check your .env file")
        
        # =============================================================================
        # NOTIFICATIONS
        # =============================================================================
        self.DISCORD_WEBHOOK_URL = os.getenv('DISCORD_WEBHOOK_URL', '')
        self.TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
        self.TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')
        
        # =============================================================================
        # TRADING PARAMETERS
        # =============================================================================
        self.USE_TESTNET = os.getenv('USE_TESTNET', 'true').lower() == 'true'
        self.DRY_RUN_MODE = os.getenv('DRY_RUN_MODE', 'true').lower() == 'true'  # Simulate trades
        self.STARTING_CAPITAL = float(os.getenv('STARTING_CAPITAL', '1000.0'))  # For simulation
        self.MAX_POSITIONS = int(os.getenv('MAX_POSITIONS', '4'))
        self.POSITION_SIZE_PCT = float(os.getenv('POSITION_SIZE_PCT', '0.25'))
        self.INITIAL_LEVERAGE = float(os.getenv('INITIAL_LEVERAGE', '2.0'))
        self.MAX_LEVERAGE = float(os.getenv('MAX_LEVERAGE', '3.0'))
        
        # =============================================================================
        # RISK MANAGEMENT
        # =============================================================================
        self.TRAILING_STOP_PCT = float(os.getenv('TRAILING_STOP_PCT', '0.05'))
        
        # Parse take profit levels
        tp_levels_str = os.getenv('TAKE_PROFIT_LEVELS', '0.02,0.05,0.10,0.20')
        self.TAKE_PROFIT_LEVELS = [float(x.strip()) for x in tp_levels_str.split(',')]
        
        # =============================================================================
        # OPERATIONAL SETTINGS
        # =============================================================================
        self.API_RATE_LIMIT = os.getenv('API_RATE_LIMIT', 'true').lower() == 'true'
        self.API_REQUESTS_PER_MINUTE = int(os.getenv('API_REQUESTS_PER_MINUTE', '1000'))
        self.LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
        
        self.SIGNAL_CHECK_INTERVAL = int(os.getenv('SIGNAL_CHECK_INTERVAL', '30'))
        self.POSITION_CHECK_INTERVAL = int(os.getenv('POSITION_CHECK_INTERVAL', '15'))
        
        self.MIN_TIMEFRAME_CONFIRMATIONS = int(os.getenv('MIN_TIMEFRAME_CONFIRMATIONS', '3'))
        
        # =============================================================================
        # TRADING SYMBOLS
        # =============================================================================
        # Use symbols that work with Kraken spot markets
        symbols_str = os.getenv('TARGET_SYMBOLS', 'BTC/USD,ETH/USD,BTC/USDT,ETH/USDT')
        self.TARGET_SYMBOLS = [s.strip() for s in symbols_str.split(',')]
        
        # =============================================================================
        # STRATEGY CONFIGURATION
        # =============================================================================
        self.TIMEFRAMES = ['5m', '15m', '1h', '4h']
        
        # Technical indicator settings
        self.INDICATORS = {
            'sma_short': 20,
            'sma_long': 50,
            'ema_fast': 12,
            'ema_slow': 26,
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'bb_period': 20,
            'bb_std': 2,
            'atr_period': 14,
            'atr_stop_multiplier': 2.0,
            'volume_ma_period': 20,
            'volume_threshold': 1.5  # 50% above average
        }
        
    def load_strategy_config(self) -> Dict[str, Any]:
        """Load strategy configuration from JSON file"""
        try:
            with open('config/strategies.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Return default strategy config
            return {
                "symbols": self.TARGET_SYMBOLS,
                "timeframes": self.TIMEFRAMES,
                "min_confirmations": self.MIN_TIMEFRAME_CONFIRMATIONS,
                "risk_management": {
                    "trailing_stop_pct": self.TRAILING_STOP_PCT,
                    "partial_take_profits": self.TAKE_PROFIT_LEVELS
                },
                "indicators": self.INDICATORS
            }
    
    def validate_config(self) -> bool:
        """Validate configuration settings"""
        errors = []
        
        # Check required API keys
        if not self.KRAKEN_API_KEY:
            errors.append("Missing KRAKEN_API_KEY")
        if not self.KRAKEN_API_SECRET:
            errors.append("Missing KRAKEN_API_SECRET")
            
        # Validate ranges
        if self.POSITION_SIZE_PCT <= 0 or self.POSITION_SIZE_PCT > 1:
            errors.append("POSITION_SIZE_PCT must be between 0 and 1")
            
        if self.INITIAL_LEVERAGE < 1 or self.INITIAL_LEVERAGE > 100:
            errors.append("INITIAL_LEVERAGE must be between 1 and 100")
            
        if self.MAX_LEVERAGE < self.INITIAL_LEVERAGE:
            errors.append("MAX_LEVERAGE must be >= INITIAL_LEVERAGE")
            
        if self.MAX_POSITIONS < 1:
            errors.append("MAX_POSITIONS must be >= 1")
            
        if errors:
            print("❌ Configuration errors:")
            for error in errors:
                print(f"  - {error}")
            return False
            
        print("✅ Configuration validation passed")
        return True
    
    def get_exchange_config(self) -> Dict[str, Any]:
        """Get exchange-specific configuration"""
        config = {
            'apiKey': self.KRAKEN_API_KEY,
            'secret': self.KRAKEN_API_SECRET,
            'enableRateLimit': self.API_RATE_LIMIT,
            'rateLimit': 60000 // self.API_REQUESTS_PER_MINUTE,  # Convert to ms between requests
            'options': {
                # Use spot for initial testing since futures demo endpoints aren't working
                'defaultType': 'spot' if self.USE_TESTNET else 'future',
            }
        }
        
        # Note: For now we'll use spot markets for testing since Kraken futures demo
        # endpoints require different authentication. Once we go live, we can switch to futures.
        
        return config
    
    def print_config_summary(self):
        """Print a summary of current configuration"""
        print("🐵 TradeMonkey Lite Configuration Summary")
        print("=" * 50)
        print(f"Exchange: Kraken ({'Testnet' if self.USE_TESTNET else 'Live'})")
        print(f"Mode: {'DRY RUN (Simulation)' if self.DRY_RUN_MODE else 'LIVE TRADING'}")
        if self.DRY_RUN_MODE:
            print(f"Starting Capital: ${self.STARTING_CAPITAL:,.2f}")
        print(f"Max Positions: {self.MAX_POSITIONS}")
        print(f"Position Size: {self.POSITION_SIZE_PCT * 100}% of capital")
        print(f"Leverage: {self.INITIAL_LEVERAGE}x - {self.MAX_LEVERAGE}x")
        print(f"Trailing Stop: {self.TRAILING_STOP_PCT * 100}%")
        print(f"Take Profits: {[f'{tp*100}%' for tp in self.TAKE_PROFIT_LEVELS]}")
        print(f"Target Symbols: {', '.join(self.TARGET_SYMBOLS)}")
        print(f"Timeframes: {', '.join(self.TIMEFRAMES)}")
        print(f"Notifications: Discord {'✅' if self.DISCORD_WEBHOOK_URL else '❌'}, Telegram {'✅' if self.TELEGRAM_BOT_TOKEN else '❌'}")
        print("=" * 50)


# Global config instance
config = TradingConfig()

# Validate configuration on import
if __name__ == "__main__":
    config.validate_config()
    config.print_config_summary()
else:
    # Silent validation on import
    if not config.validate_config():
        raise ValueError("Configuration validation failed!")
