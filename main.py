#!/usr/bin/env python3
"""
TradeMonkey Lite - Main Entry Point
"In the beginning was the Command Line, and the Command Line was with God" - The Book of Unix

Usage:
    python main.py                  # Run with default settings
    python main.py --config         # Show configuration summary
    python main.py --validate       # Validate configuration only
    python main.py --symbols        # List available trading symbols
    python main.py --live           # Force live trading (override testnet)
    python main.py --paper          # Force paper trading
"""

import sys
import argparse
import asyncio
import logging
from pathlib import Path

# Ensure we can import our modules from the current directory
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import config
from bot import TradeMonkeyLiteEnhanced
import coloredlogs

# Setup logging with style
coloredlogs.install(
    level=config.LOG_LEVEL,
    fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger('TradeMonkeyMain')


def print_banner():
    """Print the epic TradeMonkey banner"""
    banner = """
    ████████╗██████╗  █████╗ ██████╗ ███████╗    ███╗   ███╗ ██████╗ ███╗   ██╗██╗  ██╗███████╗██╗   ██╗
    ╚══██╔══╝██╔══██╗██╔══██╗██╔══██╗██╔════╝    ████╗ ████║██╔═══██╗████╗  ██║██║ ██╔╝██╔════╝╚██╗ ██╔╝
       ██║   ██████╔╝███████║██║  ██║█████╗      ██╔████╔██║██║   ██║██╔██╗ ██║█████╔╝ █████╗   ╚████╔╝ 
       ██║   ██╔══██╗██╔══██║██║  ██║██╔══╝      ██║╚██╔╝██║██║   ██║██║╚██╗██║██╔═██╗ ██╔══╝    ╚██╔╝  
       ██║   ██║  ██║██║  ██║██████╔╝███████╗    ██║ ╚═╝ ██║╚██████╔╝██║ ╚████║██║  ██╗███████╗   ██║   
       ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ ╚══════╝    ╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝   ╚═╝   
    
                                        🐵 LITE EDITION 🐵
                                    "With great leverage comes great responsibility"
                                         - Uncle Ben's crypto cousin
    """
    print(banner)


async def list_available_symbols():
    """List available trading symbols on Kraken"""
    try:
        bot = TradeMonkeyLiteEnhanced()
        logger.info("🐙 Connecting to Kraken and loading markets...")
        await asyncio.to_thread(bot.exchange.load_markets)
        
        logger.info(f"✅ Successfully loaded {len(bot.exchange.markets)} markets from Kraken")
        
        print("🐙 Available Kraken Futures Symbols:")
        print("=" * 60)
        
        # Filter for futures markets more carefully
        future_markets = []
        spot_markets = []
        
        for symbol, market in bot.exchange.markets.items():
            try:
                market_type = market.get('type', 'unknown')
                if market_type == 'future':
                    future_markets.append((symbol, market))
                elif market_type == 'spot':
                    spot_markets.append((symbol, market))
            except Exception as e:
                logger.debug(f"Skipping market {symbol}: {e}")
                continue
        
        # Show futures markets
        if future_markets:
            print("📈 FUTURES MARKETS (For leveraged trading):")
            for symbol, market in sorted(future_markets)[:20]:  # Show first 20
                base = market.get('base', 'N/A')
                quote = market.get('quote', 'N/A')
                print(f"  {symbol:<25} | Base: {base:<8} | Quote: {quote}")
            
            if len(future_markets) > 20:
                print(f"  ... and {len(future_markets) - 20} more futures markets")
        
        # Show some spot markets for reference
        if spot_markets:
            print(f"\n📊 SPOT MARKETS (For reference, showing first 10):")
            for symbol, market in sorted(spot_markets)[:10]:
                base = market.get('base', 'N/A')
                quote = market.get('quote', 'N/A')
                print(f"  {symbol:<25} | Base: {base:<8} | Quote: {quote}")
        
        print(f"\nSUMMARY:")
        print(f"  🎯 Futures markets: {len(future_markets)}")
        print(f"  💰 Spot markets: {len(spot_markets)}")
        print(f"  📊 Total markets: {len(bot.exchange.markets)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to fetch symbols: {e}")
        logger.debug(f"Full error details:", exc_info=True)
        
        # Try to give more helpful debugging info
        try:
            bot = TradeMonkeyLiteEnhanced()
            logger.info("🔍 Attempting basic connection test...")
            
            # Test basic connection
            status = await asyncio.to_thread(bot.exchange.fetch_status)
            logger.info(f"Exchange status: {status}")
            
        except Exception as debug_e:
            logger.error(f"Basic connection also failed: {debug_e}")
            
        return False


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="TradeMonkey Lite - Automated Crypto Trading Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Start trading with default config
  python main.py --paper            # Force paper trading mode
  python main.py --live             # Force live trading mode
  python main.py --config           # Show configuration and exit
  python main.py --symbols          # List available symbols and exit
  python main.py --validate         # Validate config and exit

Remember: "The market can stay irrational longer than you can stay solvent!"
Trade responsibly! 🚀
        """
    )
    
    parser.add_argument('--config', action='store_true',
                       help='Show configuration summary and exit')
    parser.add_argument('--validate', action='store_true',
                       help='Validate configuration and exit')
    parser.add_argument('--symbols', action='store_true',
                       help='List available trading symbols and exit')
    parser.add_argument('--live', action='store_true',
                       help='Force live trading (override testnet setting)')
    parser.add_argument('--paper', action='store_true',
                       help='Force paper trading (override live setting)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    return parser.parse_args()


async def main():
    """Main entry point"""
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Set verbose logging if requested
        if args.verbose:
            coloredlogs.install(level='DEBUG')
            logger.setLevel(logging.DEBUG)
        
        # Print banner
        print_banner()
        
        # Handle mode overrides
        if args.live and args.paper:
            logger.error("❌ Cannot specify both --live and --paper modes!")
            return 1
            
        if args.live:
            config.USE_TESTNET = False
            logger.warning("⚠️  Live trading mode FORCED via command line!")
            
        if args.paper:
            config.USE_TESTNET = True
            logger.info("📝 Paper trading mode forced via command line")
        
        # Handle info commands (these exit after completion)
        if args.config:
            config.print_config_summary()
            return 0
            
        if args.validate:
            if config.validate_config():
                logger.info("✅ Configuration validation passed!")
                return 0
            else:
                logger.error("❌ Configuration validation failed!")
                return 1
                
        if args.symbols:
            success = await list_available_symbols()
            return 0 if success else 1
        
        # Validate configuration before starting
        if not config.validate_config():
            logger.error("❌ Configuration validation failed!")
            return 1
        
        # Show configuration summary
        config.print_config_summary()
        
        # Final confirmation for live trading
        if not config.USE_TESTNET:
            print("\n" + "⚠️ " * 20)
            print("🚨 WARNING: LIVE TRADING MODE ENABLED! 🚨")
            print("💰 This will use REAL money on REAL markets!")
            print("⚠️ " * 20)
            
            response = input("\nType 'RELEASE THE KRAKEN' to continue: ")
            if response != "RELEASE THE KRAKEN":
                logger.info("👋 Trading cancelled by user")
                return 0
        
        # Create and run the bot
        logger.info("🚀 Initializing TradeMonkey Lite...")
        bot = TradeMonkeyLiteEnhanced()
        
        logger.info("🎯 Starting trading engine...")
        await bot.run()
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("👋 Shutdown requested by user")
        return 0
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        return 1


if __name__ == "__main__":
    # Run the main function
    exit_code = asyncio.run(main())
    sys.exit(exit_code)