#!/usr/bin/env python3
"""
Kraken Symbol Discovery Tool
"Finding the right names in the Kraken's kingdom!" 🐙🔍
"""

import ccxt
import asyncio
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('SymbolDiscovery')

async def discover_kraken_symbols():
    """Discover what symbols Kraken actually supports"""
    
    try:
        # Initialize Kraken exchange
        exchange = ccxt.kraken({
            'enableRateLimit': True,
            'rateLimit': 3000,
            'options': {'defaultType': 'spot'}
        })
        
        logger.info("🐙 Connecting to Kraken...")
        
        # Load markets
        await asyncio.to_thread(exchange.load_markets)
        
        logger.info(f"✅ Successfully loaded {len(exchange.markets)} markets from Kraken")
        
        # Find BTC-related symbols
        btc_symbols = []
        eth_symbols = []
        all_symbols = []
        
        for symbol, market in exchange.markets.items():
            all_symbols.append(symbol)
            
            # Look for BTC pairs
            if 'BTC' in symbol and ('USD' in symbol or 'USDT' in symbol):
                btc_symbols.append(symbol)
            
            # Look for ETH pairs
            if 'ETH' in symbol and ('USD' in symbol or 'USDT' in symbol):
                eth_symbols.append(symbol)
        
        print("\n🔍 KRAKEN SYMBOL DISCOVERY RESULTS:")
        print("=" * 60)
        
        print(f"\n🪙 BTC Symbols (USD/USDT pairs):")
        for symbol in sorted(btc_symbols):
            market = exchange.markets[symbol]
            print(f"  {symbol:<20} | Base: {market['base']:<8} | Quote: {market['quote']}")
        
        print(f"\n💎 ETH Symbols (USD/USDT pairs):")
        for symbol in sorted(eth_symbols):
            market = exchange.markets[symbol]
            print(f"  {symbol:<20} | Base: {market['base']:<8} | Quote: {market['quote']}")
        
        print(f"\n📊 SUMMARY:")
        print(f"  Total markets: {len(all_symbols)}")
        print(f"  BTC USD/USDT pairs: {len(btc_symbols)}")
        print(f"  ETH USD/USDT pairs: {len(eth_symbols)}")
        
        # Try to fetch some sample data for the most likely symbols
        print(f"\n🧪 TESTING SAMPLE DATA FETCH:")
        
        test_symbols = []
        if btc_symbols:
            test_symbols.append(btc_symbols[0])  # First BTC symbol
        if eth_symbols:
            test_symbols.append(eth_symbols[0])  # First ETH symbol
        
        for symbol in test_symbols[:2]:  # Test max 2 symbols
            try:
                print(f"\n📡 Testing {symbol}...")
                ohlcv = await asyncio.to_thread(
                    exchange.fetch_ohlcv, 
                    symbol, 
                    '1h', 
                    limit=10
                )
                
                if ohlcv:
                    print(f"  ✅ {symbol}: Successfully fetched {len(ohlcv)} candles")
                    print(f"  📈 Latest price: ${ohlcv[-1][4]:,.2f}")
                else:
                    print(f"  ❌ {symbol}: No data returned")
                    
                # Small delay to be polite
                await asyncio.sleep(2)
                
            except Exception as e:
                print(f"  ❌ {symbol}: Error - {e}")
        
        print(f"\n💡 RECOMMENDATIONS:")
        if btc_symbols:
            print(f"  🔧 Use '{btc_symbols[0]}' instead of 'BTC/USD'")
        if eth_symbols:
            print(f"  🔧 Use '{eth_symbols[0]}' instead of 'ETH/USD'")
        
        print(f"\n🚀 UPDATE YOUR TEST MATRIX:")
        print(f"  test_matrix = {{")
        print(f"      'symbols': ['{btc_symbols[0] if btc_symbols else 'BTC/USD'}'],")
        print(f"      'timeframes': ['1h'],")
        print(f"      # ... rest of your matrix")
        print(f"  }}")
        
        return btc_symbols, eth_symbols, all_symbols
        
    except Exception as e:
        logger.error(f"Failed to discover symbols: {e}")
        return [], [], []

async def main():
    """Run the symbol discovery"""
    print("🐙 Kraken Symbol Discovery Tool")
    print("Finding the right symbol names for our backtesting engine!")
    print("=" * 60)
    
    btc_symbols, eth_symbols, all_symbols = await discover_kraken_symbols()
    
    # Save results to a file for reference
    try:
        results_file = Path("kraken_symbols.txt")
        with open(results_file, 'w') as f:
            f.write("Kraken Symbol Discovery Results\n")
            f.write("=" * 40 + "\n\n")
            
            f.write("BTC Symbols:\n")
            for symbol in btc_symbols:
                f.write(f"  {symbol}\n")
            
            f.write("\nETH Symbols:\n")
            for symbol in eth_symbols:
                f.write(f"  {symbol}\n")
            
            f.write(f"\nTotal symbols found: {len(all_symbols)}\n")
        
        print(f"\n📄 Results saved to: {results_file}")
        
    except Exception as e:
        logger.warning(f"Failed to save results file: {e}")

if __name__ == "__main__":
    asyncio.run(main())