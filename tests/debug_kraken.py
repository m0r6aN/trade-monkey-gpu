#!/usr/bin/env python3
"""
Quick Kraken API debugging script
"Sometimes you gotta get your hands dirty to find the treasure" - Digital Pirate Wisdom
"""

import ccxt
import json
from config.settings import config

def debug_kraken():
    """Debug Kraken connection and market structure"""
    print("🐙 Kraken API Debug Session")
    print("=" * 50)
    
    try:
        # Create exchange instance
        exchange_config = config.get_exchange_config()
        exchange = ccxt.kraken(exchange_config)
        
        print(f"✅ Exchange created: {exchange.id}")
        print(f"📡 API URL: {exchange.urls['api']}")
        print(f"🧪 Testnet mode: {config.USE_TESTNET}")
        
        # Test basic connection
        print("\n🔍 Testing basic connection...")
        status = exchange.fetch_status()
        print(f"Status: {status}")
        
        # Load markets
        print("\n📊 Loading markets...")
        markets = exchange.load_markets()
        print(f"Total markets loaded: {len(markets)}")
        
        # Analyze market structure
        print("\n🔬 Analyzing market structure...")
        
        if markets:
            # Get first market as example
            first_symbol = list(markets.keys())[0]
            first_market = markets[first_symbol]
            
            print(f"Example market: {first_symbol}")
            print(f"Market structure keys: {list(first_market.keys())}")
            print(f"Market details:")
            for key, value in first_market.items():
                print(f"  {key}: {value}")
            
            # Count market types
            type_counts = {}
            for symbol, market in markets.items():
                market_type = market.get('type', 'unknown')
                type_counts[market_type] = type_counts.get(market_type, 0) + 1
            
            print(f"\nMarket types:")
            for market_type, count in type_counts.items():
                print(f"  {market_type}: {count}")
            
            # Look for our target symbols
            print(f"\n🎯 Checking target symbols:")
            target_symbols = config.TARGET_SYMBOLS
            
            for symbol in target_symbols:
                if symbol in markets:
                    market = markets[symbol]
                    print(f"  ✅ {symbol} - Type: {market.get('type')}, Active: {market.get('active')}")
                else:
                    print(f"  ❌ {symbol} - NOT FOUND")
                    
                    # Try to find similar symbols
                    similar = [s for s in markets.keys() if symbol.split('/')[0] in s][:3]
                    if similar:
                        print(f"     💡 Similar symbols: {', '.join(similar)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_kraken()
