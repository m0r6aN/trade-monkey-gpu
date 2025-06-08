#!/usr/bin/env python3
"""
Test different Kraken API endpoint versions
"Sometimes you gotta try all the doors to find the right one" - Digital Explorer Wisdom
"""

import ccxt
import asyncio
from config.settings import config

async def test_endpoint_version(version):
    """Test a specific API version"""
    print(f"\n🧪 Testing API version: v{version}")
    print("-" * 40)
    
    try:
        exchange = ccxt.kraken({
            'apiKey': config.KRAKEN_API_KEY,
            'secret': config.KRAKEN_API_SECRET,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
            },
            'urls': {
                'api': {
                    'public': f'https://demo-futures.kraken.com/derivatives/api/v{version}',
                    'private': f'https://demo-futures.kraken.com/derivatives/api/v{version}',
                }
            }
        })
        
        print(f"📡 API URL: {exchange.urls['api']}")
        
        # Test status
        status = await asyncio.to_thread(exchange.fetch_status)
        print(f"✅ Status: {status.get('status', 'unknown')}")
        
        # Test markets
        markets = await asyncio.to_thread(exchange.load_markets)
        print(f"✅ Markets loaded: {len(markets)}")
        
        if markets:
            # Show some example markets
            symbols = list(markets.keys())[:5]
            print(f"📊 Example symbols: {', '.join(symbols)}")
            
            # Look for our target symbols
            target_found = []
            target_symbols = ['BTC/USD', 'ETH/USD', 'SOL/USD']
            
            for symbol in target_symbols:
                if symbol in markets:
                    target_found.append(symbol)
            
            if target_found:
                print(f"🎯 Target symbols found: {', '.join(target_found)}")
            else:
                print("⚠️  None of our target symbols found")
                
                # Look for Bitcoin-related symbols
                btc_symbols = [s for s in symbols if 'BTC' in s][:3]
                if btc_symbols:
                    print(f"💡 BTC-related symbols: {', '.join(btc_symbols)}")
        
        return True, len(markets)
        
    except Exception as e:
        print(f"❌ Error with v{version}: {e}")
        return False, 0

async def test_spot_fallback():
    """Test spot markets as fallback"""
    print(f"\n🎯 Testing SPOT markets as fallback")
    print("-" * 40)
    
    try:
        exchange = ccxt.kraken({
            'apiKey': config.KRAKEN_API_KEY,
            'secret': config.KRAKEN_API_SECRET,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',  # Use spot instead
            }
        })
        
        # Test markets
        markets = await asyncio.to_thread(exchange.load_markets)
        print(f"✅ Spot markets loaded: {len(markets)}")
        
        if markets:
            # Look for our target symbols in different formats
            target_formats = ['BTC/USD', 'BTC/USDT', 'ETH/USD', 'ETH/USDT']
            found = []
            
            for symbol in target_formats:
                if symbol in markets:
                    found.append(symbol)
            
            if found:
                print(f"🎯 Available in spot: {', '.join(found)}")
                return True, found
        
        return False, []
        
    except Exception as e:
        print(f"❌ Spot test error: {e}")
        return False, []

async def main():
    """Test all options"""
    print("🐙 Kraken API Endpoint Testing")
    print("=" * 50)
    
    # Test different futures API versions
    versions_to_test = [3, 4]
    results = []
    
    for version in versions_to_test:
        success, market_count = await test_endpoint_version(version)
        results.append((version, success, market_count))
    
    # Test spot as fallback
    spot_success, spot_symbols = await test_spot_fallback()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    best_version = None
    best_count = 0
    
    for version, success, count in results:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"Futures API v{version}: {status} ({count} markets)")
        
        if success and count > best_count:
            best_version = version
            best_count = count
    
    spot_status = "✅ SUCCESS" if spot_success else "❌ FAILED"
    print(f"Spot markets:      {spot_status} ({len(spot_symbols) if spot_symbols else 0} target symbols)")
    
    # Recommendation
    print("\n🎯 RECOMMENDATION:")
    if best_version:
        print(f"   Use Futures API v{best_version} with {best_count} markets")
    elif spot_success:
        print(f"   Use Spot markets with symbols: {', '.join(spot_symbols)}")
    else:
        print("   None of the endpoints worked - check API keys")

if __name__ == "__main__":
    asyncio.run(main())
