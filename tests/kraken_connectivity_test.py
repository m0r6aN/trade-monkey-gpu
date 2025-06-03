#!/usr/bin/env python3
"""
Kraken Connectivity Diagnostic
"Debugging the Kraken connection, one tentacle at a time!" 🐙🔧
"""

import ccxt
import asyncio
import logging
import requests
import time
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('KrakenDiagnostic')

async def test_kraken_connectivity():
    """Test Kraken connectivity step by step"""
    
    print("🐙 KRAKEN CONNECTIVITY DIAGNOSTIC")
    print("=" * 50)
    
    # Test 1: Basic HTTP connectivity
    print("\n🌐 TEST 1: Basic HTTP Connectivity")
    try:
        response = requests.get("https://api.kraken.com/0/public/Time", timeout=10)
        if response.status_code == 200:
            server_time = response.json()
            print(f"  ✅ Basic HTTP: SUCCESS")
            print(f"  🕐 Server time: {server_time}")
        else:
            print(f"  ❌ Basic HTTP: FAILED - Status {response.status_code}")
            return False
    except Exception as e:
        print(f"  ❌ Basic HTTP: FAILED - {e}")
        return False
    
    # Test 2: Asset Pairs endpoint (the one that's failing)
    print("\n📊 TEST 2: Asset Pairs Endpoint")
    try:
        response = requests.get("https://api.kraken.com/0/public/AssetPairs", timeout=15)
        if response.status_code == 200:
            data = response.json()
            pairs_count = len(data.get('result', {}))
            print(f"  ✅ Asset Pairs: SUCCESS")
            print(f"  📈 Found {pairs_count} trading pairs")
        else:
            print(f"  ❌ Asset Pairs: FAILED - Status {response.status_code}")
            print(f"  📄 Response: {response.text[:200]}...")
            return False
    except Exception as e:
        print(f"  ❌ Asset Pairs: FAILED - {e}")
        return False
    
    # Test 3: CCXT Kraken initialization
    print("\n🔧 TEST 3: CCXT Kraken Initialization")
    try:
        exchange = ccxt.kraken({
            'enableRateLimit': True,
            'rateLimit': 3000,
            'timeout': 30000,  # 30 second timeout
            'options': {'defaultType': 'spot'}
        })
        print(f"  ✅ CCXT Init: SUCCESS")
        print(f"  🏭 Exchange ID: {exchange.id}")
        print(f"  🌍 URLs: {exchange.urls['api']}")
    except Exception as e:
        print(f"  ❌ CCXT Init: FAILED - {e}")
        return False
    
    # Test 4: Load markets with detailed error handling
    print("\n📊 TEST 4: Load Markets (The Failing Step)")
    try:
        print("  🔄 Loading markets...")
        start_time = time.time()
        
        # Use asyncio.wait_for to add a timeout
        markets = await asyncio.wait_for(
            asyncio.to_thread(exchange.load_markets),
            timeout=45.0  # 45 second timeout
        )
        
        end_time = time.time()
        print(f"  ✅ Load Markets: SUCCESS")
        print(f"  ⏱️  Time taken: {end_time - start_time:.2f} seconds")
        print(f"  📊 Loaded {len(markets)} markets")
        
        # Check for our target symbols
        target_symbols = ['BTC/USD', 'ETH/USD', 'BTC/USDT', 'ETH/USDT']
        found_symbols = []
        
        for symbol in target_symbols:
            if symbol in markets:
                found_symbols.append(symbol)
        
        print(f"  🎯 Target symbols found: {found_symbols}")
        
    except asyncio.TimeoutError:
        print(f"  ❌ Load Markets: TIMEOUT (>45 seconds)")
        return False
    except Exception as e:
        print(f"  ❌ Load Markets: FAILED - {e}")
        print(f"  🔍 Error type: {type(e).__name__}")
        return False
    
    # Test 5: Fetch sample OHLCV data
    print("\n📈 TEST 5: Fetch Sample OHLCV Data")
    try:
        symbol = 'BTC/USD'
        print(f"  📡 Fetching {symbol} 1h data...")
        
        ohlcv = await asyncio.wait_for(
            asyncio.to_thread(
                exchange.fetch_ohlcv,
                symbol,
                '1h',
                limit=5
            ),
            timeout=30.0
        )
        
        if ohlcv:
            print(f"  ✅ OHLCV Fetch: SUCCESS")
            print(f"  📊 Received {len(ohlcv)} candles")
            print(f"  💰 Latest price: ${ohlcv[-1][4]:,.2f}")
        else:
            print(f"  ❌ OHLCV Fetch: No data returned")
            
    except Exception as e:
        print(f"  ❌ OHLCV Fetch: FAILED - {e}")
        return False
    
    print(f"\n🎉 ALL TESTS PASSED!")
    print(f"🔧 Kraken connectivity is working properly")
    return True

async def test_specific_date_ranges():
    """Test the specific date ranges we're using"""
    
    print(f"\n📅 TESTING SPECIFIC DATE RANGES")
    print("=" * 40)
    
    try:
        exchange = ccxt.kraken({
            'enableRateLimit': True,
            'rateLimit': 3000,
            'timeout': 30000,
        })
        
        await asyncio.to_thread(exchange.load_markets)
        
        # Test the exact date ranges from our backtester
        test_periods = [
            ("Q1_2024_Bull", "2024-01-01", "2024-03-31"),
            ("Summer_2024_Bear", "2024-06-01", "2024-08-31"),
            ("Fall_2024_Recovery", "2024-09-01", "2024-11-30")
        ]
        
        for name, start_date, end_date in test_periods:
            print(f"\n🗓️  Testing {name}: {start_date} to {end_date}")
            
            try:
                # Convert to timestamps
                start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
                
                ohlcv = await asyncio.to_thread(
                    exchange.fetch_ohlcv,
                    'BTC/USD',
                    '1h',
                    since=start_ts,
                    limit=100
                )
                
                if ohlcv:
                    first_candle = datetime.fromtimestamp(ohlcv[0][0] / 1000)
                    last_candle = datetime.fromtimestamp(ohlcv[-1][0] / 1000)
                    
                    print(f"    ✅ SUCCESS: {len(ohlcv)} candles")
                    print(f"    📅 Range: {first_candle.date()} to {last_candle.date()}")
                else:
                    print(f"    ❌ No data for {name}")
                    
                # Be polite to the API
                await asyncio.sleep(2)
                
            except Exception as e:
                print(f"    ❌ FAILED: {e}")
        
    except Exception as e:
        print(f"❌ Date range test failed: {e}")

async def main():
    """Run all diagnostics"""
    
    # Run connectivity tests
    success = await test_kraken_connectivity()
    
    if success:
        # Run date range tests
        await test_specific_date_ranges()
        
        print(f"\n💡 RECOMMENDATIONS:")
        print(f"  🔧 Your BacktestingEngine should work fine")
        print(f"  🎯 Use symbols: BTC/USD, ETH/USD")
        print(f"  ⏱️  Consider increasing timeout values if you see intermittent failures")
        
    else:
        print(f"\n🚨 ISSUES DETECTED:")
        print(f"  🌐 Check your internet connection")
        print(f"  🔥 Check if there's a firewall blocking Kraken API")
        print(f"  ⏱️  Try running this test again in a few minutes")

if __name__ == "__main__":
    asyncio.run(main())