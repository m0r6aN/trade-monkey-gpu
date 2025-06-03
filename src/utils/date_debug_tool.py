#!/usr/bin/env python3
"""
Date Handling Debug Tool
"Time is an illusion, but accurate timestamps are not!" 🕐🔍
"""

import ccxt
import asyncio
import pandas as pd
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DateDebug')

async def debug_date_handling():
    """Debug our date conversion and fetching logic"""
    
    print("🕐 DATE HANDLING DEBUG TOOL")
    print("=" * 40)
    
    # Initialize exchange
    exchange = ccxt.kraken({
        'enableRateLimit': True,
        'rateLimit': 3000,
        'timeout': 30000,
    })
    
    await asyncio.to_thread(exchange.load_markets)
    
    # Test our exact date conversion logic
    test_periods = [
        ("Q1_2024_Bull", "2024-01-01", "2024-03-31"),
        ("Summer_2024_Bear", "2024-06-01", "2024-08-31"),
        ("Recent_Period", "2025-05-01", "2025-05-31")  # Current month for comparison
    ]
    
    for name, start_date, end_date in test_periods:
        print(f"\n📅 TESTING {name}: {start_date} to {end_date}")
        
        # Our exact conversion logic
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        print(f"  📅 Parsed start: {start_dt}")
        print(f"  📅 Parsed end: {end_dt}")
        
        # Check if dates are in the future
        now = datetime.now()
        print(f"  🕐 Current time: {now}")
        
        if start_dt > now:
            print(f"  ⚠️  WARNING: Start date is in the FUTURE!")
            continue
        
        # Convert to timestamps (our exact logic)
        since = int(start_dt.timestamp() * 1000)
        end_timestamp = int(end_dt.timestamp() * 1000)
        
        print(f"  🔢 Since timestamp: {since}")
        print(f"  🔢 End timestamp: {end_timestamp}")
        
        # Fetch data with our logic
        try:
            print(f"  📡 Fetching data since {since}...")
            
            ohlcv = await asyncio.to_thread(
                exchange.fetch_ohlcv,
                'BTC/USD',
                '1h',
                since=since,
                limit=500
            )
            
            if ohlcv:
                # Convert to DataFrame like our engine does
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                
                # Filter to date range (our exact logic)
                period_candles = [c for c in ohlcv if since <= c[0] <= end_timestamp]
                
                print(f"  📊 Total candles received: {len(ohlcv)}")
                print(f"  🎯 Candles in date range: {len(period_candles)}")
                
                if ohlcv:
                    first_candle = datetime.fromtimestamp(ohlcv[0][0] / 1000)
                    last_candle = datetime.fromtimestamp(ohlcv[-1][0] / 1000)
                    
                    print(f"  📅 Actual data range: {first_candle} to {last_candle}")
                    
                    # Check if any candles are actually in our target range
                    if period_candles:
                        first_in_range = datetime.fromtimestamp(period_candles[0][0] / 1000)
                        last_in_range = datetime.fromtimestamp(period_candles[-1][0] / 1000)
                        print(f"  🎯 Filtered range: {first_in_range} to {last_in_range}")
                    else:
                        print(f"  ❌ NO CANDLES in target date range!")
                        print(f"  💡 This explains why our backtester gets no data!")
                
            else:
                print(f"  ❌ No data returned")
                
            # Small delay
            await asyncio.sleep(2)
            
        except Exception as e:
            print(f"  ❌ Error: {e}")

async def test_recent_vs_historical():
    """Test fetching recent data vs historical data"""
    
    print(f"\n🔍 RECENT VS HISTORICAL DATA TEST")
    print("=" * 50)
    
    exchange = ccxt.kraken({
        'enableRateLimit': True,
        'rateLimit': 3000,
        'timeout': 30000,
    })
    
    await asyncio.to_thread(exchange.load_markets)
    
    # Test 1: Recent data (should work)
    print(f"\n📊 TEST 1: Recent Data (Last 7 days)")
    try:
        recent_data = await asyncio.to_thread(
            exchange.fetch_ohlcv,
            'BTC/USD',
            '1h',
            limit=168  # 7 days * 24 hours
        )
        
        if recent_data:
            first = datetime.fromtimestamp(recent_data[0][0] / 1000)
            last = datetime.fromtimestamp(recent_data[-1][0] / 1000)
            print(f"  ✅ Recent data: {len(recent_data)} candles")
            print(f"  📅 Range: {first} to {last}")
        
    except Exception as e:
        print(f"  ❌ Recent data failed: {e}")
    
    # Test 2: Historical data with specific timestamp
    print(f"\n📊 TEST 2: Historical Data (January 2024)")
    try:
        jan_2024 = datetime(2024, 1, 1)
        since_ts = int(jan_2024.timestamp() * 1000)
        
        print(f"  🎯 Requesting data since: {jan_2024}")
        print(f"  🔢 Timestamp: {since_ts}")
        
        historical_data = await asyncio.to_thread(
            exchange.fetch_ohlcv,
            'BTC/USD',
            '1h',
            since=since_ts,
            limit=500
        )
        
        if historical_data:
            first = datetime.fromtimestamp(historical_data[0][0] / 1000)
            last = datetime.fromtimestamp(historical_data[-1][0] / 1000)
            print(f"  📊 Historical data: {len(historical_data)} candles")
            print(f"  📅 Actual range: {first} to {last}")
            
            # Check if we got 2024 data
            if first.year == 2024:
                print(f"  ✅ SUCCESS: Got 2024 data as requested!")
            else:
                print(f"  ❌ PROBLEM: Asked for 2024, got {first.year} data!")
                print(f"  💡 This suggests Kraken might not have that historical data")
        
    except Exception as e:
        print(f"  ❌ Historical data failed: {e}")

async def main():
    """Run all date debugging tests"""
    await debug_date_handling()
    await test_recent_vs_historical()
    
    print(f"\n💡 ANALYSIS:")
    print(f"  🔍 If we're getting 2025 data when requesting 2024...")
    print(f"  📊 Either Kraken doesn't have that historical data")
    print(f"  🕐 Or our date conversion has a bug")
    print(f"  ⚙️  Or Kraken ignores 'since' param for very old data")

if __name__ == "__main__":
    asyncio.run(main())