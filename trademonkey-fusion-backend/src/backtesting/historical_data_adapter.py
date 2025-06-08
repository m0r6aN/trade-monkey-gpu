#!/usr/bin/env python3
"""
Historical Data Adapter - THE TIME MACHINE! 🕰️⚡
"7+ years of BTC data? Hold my coffee while I build an empire!" ☕👑

Features:
- Loads your giant CSV file efficiently
- Converts 1m data to any timeframe (5m, 15m, 1h, 4h, 1d)
- Smart chunking for memory management
- Caches processed timeframes
- Integrates seamlessly with BacktestingEngine
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pickle
import hashlib

logger = logging.getLogger('HistoricalDataAdapter')

@dataclass
class HistoricalPeriod:
    """Define a historical testing period"""
    name: str
    start_date: str
    end_date: str
    description: str

class TimeframeConverter:
    """Converts 1m data to higher timeframes like a boss! 📊"""
    
    @staticmethod
    def convert_timeframe(df_1m: pd.DataFrame, target_timeframe: str) -> pd.DataFrame:
        """Convert 1-minute data to target timeframe"""
        
        # Timeframe mapping
        timeframe_map = {
            '1m': '1T',    # 1 minute
            '5m': '5T',    # 5 minutes  
            '15m': '15T',  # 15 minutes
            '30m': '30T',  # 30 minutes
            '1h': '1H',    # 1 hour
            '4h': '4H',    # 4 hours
            '1d': '1D',    # 1 day
        }
        
        if target_timeframe not in timeframe_map:
            raise ValueError(f"Unsupported timeframe: {target_timeframe}")
        
        if target_timeframe == '1m':
            return df_1m.copy()  # Already 1m data
        
        pandas_freq = timeframe_map[target_timeframe]
        
        # Resample the data
        resampled = df_1m.resample(pandas_freq).agg({
            'open': 'first',
            'high': 'max', 
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        logger.info(f"Converted {len(df_1m)} 1m candles to {len(resampled)} {target_timeframe} candles")
        
        return resampled

class HistoricalDataAdapter:
    """THE TIME MACHINE - Handles your 7+ years of BTC data! 🚀"""
    
    def __init__(self, csv_file_path: str, cache_dir: str = "historical_cache"):
        self.csv_file_path = Path(csv_file_path)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Cache for processed data
        self.timeframe_cache: Dict[str, pd.DataFrame] = {}
        
        # Converter instance
        self.converter = TimeframeConverter()
        
        logger.info(f"🕰️ Historical Data Adapter initialized")
        logger.info(f"📁 CSV file: {self.csv_file_path}")
        logger.info(f"🗄️ Cache dir: {self.cache_dir}")
    
    def _get_cache_key(self, timeframe: str, start_date: str = None, end_date: str = None) -> str:
        """Generate cache key for processed data"""
        key_parts = [timeframe]
        if start_date:
            key_parts.append(start_date)
        if end_date:
            key_parts.append(end_date)
        
        key_str = "_".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()[:16]
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path"""
        return self.cache_dir / f"{cache_key}.parquet"
    
    def load_raw_data(self, chunk_size: int = 100000) -> pd.DataFrame:
        """Load the raw 1m CSV data with memory-efficient chunking"""
        
        logger.info(f"📡 Loading raw data from {self.csv_file_path}")
        
        try:
            # First, let's peek at the actual columns in the file
            sample = pd.read_csv(self.csv_file_path, nrows=1)
            actual_columns = list(sample.columns)
            logger.info(f"🔍 Detected columns: {actual_columns}")
            
            # Check if columns are already in our format or need mapping
            expected_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            old_format_columns = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume']
            
            if all(col in actual_columns for col in expected_columns):
                # Already in our format!
                logger.info("✅ Columns already in expected format")
                columns_to_use = expected_columns
                column_mapping = {}  # No mapping needed
                
            elif all(col in actual_columns for col in old_format_columns):
                # Old format, need mapping
                logger.info("🔄 Using old format columns with mapping")
                columns_to_use = old_format_columns
                column_mapping = {
                    'Open time': 'timestamp',
                    'Open': 'open',
                    'High': 'high', 
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                }
                
            else:
                # Try to auto-detect by looking for similar column names
                logger.info("🔍 Auto-detecting column mapping...")
                column_mapping = {}
                columns_to_use = []
                
                # Smart matching
                for target_col in expected_columns:
                    found_col = None
                    
                    # Look for exact match (case insensitive)
                    for actual_col in actual_columns:
                        if actual_col.lower() == target_col.lower():
                            found_col = actual_col
                            break
                    
                    # Look for partial matches
                    if not found_col:
                        search_terms = {
                            'timestamp': ['time', 'date'],
                            'open': ['open'],
                            'high': ['high'],
                            'low': ['low'], 
                            'close': ['close'],
                            'volume': ['volume', 'vol']
                        }
                        
                        for actual_col in actual_columns:
                            for term in search_terms.get(target_col, []):
                                if term in actual_col.lower():
                                    found_col = actual_col
                                    break
                            if found_col:
                                break
                    
                    if found_col:
                        columns_to_use.append(found_col)
                        if found_col != target_col:
                            column_mapping[found_col] = target_col
                        logger.info(f"  📍 Mapped '{found_col}' → '{target_col}'")
                    else:
                        raise ValueError(f"Could not find column for '{target_col}' in {actual_columns}")
            
            # Load in chunks for memory efficiency
            chunks = []
            chunk_count = 0
            
            for chunk in pd.read_csv(self.csv_file_path, chunksize=chunk_size, usecols=columns_to_use):
                
                # Apply column mapping if needed
                if column_mapping:
                    chunk = chunk.rename(columns=column_mapping)
                
                # Convert timestamp
                chunk['timestamp'] = pd.to_datetime(chunk['timestamp'])
                chunk = chunk.set_index('timestamp')
                
                # Ensure numeric types
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
                
                chunks.append(chunk)
                chunk_count += 1
                
                if chunk_count % 10 == 0:
                    logger.info(f"📊 Processed {chunk_count} chunks ({chunk_count * chunk_size:,} rows)")
            
            # Combine all chunks
            logger.info("🔄 Combining chunks...")
            df = pd.concat(chunks, ignore_index=False)
            
            # Sort by timestamp and remove duplicates
            df = df.sort_index()
            df = df[~df.index.duplicated(keep='first')]
            
            logger.info(f"✅ Loaded {len(df):,} rows of 1m data")
            logger.info(f"📅 Date range: {df.index[0]} to {df.index[-1]}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load raw data: {e}")
            raise
    
    def get_data_for_period(self, timeframe: str, start_date: str, end_date: str, 
                           force_reload: bool = False) -> Optional[pd.DataFrame]:
        """Get data for a specific timeframe and date range"""
        
        cache_key = self._get_cache_key(timeframe, start_date, end_date)
        cache_path = self._get_cache_path(cache_key)
        
        # Try cache first (unless force reload)
        if not force_reload and cache_path.exists():
            try:
                logger.info(f"💾 CACHE HIT: Loading {timeframe} data from cache")
                df = pd.read_parquet(cache_path)
                if 'timestamp' in df.columns:
                    df = df.set_index('timestamp')
                elif not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                return df
            except Exception as e:
                logger.warning(f"Cache load failed: {e}")
        
        logger.info(f"🌐 CACHE MISS: Processing {timeframe} data for {start_date} to {end_date}")
        
        try:
            # Load raw 1m data if not already loaded
            if '1m_full' not in self.timeframe_cache:
                logger.info("📡 Loading full 1m dataset...")
                self.timeframe_cache['1m_full'] = self.load_raw_data()
            
            raw_data = self.timeframe_cache['1m_full']
            
            # Filter to date range
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            filtered_data = raw_data[(raw_data.index >= start_dt) & (raw_data.index <= end_dt)]
            
            if len(filtered_data) == 0:
                logger.warning(f"No data found for period {start_date} to {end_date}")
                return None
            
            # Convert to target timeframe
            converted_data = self.converter.convert_timeframe(filtered_data, timeframe)
            
            if len(converted_data) < 100:
                logger.warning(f"Insufficient data after conversion: {len(converted_data)} candles")
                return None
            
            # Save to cache
            try:
                cache_df = converted_data.reset_index()
                cache_df.to_parquet(cache_path, compression='snappy')
                logger.info(f"💾 Cached {timeframe} data to {cache_path}")
            except Exception as e:
                logger.warning(f"Failed to cache data: {e}")
            
            logger.info(f"✅ Processed {len(converted_data)} {timeframe} candles for {start_date} to {end_date}")
            
            return converted_data
            
        except Exception as e:
            logger.error(f"Failed to process data for period: {e}")
            return None
    
    def get_epic_test_periods(self) -> List[HistoricalPeriod]:
        """Define EPIC historical test periods covering major market cycles! 🎢"""
        
        return [
            # The OG Bull Run
            HistoricalPeriod(
                name="2017_Bull_Mania",
                start_date="2017-08-17", 
                end_date="2017-12-31",
                description="The original crypto mania - BTC $4K to $20K"
            ),
            
            # The Great Crash
            HistoricalPeriod(
                name="2018_Bear_Bloodbath", 
                start_date="2018-01-01",
                end_date="2018-12-31", 
                description="Crypto winter - BTC $20K to $3.2K"
            ),
            
            # The Long Grind
            HistoricalPeriod(
                name="2019_Sideways_Grind",
                start_date="2019-01-01",
                end_date="2019-12-31",
                description="Accumulation phase - slow recovery"
            ),
            
            # COVID Crash & Recovery
            HistoricalPeriod(
                name="2020_COVID_Chaos",
                start_date="2020-01-01", 
                end_date="2020-12-31",
                description="March crash + institutional adoption begins"
            ),
            
            # Institutional FOMO
            HistoricalPeriod(
                name="2021_Institutional_FOMO",
                start_date="2021-01-01",
                end_date="2021-11-30", 
                description="Tesla, MicroStrategy, ATH $69K"
            ),
            
            # The Recent Correction
            HistoricalPeriod(
                name="2022_Reality_Check", 
                start_date="2021-12-01",
                end_date="2022-12-31",
                description="FTX collapse, rate hikes, crypto winter 2.0"
            ),
            
            # Recovery Attempt
            HistoricalPeriod(
                name="2023_Recovery_Hope",
                start_date="2023-01-01",
                end_date="2023-12-31", 
                description="Banking crisis, ETF hopes, slow recovery"
            ),
            
            # ETF Approval Era
            HistoricalPeriod(
                name="2024_ETF_Approval",
                start_date="2024-01-01",
                end_date="2024-12-31",
                description="Spot ETF approval, new ATH attempts"
            )
        ]
    
    def create_custom_period(self, name: str, start_date: str, end_date: str, description: str) -> HistoricalPeriod:
        """Create a custom test period"""
        return HistoricalPeriod(name, start_date, end_date, description)
    
    def preview_data(self, num_rows: int = 10) -> None:
        """Preview the historical data"""
        
        logger.info("👀 Previewing historical data...")
        
        try:
            # Load a small sample
            sample = pd.read_csv(self.csv_file_path, nrows=num_rows)
            
            print("\n📊 DATA PREVIEW:")
            print("=" * 60)
            print(sample.head())
            
            print(f"\n📏 COLUMNS:")
            for i, col in enumerate(sample.columns, 1):
                print(f"  {i:2d}. {col}")
            
            print(f"\n📈 DATA TYPES:")
            print(sample.dtypes)
            
            # Try to get file size
            file_size = self.csv_file_path.stat().st_size / (1024**3)  # GB
            print(f"\n💾 FILE SIZE: {file_size:.2f} GB")
            
        except Exception as e:
            logger.error(f"Preview failed: {e}")
    
    def analyze_data_coverage(self) -> None:
        """Analyze what data coverage we have"""
        
        logger.info("📊 Analyzing data coverage...")
        
        try:
            # Load just the timestamps to analyze coverage
            timestamp_col = pd.read_csv(self.csv_file_path, usecols=['Open time'])
            timestamp_col['Open time'] = pd.to_datetime(timestamp_col['Open time'])
            
            start_date = timestamp_col['Open time'].min()
            end_date = timestamp_col['Open time'].max()
            total_rows = len(timestamp_col)
            
            # Calculate expected vs actual rows
            total_minutes = (end_date - start_date).total_seconds() / 60
            coverage_pct = (total_rows / total_minutes) * 100
            
            print(f"\n📅 DATA COVERAGE ANALYSIS:")
            print("=" * 50)
            print(f"📅 Start Date: {start_date}")
            print(f"📅 End Date: {end_date}")
            print(f"⏱️  Total Period: {(end_date - start_date).days:,} days")
            print(f"📊 Total Rows: {total_rows:,}")
            print(f"⚡ Expected 1m Rows: {total_minutes:,.0f}")
            print(f"✅ Coverage: {coverage_pct:.1f}%")
            
            # Gaps analysis
            if coverage_pct < 95:
                print(f"⚠️  Warning: {100-coverage_pct:.1f}% data missing")
            else:
                print(f"🎯 Excellent coverage!")
                
        except Exception as e:
            logger.error(f"Coverage analysis failed: {e}")

# Integration with BacktestingEngine
def integrate_historical_adapter(backtesting_engine, historical_adapter, test_period: HistoricalPeriod, timeframe: str):
    """Integrate historical data adapter with existing BacktestingEngine"""
    
    # Override the data fetching method
    async def fetch_historical_data(symbol, tf, period):
        """Fetch data from historical adapter instead of API"""
        logger.info(f"📡 Fetching historical data: {symbol} {tf} {period.name}")
        
        # Map the test period to historical period
        return historical_adapter.get_data_for_period(
            timeframe=tf,
            start_date=test_period.start_date, 
            end_date=test_period.end_date
        )
    
    # Replace the API fetcher with our historical fetcher
    backtesting_engine._fetch_raw_data_from_api = fetch_historical_data
    
    return backtesting_engine

# Example usage
if __name__ == "__main__":
    # Initialize the adapter
    adapter = HistoricalDataAdapter("path/to/your/btc_1m_data.csv")
    
    # Preview the data
    adapter.preview_data()
    
    # Analyze coverage  
    adapter.analyze_data_coverage()
    
    # Get epic test periods
    epic_periods = adapter.get_epic_test_periods()
    
    print(f"\n🎢 EPIC TEST PERIODS AVAILABLE:")
    print("=" * 50)
    for period in epic_periods:
        print(f"📅 {period.name}: {period.start_date} to {period.end_date}")
        print(f"   📝 {period.description}")
        print()
    
    # Test loading a period
    print("🧪 Testing data load for 2017 Bull Run...")
    bull_run_data = adapter.get_data_for_period("1h", "2017-10-01", "2017-12-31")
    
    if bull_run_data is not None:
        print(f"✅ Successfully loaded {len(bull_run_data)} hourly candles!")
        print(f"📈 Price range: ${bull_run_data['low'].min():.2f} - ${bull_run_data['high'].max():.2f}")
    else:
        print("❌ Failed to load test data")
