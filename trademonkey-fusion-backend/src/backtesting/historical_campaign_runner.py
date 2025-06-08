#!/usr/bin/env python3
"""
Historical Campaign Runner - THE EPIC JOURNEY! 🚀🕰️
"7+ years of BTC data meets our perfected BacktestingEngine!"

This script:
- Loads your historical BTC data
- Runs comprehensive backtests across major market cycles
- Tests multiple timeframes and parameters
- Saves results for epic analysis
"""

import asyncio
import logging
from pathlib import Path
import sys

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from backtesting_engine import BacktestingEngine, BacktestConfig, TestPeriod, MarketCondition
from historical_data_adapter import HistoricalDataAdapter, HistoricalPeriod

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('HistoricalCampaignRunner')

class EpicHistoricalCampaign:
    """Run epic backtests across 7+ years of market history! 🏰"""
    
    def __init__(self, csv_file_path: str, initial_capital: float = 10000):
        self.csv_file_path = csv_file_path
        self.initial_capital = initial_capital
        
        # Initialize components
        self.historical_adapter = HistoricalDataAdapter(csv_file_path)
        self.backtesting_engine = BacktestingEngine(initial_capital, use_futures=False)
        
        # Override the engine's data fetching
        self._integrate_historical_data()
        
        logger.info("🏰 Epic Historical Campaign initialized!")
        logger.info(f"📁 Data file: {csv_file_path}")
        logger.info(f"💰 Capital: ${initial_capital:,.2f}")
    
    def _integrate_historical_data(self):
        """Integrate historical data adapter with backtesting engine"""
        
        # Store original method
        self.original_fetch_method = self.backtesting_engine._fetch_raw_data_from_api
        
        # Create new fetch method that uses our historical data
        async def fetch_historical_data(symbol, timeframe, test_period):
            """Fetch from historical data instead of API"""
            logger.info(f"📡 Fetching historical: {symbol} {timeframe} {test_period.name}")
            
            # Convert BacktestingEngine TestPeriod to our data call
            return self.historical_adapter.get_data_for_period(
                timeframe=timeframe,
                start_date=test_period.start_date,
                end_date=test_period.end_date
            )
        
        # Replace the method
        self.backtesting_engine._fetch_raw_data_from_api = fetch_historical_data
        
        logger.info("✅ Historical data integration complete")
    
    def get_epic_test_periods(self) -> list:
        """Convert HistoricalPeriods to BacktestingEngine TestPeriods"""
        
        historical_periods = self.historical_adapter.get_epic_test_periods()
        
        # Convert to BacktestingEngine format
        test_periods = []
        
        for hp in historical_periods:
            # Map to market conditions based on period name
            if "Bull" in hp.name or "FOMO" in hp.name:
                condition = MarketCondition.BULL_RUN
            elif "Bear" in hp.name or "Crash" in hp.name or "Bloodbath" in hp.name:
                condition = MarketCondition.BEAR_MARKET
            elif "Sideways" in hp.name or "Grind" in hp.name:
                condition = MarketCondition.SIDEWAYS
            else:
                condition = MarketCondition.VOLATILE
            
            test_period = TestPeriod(
                name=hp.name,
                start_date=hp.start_date,
                end_date=hp.end_date,
                expected_condition=condition,
                description=hp.description
            )
            test_periods.append(test_period)
        
        return test_periods
    
    def create_epic_test_matrix(self) -> dict:
        """Create an EPIC test matrix for historical analysis"""
        
        return {
            'symbols': ['BTC/USD'],  # Your data is BTC
            'timeframes': ['15m', '1h', '4h'],  # Multiple timeframes from 1m data
            'signal_thresholds': [60, 75, 90],  # Signal sensitivity
            'position_sizes': [0.1, 0.25, 0.5],  # Position sizing
            'atr_stops': [1.5, 2.0, 2.5],  # Stop loss multipliers
            'atr_profits': [3.0, 4.0, 5.0],  # Take profit multipliers
            'leverage': [1.0]  # Spot trading for historical analysis
        }
        # This gives us: 1×3×8×3×3×3×3×1 = 1,944 configurations across 8 market periods! 🤯
    
    def create_conservative_test_matrix(self) -> dict:
        """Create a smaller test matrix for initial testing"""
        
        return {
            'symbols': ['BTC/USD'],
            'timeframes': ['1h'],  # Just 1h for speed
            'signal_thresholds': [75],  # Just middle threshold
            'position_sizes': [0.25],  # Just middle position size
            'atr_stops': [2.0],  # Just middle stop
            'atr_profits': [3.0, 4.0],  # Two profit targets
            'leverage': [1.0]
        }
        # This gives us: 1×1×8×1×1×1×2×1 = 16 configurations (perfect for testing!)
    
    async def run_historical_campaign(self, test_matrix: dict = None, max_periods: int = None):
        """Run the epic historical campaign!"""
        
        if test_matrix is None:
            test_matrix = self.create_conservative_test_matrix()
        
        logger.info("🏰 STARTING EPIC HISTORICAL CAMPAIGN!")
        
        # Override the engine's test periods with our historical ones
        epic_periods = self.get_epic_test_periods()
        
        if max_periods:
            epic_periods = epic_periods[:max_periods]
            logger.info(f"🎯 Testing first {max_periods} periods only")
        
        self.backtesting_engine.test_periods = epic_periods
        
        # Show what we're about to test
        logger.info(f"📊 Test Matrix: {test_matrix}")
        logger.info(f"📅 Testing {len(epic_periods)} historical periods:")
        
        for period in epic_periods:
            logger.info(f"  🎢 {period.name}: {period.start_date} to {period.end_date}")
            logger.info(f"     📝 {period.description}")
        
        # Calculate total combinations
        total_combinations = (
            len(test_matrix['symbols']) *
            len(test_matrix['timeframes']) *
            len(epic_periods) *
            len(test_matrix['signal_thresholds']) *
            len(test_matrix['position_sizes']) *
            len(test_matrix['atr_stops']) *
            len(test_matrix['atr_profits']) *
            len(test_matrix['leverage'])
        )
        
        logger.info(f"🎯 Total configurations: {total_combinations:,}")
        
        # Run the campaign using the existing engine
        results = await self.backtesting_engine.run_campaign(test_matrix)
        
        logger.info(f"🎉 EPIC CAMPAIGN COMPLETE!")
        logger.info(f"📊 Generated {len(results)} results across {len(epic_periods)} market periods")
        
        return results
    
    def analyze_results_by_period(self, results):
        """Analyze results grouped by market period"""
        
        if not results:
            logger.warning("No results to analyze!")
            return
        
        logger.info("\n📊 RESULTS BY MARKET PERIOD:")
        logger.info("=" * 60)
        
        # Group results by test period
        period_results = {}
        for result in results:
            period_name = result.config.test_period.name
            if period_name not in period_results:
                period_results[period_name] = []
            period_results[period_name].append(result)
        
        # Analyze each period
        for period_name, period_results_list in period_results.items():
            if not period_results_list:
                continue
                
            # Get best result for this period
            valid_results = [r for r in period_results_list if r.total_trades > 0]
            
            if not valid_results:
                logger.info(f"\n🎢 {period_name}: No valid results (no trades)")
                continue
            
            # Sort by Sharpe ratio
            best_result = max(valid_results, key=lambda x: x.sharpe_ratio if not pd.isna(x.sharpe_ratio) else -999)
            
            logger.info(f"\n🎢 {period_name}:")
            logger.info(f"  📊 Configs tested: {len(period_results_list)}")
            logger.info(f"  ✅ Valid results: {len(valid_results)}")
            logger.info(f"  🏆 Best Sharpe: {best_result.sharpe_ratio:.3f}")
            logger.info(f"  💰 Best Return: {best_result.total_return_pct:.2%}")
            logger.info(f"  📈 Best Trades: {best_result.total_trades}")
            logger.info(f"  ⚙️  Best Config: T:{best_result.config.signal_threshold}, "
                       f"ATR:{best_result.config.atr_stop_multiplier}/{best_result.config.atr_profit_multiplier}")
    
    def preview_data(self):
        """Preview the historical data"""
        self.historical_adapter.preview_data()
        self.historical_adapter.analyze_data_coverage()

async def run_epic_campaign(csv_file_path: str, test_mode: str = "conservative"):
    """Main function to run the epic historical campaign"""
    
    logger.info("🚀 EPIC HISTORICAL BACKTESTING CAMPAIGN!")
    logger.info("=" * 60)
    
    # Initialize campaign
    campaign = EpicHistoricalCampaign(csv_file_path)
    
    # Preview data first
    logger.info("👀 Previewing historical data...")
    campaign.preview_data()
    
    # Choose test matrix
    if test_mode == "conservative":
        test_matrix = campaign.create_conservative_test_matrix()
        max_periods = 3  # Test first 3 periods only
        logger.info("🧪 Running CONSERVATIVE test (first 3 periods)")
    elif test_mode == "epic":
        test_matrix = campaign.create_epic_test_matrix()
        max_periods = None  # Test all periods
        logger.info("🏰 Running FULL EPIC campaign (all periods)")
    else:
        raise ValueError("test_mode must be 'conservative' or 'epic'")
    
    # Run the campaign
    results = await campaign.run_historical_campaign(test_matrix, max_periods)
    
    # Analyze results
    campaign.analyze_results_by_period(results)
    
    return results, campaign

if __name__ == "__main__":
    import sys
    
    # Check if CSV file path provided
    if len(sys.argv) < 2:
        print("Usage: python historical_campaign_runner.py <path_to_btc_csv> [test_mode]")
        print("test_mode: 'conservative' (default) or 'epic'")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    test_mode = sys.argv[2] if len(sys.argv) > 2 else "conservative"
    
    # Verify file exists
    if not Path(csv_file).exists():
        print(f"❌ File not found: {csv_file}")
        sys.exit(1)
    
    print(f"🕰️ THE TIME MACHINE ACTIVATES!")
    print(f"📁 Data file: {csv_file}")
    print(f"🎯 Test mode: {test_mode}")
    print("🚀 LFG!")
    
    # Run the epic campaign
    try:
        results, campaign = asyncio.run(run_epic_campaign(csv_file, test_mode))
        print(f"\n🎉 EPIC SUCCESS! {len(results)} backtests completed!")
        print(f"📊 Results saved in backtest_results/ directory")
        print(f"💡 Time to analyze some LEGENDARY data!")
        
    except KeyboardInterrupt:
        print("\n👋 Campaign interrupted by user")
    except Exception as e:
        print(f"\n💥 Campaign failed: {e}")
        logger.error(f"Campaign error: {e}", exc_info=True)