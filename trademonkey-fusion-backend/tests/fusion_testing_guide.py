#!/usr/bin/env python3
"""
TradeMonkey Fusion - Testing Battle Plan
"The moment of truth - when legends prove their worth" 🚀🔥

TESTING SEQUENCE:
1. Hardware Check (GPU Power Test)
2. Data Pipeline Test (Feed the Beast)
3. Model Training Test (Teach the Beast)
4. Signal Generation Test (Beast Brain Test)
5. Backtest Validation (Prove the Legend)
6. Paper Trading Setup (Kraken Demo Account)
7. Live Paper Trading (The Final Test)
"""

import torch
import pandas as pd
import numpy as np
import asyncio
import ccxt
import json
import os
from datetime import datetime, timedelta
import logging
from pathlib import Path

# Setup legendary logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('FusionTester')

class FusionBeastTester:
    """The Ultimate Testing Suite for Our Legendary Beast"""
    
    def __init__(self):
        self.test_results = {}
        self.gpu_available = torch.cuda.is_available()
        logger.info("🚀 FUSION BEAST TESTER INITIALIZED!")
        
    async def run_complete_test_suite(self):
        """Run the complete testing gauntlet - PROVE OUR LEGEND!"""
        logger.info("🔥 BEGINNING COMPLETE TEST SUITE - LFG!")
        logger.info("=" * 60)
        
        # Phase 1: Hardware Verification
        await self.test_gpu_power()
        
        # Phase 2: Data Pipeline
        await self.test_data_pipeline()
        
        # Phase 3: Model Architecture
        await self.test_model_architecture()
        
        # Phase 4: Feature Engineering
        await self.test_feature_engineering()
        
        # Phase 5: Signal Generation
        await self.test_signal_generation()
        
        # Phase 6: Backtest Engine
        await self.test_backtest_engine()
        
        # Phase 7: Kraken Integration
        await self.test_kraken_integration()
        
        # Final Results
        self.print_test_summary()
        
        return self.test_results
    
    async def test_gpu_power(self):
        """Test 1: Verify GPU Power - The Foundation of Speed"""
        logger.info("🔥 TEST 1: GPU POWER VERIFICATION")
        
        try:
            if not self.gpu_available:
                logger.warning("⚠️  No CUDA GPU detected - will use CPU mode")
                self.test_results['gpu'] = {'available': False, 'performance': 'CPU_ONLY'}
                return
            
            # GPU Info
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            logger.info(f"🎯 GPU Detected: {gpu_name}")
            logger.info(f"💾 GPU Memory: {gpu_memory:.1f} GB")
            
            # Speed Test: Matrix Operations
            logger.info("⚡ Running GPU Speed Test...")
            
            # CPU Test
            cpu_start = datetime.now()
            cpu_tensor = torch.randn(5000, 5000)
            cpu_result = torch.matmul(cpu_tensor, cpu_tensor)
            cpu_time = (datetime.now() - cpu_start).total_seconds()
            
            # GPU Test
            gpu_start = datetime.now()
            gpu_tensor = torch.randn(5000, 5000, device='cuda')
            gpu_result = torch.matmul(gpu_tensor, gpu_tensor)
            torch.cuda.synchronize()  # Wait for GPU to finish
            gpu_time = (datetime.now() - gpu_start).total_seconds()
            
            speedup = cpu_time / gpu_time
            
            logger.info(f"📊 CPU Time: {cpu_time:.3f}s")
            logger.info(f"⚡ GPU Time: {gpu_time:.3f}s")
            logger.info(f"🚀 GPU Speedup: {speedup:.1f}x FASTER!")
            
            self.test_results['gpu'] = {
                'available': True,
                'name': gpu_name,
                'memory_gb': gpu_memory,
                'speedup': speedup,
                'status': 'LEGENDARY' if speedup > 10 else 'GOOD' if speedup > 5 else 'OK'
            }
            
            logger.info("✅ GPU Power Test: PASSED")
            
        except Exception as e:
            logger.error(f"❌ GPU Test Failed: {e}")
            self.test_results['gpu'] = {'error': str(e)}
    
    async def test_data_pipeline(self):
        """Test 2: Data Pipeline - Feed the Beast"""
        logger.info("📊 TEST 2: DATA PIPELINE VERIFICATION")
        
        try:
            # Create sample market data
            sample_data = self.generate_sample_market_data(1000)
            
            logger.info(f"📈 Generated {len(sample_data)} sample bars")
            logger.info(f"📅 Date range: {sample_data.index[0]} to {sample_data.index[-1]}")
            
            # Test data validation
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in sample_data.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Test data quality
            null_counts = sample_data.isnull().sum()
            if null_counts.any():
                logger.warning(f"⚠️  Null values detected: {null_counts[null_counts > 0].to_dict()}")
            
            # Test OHLC logic
            invalid_bars = sample_data[(sample_data['high'] < sample_data['low']) |
                                     (sample_data['high'] < sample_data['open']) |
                                     (sample_data['high'] < sample_data['close']) |
                                     (sample_data['low'] > sample_data['open']) |
                                     (sample_data['low'] > sample_data['close'])]
            
            if len(invalid_bars) > 0:
                logger.warning(f"⚠️  {len(invalid_bars)} invalid OHLC bars detected")
            
            self.test_results['data_pipeline'] = {
                'sample_size': len(sample_data),
                'columns': list(sample_data.columns),
                'null_counts': null_counts.to_dict(),
                'invalid_bars': len(invalid_bars),
                'status': 'PASSED'
            }
            
            # Save sample data for further testing
            sample_data.to_csv('test_data.csv')
            logger.info("💾 Sample data saved to test_data.csv")
            logger.info("✅ Data Pipeline Test: PASSED")
            
        except Exception as e:
            logger.error(f"❌ Data Pipeline Test Failed: {e}")
            self.test_results['data_pipeline'] = {'error': str(e)}
    
    async def test_model_architecture(self):
        """Test 3: Model Architecture - The Brain Test"""
        logger.info("🧠 TEST 3: MODEL ARCHITECTURE VERIFICATION")
        
        try:
            from trademonkey_fusion_core import FusionConfig, FusionTransformer
            
            # Create test configuration
            config = FusionConfig(
                use_gpu=self.gpu_available,
                sequence_length=50,
                hidden_dim=256,  # Smaller for testing
                num_heads=8,
                num_layers=4,    # Smaller for testing
                dropout=0.1
            )
            
            # Initialize model
            model = FusionTransformer(config)
            if self.gpu_available:
                model = model.cuda()
            
            # Test forward pass
            batch_size = 2
            seq_len = config.sequence_length
            input_dim = 28  # Updated feature count
            
            # Create test input
            test_input = torch.randn(batch_size, seq_len, input_dim)
            if self.gpu_available:
                test_input = test_input.cuda()
            
            logger.info(f"🎯 Testing forward pass with input shape: {test_input.shape}")
            
            # Forward pass
            with torch.no_grad():
                outputs = model(test_input)
            
            # Verify outputs
            expected_outputs = ['regime_logits', 'price_prediction', 'volatility_prediction', 'confidence']
            
            for output_name in expected_outputs:
                if output_name not in outputs:
                    raise ValueError(f"Missing output: {output_name}")
                
                output_tensor = outputs[output_name]
                logger.info(f"📊 {output_name}: {output_tensor.shape}")
                
                # Check for NaN values
                if torch.isnan(output_tensor).any():
                    raise ValueError(f"NaN values in {output_name}")
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            logger.info(f"🔢 Total Parameters: {total_params:,}")
            logger.info(f"🎯 Trainable Parameters: {trainable_params:,}")
            
            self.test_results['model_architecture'] = {
                'total_params': total_params,
                'trainable_params': trainable_params,
                'outputs': {name: list(tensor.shape) for name, tensor in outputs.items()},
                'device': 'cuda' if self.gpu_available else 'cpu',
                'status': 'PASSED'
            }
            
            logger.info("✅ Model Architecture Test: PASSED")
            
        except Exception as e:
            logger.error(f"❌ Model Architecture Test Failed: {e}")
            self.test_results['model_architecture'] = {'error': str(e)}
    
    async def test_feature_engineering(self):
        """Test 4: Feature Engineering - The Sensory System"""
        logger.info("⚡ TEST 4: FEATURE ENGINEERING VERIFICATION")
        
        try:
            from trademonkey_fusion_core import GPUAcceleratedFeatures
            
            # Load test data
            df = pd.read_csv('test_data.csv', index_col=0, parse_dates=True)
            
            # Initialize feature engine
            device = 'cuda' if self.gpu_available else 'cpu'
            feature_engine = GPUAcceleratedFeatures(device)
            
            logger.info(f"🔥 Testing feature calculation on {device.upper()}")
            
            # Time the feature calculation
            start_time = datetime.now()
            features = feature_engine.calculate_all_features(df)
            calculation_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"⚡ Feature calculation time: {calculation_time:.3f}s")
            logger.info(f"📊 Features shape: {features.shape}")
            logger.info(f"🎯 Features per bar: {features.shape[1]}")
            
            # Check for NaN values
            nan_count = torch.isnan(features).sum().item()
            if nan_count > 0:
                logger.warning(f"⚠️  {nan_count} NaN values in features")
            
            # Check feature ranges (basic sanity checks)
            feature_stats = {
                'mean': features.mean(dim=0).cpu().numpy(),
                'std': features.std(dim=0).cpu().numpy(),
                'min': features.min(dim=0)[0].cpu().numpy(),
                'max': features.max(dim=0)[0].cpu().numpy()
            }
            
            self.test_results['feature_engineering'] = {
                'calculation_time': calculation_time,
                'feature_count': features.shape[1],
                'data_points': features.shape[0],
                'nan_count': nan_count,
                'device': device,
                'status': 'PASSED'
            }
            
            logger.info("✅ Feature Engineering Test: PASSED")
            
        except Exception as e:
            logger.error(f"❌ Feature Engineering Test Failed: {e}")
            self.test_results['feature_engineering'] = {'error': str(e)}
    
    async def test_signal_generation(self):
        """Test 5: Signal Generation - The Decision Engine"""
        logger.info("🎯 TEST 5: SIGNAL GENERATION VERIFICATION")
        
        try:
            from trademonkey_fusion_core import TradeMonkeyFusionCore, FusionConfig
            
            # Load test data
            df = pd.read_csv('test_data.csv', index_col=0, parse_dates=True)
            
            # Create configuration
            config = FusionConfig(
                use_gpu=self.gpu_available,
                sequence_length=50,
                hidden_dim=256,
                num_heads=8,
                num_layers=4
            )
            
            # Initialize fusion core
            fusion_core = TradeMonkeyFusionCore(config)
            
            logger.info("🧠 Testing signal generation...")
            
            # Generate signal
            start_time = datetime.now()
            signal = await fusion_core.generate_signals(df)
            signal_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"⚡ Signal generation time: {signal_time:.3f}s")
            logger.info(f"🎯 Generated signal: {signal}")
            
            # Validate signal structure
            required_keys = ['action', 'confidence', 'regime', 'signal_strength']
            missing_keys = [key for key in required_keys if key not in signal]
            
            if missing_keys:
                raise ValueError(f"Missing signal keys: {missing_keys}")
            
            # Validate signal values
            if signal['action'] not in ['buy', 'sell', 'hold']:
                raise ValueError(f"Invalid action: {signal['action']}")
            
            if not 0 <= signal['confidence'] <= 1:
                raise ValueError(f"Invalid confidence: {signal['confidence']}")
            
            self.test_results['signal_generation'] = {
                'generation_time': signal_time,
                'signal': signal,
                'action': signal['action'],
                'confidence': signal['confidence'],
                'status': 'PASSED'
            }
            
            logger.info("✅ Signal Generation Test: PASSED")
            
        except Exception as e:
            logger.error(f"❌ Signal Generation Test Failed: {e}")
            self.test_results['signal_generation'] = {'error': str(e)}
    
    async def test_backtest_engine(self):
        """Test 6: Backtest Engine - Prove Our Worth"""
        logger.info("🧪 TEST 6: BACKTEST ENGINE VERIFICATION")
        
        try:
            from trademonkey_fusion_core import TradeMonkeyFusionCore, FusionConfig
            
            # Load test data
            df = pd.read_csv('test_data.csv', index_col=0, parse_dates=True)
            
            # Create configuration
            config = FusionConfig(
                use_gpu=self.gpu_available,
                initial_capital=10000.0,
                position_size_pct=0.25,
                risk_per_trade=0.01
            )
            
            # Initialize fusion core
            fusion_core = TradeMonkeyFusionCore(config)
            
            logger.info("📊 Running sample backtest...")
            
            # Run backtest
            start_time = datetime.now()
            backtest_results = await fusion_core.run_enhanced_backtest(df)
            backtest_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"⚡ Backtest time: {backtest_time:.3f}s")
            logger.info(f"💰 Total Return: {backtest_results['total_return']:.2%}")
            logger.info(f"📊 Win Rate: {backtest_results['win_rate']:.2%}")
            logger.info(f"🎯 Total Trades: {backtest_results['total_trades']}")
            logger.info(f"📉 Max Drawdown: {backtest_results['max_drawdown']:.2%}")
            
            self.test_results['backtest_engine'] = {
                'backtest_time': backtest_time,
                'total_return': backtest_results['total_return'],
                'win_rate': backtest_results['win_rate'],
                'total_trades': backtest_results['total_trades'],
                'max_drawdown': backtest_results['max_drawdown'],
                'status': 'PASSED'
            }
            
            logger.info("✅ Backtest Engine Test: PASSED")
            
        except Exception as e:
            logger.error(f"❌ Backtest Engine Test Failed: {e}")
            self.test_results['backtest_engine'] = {'error': str(e)}
    
    async def test_kraken_integration(self):
        """Test 7: Kraken Integration - Connect to the Market"""
        logger.info("🐙 TEST 7: KRAKEN INTEGRATION VERIFICATION")
        
        try:
            # Test basic Kraken connection
            exchange = ccxt.kraken({
                'enableRateLimit': True,
                'sandbox': True  # Use sandbox mode for testing
            })
            
            logger.info("🔗 Testing Kraken connection...")
            
            # Test markets loading
            markets = await asyncio.to_thread(exchange.load_markets)
            logger.info(f"📊 Loaded {len(markets)} markets from Kraken")
            
            # Test ticker fetch
            ticker = await asyncio.to_thread(exchange.fetch_ticker, 'BTC/USD')
            logger.info(f"💰 BTC/USD Price: ${ticker['last']:.2f}")
            
            # Test OHLCV fetch
            ohlcv = await asyncio.to_thread(exchange.fetch_ohlcv, 'BTC/USD', '1h', limit=10)
            logger.info(f"📈 Fetched {len(ohlcv)} OHLCV bars")
            
            self.test_results['kraken_integration'] = {
                'markets_count': len(markets),
                'btc_price': ticker['last'],
                'ohlcv_bars': len(ohlcv),
                'status': 'PASSED'
            }
            
            logger.info("✅ Kraken Integration Test: PASSED")
            
        except Exception as e:
            logger.error(f"❌ Kraken Integration Test Failed: {e}")
            self.test_results['kraken_integration'] = {'error': str(e)}
    
    def generate_sample_market_data(self, bars: int = 1000) -> pd.DataFrame:
        """Generate realistic sample market data for testing"""
        
        # Start from a year ago
        start_date = datetime.now() - timedelta(days=365)
        dates = pd.date_range(start=start_date, periods=bars, freq='1H')
        
        # Generate realistic price data with trends and volatility
        np.random.seed(42)  # For reproducible results
        
        # Initial price
        price = 50000.0
        prices = []
        volumes = []
        
        for i in range(bars):
            # Add trend component
            trend = np.sin(i / 100) * 0.001  # Slow oscillating trend
            
            # Add random walk
            random_walk = np.random.normal(0, 0.01)
            
            # Occasional large moves (market events)
            if np.random.random() < 0.02:  # 2% chance
                random_walk += np.random.normal(0, 0.05)
            
            # Update price
            price_change = trend + random_walk
            price *= (1 + price_change)
            
            # Generate OHLC
            volatility = abs(np.random.normal(0, 0.005))
            high = price * (1 + volatility)
            low = price * (1 - volatility)
            open_price = price * (1 + np.random.normal(0, 0.002))
            close_price = price
            
            # Ensure OHLC logic
            high = max(high, open_price, close_price)
            low = min(low, open_price, close_price)
            
            prices.append([open_price, high, low, close_price])
            
            # Generate volume (correlated with volatility)
            base_volume = 1000000
            volume_mult = 1 + volatility * 5 + abs(np.random.normal(0, 0.5))
            volume = base_volume * volume_mult
            volumes.append(volume)
        
        # Create DataFrame
        df = pd.DataFrame(prices, columns=['open', 'high', 'low', 'close'], index=dates)
        df['volume'] = volumes
        
        return df
    
    def print_test_summary(self):
        """Print the epic test summary"""
        logger.info("🏆 FUSION BEAST TEST SUMMARY")
        logger.info("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() 
                          if isinstance(result, dict) and result.get('status') == 'PASSED')
        
        logger.info(f"📊 Total Tests: {total_tests}")
        logger.info(f"✅ Passed: {passed_tests}")
        logger.info(f"❌ Failed: {total_tests - passed_tests}")
        
        if passed_tests == total_tests:
            logger.info("🎉 ALL TESTS PASSED - THE BEAST IS READY!")
            logger.info("🚀 READY FOR MARKET DOMINATION!")
        else:
            logger.warning("⚠️  Some tests failed - check logs above")
        
        logger.info("=" * 60)
        
        # Save results
        with open('test_results.json', 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        logger.info("💾 Test results saved to test_results.json")

# KRAKEN PAPER TRADING SETUP
class KrakenPaperTradingSetup:
    """Setup Kraken paper trading account - GET THAT DEMO MONEY!"""
    
    def __init__(self):
        logger.info("🐙 KRAKEN PAPER TRADING SETUP INITIALIZED!")
    
    def setup_kraken_demo_account(self):
        """Step-by-step guide to setup Kraken demo account"""
        
        logger.info("🐙 KRAKEN DEMO ACCOUNT SETUP GUIDE")
        logger.info("=" * 50)
        
        steps = [
            "1. Go to https://demo-futures.kraken.com/",
            "2. Click 'Sign Up' to create demo account",
            "3. Fill out registration (use any email)",
            "4. Verify email and login",
            "5. Navigate to 'Account' → 'API Management'",
            "6. Create new API key with permissions:",
            "   - Query Funds: ✅",
            "   - Query Orders: ✅", 
            "   - Query Trades: ✅",
            "   - Create & Modify Orders: ✅",
            "   - Cancel Orders: ✅",
            "7. Copy API Key and Secret",
            "8. Demo account starts with $100,000 fake money! 💰"
        ]
        
        for step in steps:
            logger.info(step)
        
        logger.info("\n🎯 IMPORTANT NOTES:")
        logger.info("- Demo account resets every 30 days")
        logger.info("- Perfect for testing our BEAST!")
        logger.info("- No real money at risk")
        logger.info("- Same API as live trading")
        
        # Create environment template
        env_template = """
# Kraken Demo Trading Configuration
KRAKEN_API_KEY=your_demo_api_key_here
KRAKEN_API_SECRET=your_demo_api_secret_here
KRAKEN_USE_SANDBOX=true

# Trading Configuration  
INITIAL_CAPITAL=100000
MAX_POSITIONS=4
POSITION_SIZE_PCT=0.25
RISK_PER_TRADE=0.01

# Fusion Beast Settings
USE_GPU=true
MIXED_PRECISION=true
SIGNAL_THRESHOLD=0.65
"""
        
        with open('.env.demo', 'w') as f:
            f.write(env_template)
        
        logger.info("📝 Created .env.demo template")
        logger.info("🔧 Edit .env.demo with your demo API credentials")
        
        return env_template

# MAIN TESTING EXECUTION
async def run_fusion_beast_tests():
    """Execute the complete testing gauntlet - PROVE OUR LEGEND!"""
    
    logger.info("🔥🔥🔥 TRADEMONKEY FUSION BEAST TESTING 🔥🔥🔥")
    logger.info("🚀 PREPARE FOR LEGENDARY PERFORMANCE VERIFICATION!")
    logger.info("⚡ LET'S PROVE THIS BEAST IS READY FOR BATTLE!")
    
    # Initialize tester
    tester = FusionBeastTester()
    
    # Run complete test suite
    results = await tester.run_complete_test_suite()
    
    # Setup Kraken demo account guide
    kraken_setup = KrakenPaperTradingSetup()
    kraken_setup.setup_kraken_demo_account()
    
    return results

if __name__ == "__main__":
    # RUN THE GAUNTLET!
    asyncio.run(run_fusion_beast_tests())
