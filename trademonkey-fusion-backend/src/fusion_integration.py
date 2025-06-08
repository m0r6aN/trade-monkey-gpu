#!/usr/bin/env python3
"""
TradeMonkey Fusion - Complete Sentiment Integration
"When AI meets market psychology, profits are born" 🧠💎

This module connects the data pipeline to the sentiment engine
and integrates everything with the main TradeMonkey Fusion system
"""

import asyncio
import json
import redis.asyncio as redis
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np

# Import our sentiment components
from sentiment_engine import EnhancedSentimentAnalyzer, SentimentConfig

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('TradeMonkeyIntegration')

@dataclass
class IntegrationConfig:
    """Configuration for the complete integration system"""
    redis_url: str = "redis://localhost:6379"
    queue_key: str = "sentiment_queue"
    processed_key: str = "processed_sentiment"
    batch_size: int = 10
    polling_interval: int = 5  # seconds
    max_queue_size: int = 1000
    sentiment_weight: float = 0.25  # How much sentiment affects final signal

class SentimentProcessor:
    """Processes sentiment data from the pipeline and feeds TradeMonkey"""
    
    def __init__(self, config: IntegrationConfig = None):
        self.config = config or IntegrationConfig()
        self.redis = redis.from_url(self.config.redis_url)
        
        # Initialize sentiment analyzer
        sentiment_config = SentimentConfig(
            use_ensemble=True,
            batch_size=self.config.batch_size,
            sentiment_weight=self.config.sentiment_weight
        )
        self.sentiment_analyzer = EnhancedSentimentAnalyzer(sentiment_config)
        
        # Performance tracking
        self.processed_count = 0
        self.error_count = 0
        self.last_processed = datetime.now()
        
        logger.info("🔌 SentimentProcessor initialized and ready to rock!")
    
    async def start(self):
        """Start the sentiment processing loop"""
        logger.info("🚀 Starting sentiment processing loop...")
        
        while True:
            try:
                # Pull batch from Redis queue
                batch = await self._get_batch_from_queue()
                
                if batch:
                    # Process sentiment
                    results = await self._process_sentiment_batch(batch)
                    
                    # Store processed results
                    await self._store_results(results)
                    
                    # Update stats
                    self.processed_count += len(batch)
                    self.last_processed = datetime.now()
                    
                    logger.info(f"📊 Processed {len(batch)} texts | Total: {self.processed_count}")
                
                else:
                    # No data, sleep a bit
                    await asyncio.sleep(self.config.polling_interval)
                    
            except Exception as e:
                logger.error(f"❌ Processing error: {e}")
                self.error_count += 1
                await asyncio.sleep(10)  # Backoff on error
    
    async def _get_batch_from_queue(self) -> List[Dict]:
        """Pull a batch of texts from the Redis queue"""
        batch = []
        
        try:
            # Use pipeline for efficiency
            async with self.redis.pipeline() as pipe:
                for _ in range(self.config.batch_size):
                    await pipe.rpop(self.config.queue_key)
                results = await pipe.execute()
            
            # Parse JSON objects
            for result in results:
                if result:
                    try:
                        data = json.loads(result)
                        batch.append(data)
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON in queue: {result}")
                        
        except Exception as e:
            logger.error(f"Queue read error: {e}")
        
        return batch
    
    async def _process_sentiment_batch(self, batch: List[Dict]) -> List[Dict]:
        """Process sentiment for a batch of texts"""
        texts = [item['text'] for item in batch]
        timestamps = [item['timestamp'] for item in batch]
        
        # Run sentiment analysis
        sentiment_results = self.sentiment_analyzer.analyze_batch(texts)
        
        # Combine with timestamps and add metadata
        processed_results = []
        for i, (text_data, sentiment) in enumerate(zip(batch, sentiment_results)):
            result = {
                'original_text': text_data['text'],
                'timestamp': text_data['timestamp'],
                'processed_at': datetime.now().isoformat(),
                'sentiment': sentiment['sentiment'],
                'confidence': sentiment['confidence'],
                'source': sentiment['source']
            }
            processed_results.append(result)
        
        return processed_results
    
    async def _store_results(self, results: List[Dict]):
        """Store processed sentiment results in Redis"""
        try:
            async with self.redis.pipeline() as pipe:
                for result in results:
                    # Store with timestamp-based key for easy retrieval
                    key = f"{self.config.processed_key}:{result['processed_at']}"
                    await pipe.setex(key, 3600, json.dumps(result))  # 1 hour TTL
                
                await pipe.execute()
                
        except Exception as e:
            logger.error(f"Result storage error: {e}")
    
    async def get_recent_sentiment(self, minutes_back: int = 30) -> Dict[str, float]:
        """Get aggregated sentiment from recent results"""
        try:
            # Get temporal sentiment from analyzer (more efficient)
            temporal_sentiment = self.sentiment_analyzer.get_temporal_sentiment(
                hours_back=minutes_back / 60.0
            )
            
            return temporal_sentiment
            
        except Exception as e:
            logger.error(f"Recent sentiment retrieval error: {e}")
            return {
                'avg_sentiment': 0.0,
                'sentiment_trend': 0.0,
                'confidence': 0.0,
                'sample_count': 0
            }

class TradeMonkeyFusionIntegration:
    """Main integration class that connects everything to TradeMonkey Fusion"""
    
    def __init__(self, config: IntegrationConfig = None):
        self.config = config or IntegrationConfig()
        self.sentiment_processor = SentimentProcessor(config)
        
        # Integration state
        self.is_running = False
        self.performance_stats = {
            'total_signals_enhanced': 0,
            'sentiment_boosts': 0,
            'sentiment_dampens': 0,
            'avg_enhancement': 0.0
        }
        
        logger.info("🦍 TradeMonkey Fusion Integration ready for market domination!")
    
    async def start_processing(self):
        """Start the background sentiment processing"""
        if not self.is_running:
            self.is_running = True
            # Run sentiment processor in background
            asyncio.create_task(self.sentiment_processor.start())
            logger.info("✅ Sentiment processing started in background")
    
    async def enhance_trading_signal(self, original_signal: Dict) -> Dict:
        """Enhance a trading signal with real-time sentiment analysis"""
        try:
            # Get recent sentiment data
            sentiment_data = await self.sentiment_processor.get_recent_sentiment(30)
            
            if sentiment_data['sample_count'] < 3:
                # Not enough sentiment data, return original signal
                logger.debug("Insufficient sentiment data for enhancement")
                return original_signal
            
            # Calculate sentiment adjustments
            sentiment_adjustment = self._calculate_sentiment_adjustment(
                original_signal, sentiment_data
            )
            
            # Apply enhancements
            enhanced_signal = original_signal.copy()
            
            # Adjust confidence based on sentiment alignment
            confidence_multiplier = sentiment_adjustment['confidence_multiplier']
            enhanced_signal['confidence'] = min(
                1.0, 
                original_signal.get('confidence', 0.5) * confidence_multiplier
            )
            
            # Adjust position size based on sentiment strength
            if 'position_size' in enhanced_signal:
                sentiment_boost = sentiment_adjustment['signal_strength']
                enhanced_signal['position_size'] = min(
                    enhanced_signal['position_size'] * (1 + sentiment_boost * 0.2),
                    enhanced_signal['position_size'] * 1.5  # Max 50% increase
                )
            
            # Add sentiment metadata
            enhanced_signal['sentiment_analysis'] = {
                'raw_sentiment': sentiment_data['avg_sentiment'],
                'sentiment_trend': sentiment_data['sentiment_trend'],
                'confidence': sentiment_data['confidence'],
                'sample_count': sentiment_data['sample_count'],
                'adjustment_applied': sentiment_adjustment['sentiment_adjustment']
            }
            
            # Update performance stats
            self._update_performance_stats(original_signal, enhanced_signal, sentiment_adjustment)
            
            logger.info(f"🚀 Signal enhanced! Sentiment: {sentiment_data['avg_sentiment']:.3f} | "
                       f"Confidence boost: {confidence_multiplier:.3f}")
            
            return enhanced_signal
            
        except Exception as e:
            logger.error(f"Signal enhancement error: {e}")
            return original_signal  # Return original on error
    
    def _calculate_sentiment_adjustment(self, signal: Dict, sentiment_data: Dict) -> Dict:
        """Calculate how sentiment should adjust the trading signal"""
        raw_sentiment = sentiment_data['avg_sentiment']
        sentiment_confidence = sentiment_data['confidence']
        sentiment_trend = sentiment_data['sentiment_trend']
        
        # Determine signal direction
        signal_direction = 1 if signal.get('action') == 'buy' else -1 if signal.get('action') == 'sell' else 0
        
        # Calculate alignment between sentiment and signal
        sentiment_alignment = raw_sentiment * signal_direction
        
        # Base adjustments
        sentiment_adjustment = raw_sentiment * self.config.sentiment_weight
        
        # Confidence multiplier based on alignment
        if sentiment_alignment > 0:
            # Sentiment agrees with signal - boost confidence
            confidence_multiplier = 1.0 + (abs(sentiment_alignment) * 0.3)
        else:
            # Sentiment disagrees - reduce confidence
            confidence_multiplier = 1.0 - (abs(sentiment_alignment) * 0.2)
        
        # Trend boost
        if abs(sentiment_trend) > 0.1:
            trend_boost = min(abs(sentiment_trend) * 0.5, 0.2)
        else:
            trend_boost = 0.0
        
        # Overall signal strength
        signal_strength = (abs(raw_sentiment) * sentiment_confidence) + trend_boost
        
        return {
            'sentiment_adjustment': sentiment_adjustment,
            'confidence_multiplier': confidence_multiplier,
            'signal_strength': signal_strength,
            'alignment': sentiment_alignment
        }
    
    def _update_performance_stats(self, original: Dict, enhanced: Dict, adjustment: Dict):
        """Update performance tracking stats"""
        self.performance_stats['total_signals_enhanced'] += 1
        
        if adjustment['confidence_multiplier'] > 1.0:
            self.performance_stats['sentiment_boosts'] += 1
        elif adjustment['confidence_multiplier'] < 1.0:
            self.performance_stats['sentiment_dampens'] += 1
        
        # Running average of enhancement strength
        enhancement = abs(adjustment['sentiment_adjustment'])
        total = self.performance_stats['total_signals_enhanced']
        current_avg = self.performance_stats['avg_enhancement']
        self.performance_stats['avg_enhancement'] = (current_avg * (total - 1) + enhancement) / total
    
    async def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        sentiment_data = await self.sentiment_processor.get_recent_sentiment(60)
        
        return {
            'sentiment_processor': {
                'processed_count': self.sentiment_processor.processed_count,
                'error_count': self.sentiment_processor.error_count,
                'last_processed': self.sentiment_processor.last_processed.isoformat(),
                'is_running': self.is_running
            },
            'current_sentiment': sentiment_data,
            'performance_stats': self.performance_stats,
            'system_health': {
                'sentiment_data_fresh': sentiment_data['sample_count'] > 0,
                'error_rate': self.sentiment_processor.error_count / max(1, self.sentiment_processor.processed_count),
                'overall_status': 'healthy' if sentiment_data['sample_count'] > 0 else 'degraded'
            }
        }
    
    async def shutdown(self):
        """Graceful shutdown"""
        self.is_running = False
        await self.sentiment_processor.redis.close()
        logger.info("🛑 TradeMonkey Fusion Integration shutdown complete")

# Integration testing functions
class IntegrationTester:
    """Test the complete integration pipeline"""
    
    def __init__(self):
        self.integration = TradeMonkeyFusionIntegration()
    
    async def test_end_to_end(self):
        """Test the complete pipeline end-to-end"""
        logger.info("🧪 Starting end-to-end integration test...")
        
        # Start processing
        await self.integration.start_processing()
        
        # Simulate some sentiment data
        test_texts = [
            "Bitcoin to the moon! 🚀🚀🚀 This bull run is incredible!",
            "Market crash incoming, everything is dumping hard 📉",
            "Hodling strong with these diamond hands 💎🙌"
        ]
        
        # Inject test data into queue
        await self._inject_test_data(test_texts)
        
        # Wait for processing
        await asyncio.sleep(10)
        
        # Test signal enhancement
        test_signal = {
            'action': 'buy',
            'confidence': 0.7,
            'position_size': 0.25,
            'predicted_return': 0.03
        }
        
        enhanced_signal = await self.integration.enhance_trading_signal(test_signal)
        
        logger.info(f"📊 Original signal: {test_signal}")
        logger.info(f"🚀 Enhanced signal: {enhanced_signal}")
        
        # Get system status
        status = await self.integration.get_system_status()
        logger.info(f"💻 System status: {status}")
        
        logger.info("✅ End-to-end test completed!")
        
        return enhanced_signal, status
    
    async def _inject_test_data(self, texts: List[str]):
        """Inject test data into the sentiment queue"""
        redis_client = redis.from_url("redis://localhost:6379")
        
        try:
            for text in texts:
                data = {
                    'text': text,
                    'timestamp': datetime.now().isoformat()
                }
                await redis_client.lpush("sentiment_queue", json.dumps(data))
            
            logger.info(f"Injected {len(texts)} test texts into queue")
            
        finally:
            await redis_client.close()

# Example usage
async def main():
    """Example of how to use the complete integration"""
    logger.info("🔥 TradeMonkey Fusion Integration Demo!")
    
    # Initialize integration
    integration = TradeMonkeyFusionIntegration()
    
    # Start background processing
    await integration.start_processing()
    
    # Simulate trading signals getting enhanced
    test_signals = [
        {'action': 'buy', 'confidence': 0.7, 'position_size': 0.25},
        {'action': 'sell', 'confidence': 0.6, 'position_size': 0.20},
        {'action': 'hold', 'confidence': 0.4, 'position_size': 0.0}
    ]
    
    for signal in test_signals:
        enhanced = await integration.enhance_trading_signal(signal)
        logger.info(f"Signal: {signal} -> Enhanced: {enhanced}")
        await asyncio.sleep(2)
    
    # Show system status
    status = await integration.get_system_status()
    logger.info(f"System Status: {json.dumps(status, indent=2)}")
    
    # Cleanup
    await integration.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
