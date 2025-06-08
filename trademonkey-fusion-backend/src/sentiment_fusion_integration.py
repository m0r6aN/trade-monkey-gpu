#!/usr/bin/env python3
"""
TradeMonkey Fusion - Complete Sentiment Integration
"When sentiment meets signals, profits are INEVITABLE!" 💎🚀

This module connects our enhanced sentiment engine to the main TradeMonkey Fusion system
Authors: Grok + Claude (The Dream Team)
Date: June 5, 2025 - The Day We Achieved Market Telepathy
"""

import asyncio
import json
import redis.asyncio as redis
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import pickle
from pathlib import Path

# Import our enhanced sentiment engine
from sentiment_engine_v2 import (
    LightweightSentimentEnsemble, 
    FusionSentimentConfig,
    GPUMemoryOptimizer
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('TradeMonkeyFusionIntegration')

@dataclass
class FusionIntegrationConfig:
    """Configuration for complete TradeMonkey Fusion integration"""
    # Redis configuration
    redis_url: str = "redis://localhost:6379"
    sentiment_queue: str = "trademonkey:sentiment:queue"
    processed_queue: str = "trademonkey:sentiment:processed"
    signal_enhancement_key: str = "trademonkey:signals:enhanced"
    
    # Processing parameters
    batch_size: int = 50
    polling_interval: float = 2.0  # seconds
    max_queue_size: int = 10000
    
    # Sentiment integration weights
    sentiment_weight: float = 0.3  # How much sentiment affects final signal
    confidence_threshold: float = 0.6  # Minimum confidence to apply sentiment
    crypto_boost_multiplier: float = 1.5  # Extra boost for crypto-related content
    
    # Performance settings
    max_processing_time: float = 30.0  # Max seconds for batch processing
    error_retry_attempts: int = 3
    error_backoff_seconds: float = 5.0
    
    # Historical sentiment settings
    sentiment_memory_hours: int = 6  # How long to keep sentiment history
    trend_analysis_window: int = 100  # Number of recent sentiments for trend
    
    # Signal enhancement parameters
    max_confidence_boost: float = 0.4  # Maximum confidence boost from sentiment
    max_position_size_boost: float = 0.3  # Maximum position size boost
    sentiment_alignment_threshold: float = 0.7  # Threshold for strong alignment

class SentimentHistoryManager:
    """Manages historical sentiment data for trend analysis"""
    
    def __init__(self, config: FusionIntegrationConfig):
        self.config = config
        self.sentiment_history: List[Dict] = []
        self.redis_client = None
        
    async def initialize_redis(self):
        """Initialize Redis connection"""
        self.redis_client = redis.from_url(self.config.redis_url)
        
        # Load existing history from Redis
        try:
            history_data = await self.redis_client.get("trademonkey:sentiment:history")
            if history_data:
                self.sentiment_history = json.loads(history_data)
                logger.info(f"📂 Loaded {len(self.sentiment_history)} historical sentiment entries")
        except Exception as e:
            logger.warning(f"Could not load sentiment history: {e}")
    
    async def add_sentiment(self, sentiment_data: Dict):
        """Add new sentiment data to history"""
        sentiment_entry = {
            'timestamp': datetime.now().isoformat(),
            'sentiment': sentiment_data['sentiment'],
            'confidence': sentiment_data['confidence'],
            'crypto_related': sentiment_data.get('crypto_related', False),
            'source': sentiment_data.get('source', 'unknown')
        }
        
        self.sentiment_history.append(sentiment_entry)
        
        # Cleanup old entries
        cutoff_time = datetime.now() - timedelta(hours=self.config.sentiment_memory_hours)
        self.sentiment_history = [
            entry for entry in self.sentiment_history
            if datetime.fromisoformat(entry['timestamp']) > cutoff_time
        ]
        
        # Save to Redis
        if self.redis_client:
            try:
                await self.redis_client.setex(
                    "trademonkey:sentiment:history",
                    3600 * self.config.sentiment_memory_hours,
                    json.dumps(self.sentiment_history[-1000:])  # Keep last 1000 entries
                )
            except Exception as e:
                logger.warning(f"Could not save sentiment history: {e}")
    
    def get_sentiment_trend(self, hours_back: float = 1.0) -> Dict[str, float]:
        """Calculate sentiment trend over specified time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        recent_sentiments = [
            entry for entry in self.sentiment_history
            if datetime.fromisoformat(entry['timestamp']) > cutoff_time
        ]
        
        if len(recent_sentiments) < 3:
            return {
                'avg_sentiment': 0.0,
                'sentiment_trend': 0.0,
                'confidence': 0.0,
                'sample_count': 0,
                'crypto_ratio': 0.0
            }
        
        # Calculate metrics
        sentiments = [entry['sentiment'] for entry in recent_sentiments]
        confidences = [entry['confidence'] for entry in recent_sentiments]
        crypto_count = sum(1 for entry in recent_sentiments if entry['crypto_related'])
        
        avg_sentiment = np.mean(sentiments)
        avg_confidence = np.mean(confidences)
        crypto_ratio = crypto_count / len(recent_sentiments)
        
        # Calculate trend (recent vs older)
        if len(sentiments) >= 6:
            recent_half = sentiments[len(sentiments)//2:]
            older_half = sentiments[:len(sentiments)//2]
            sentiment_trend = np.mean(recent_half) - np.mean(older_half)
        else:
            sentiment_trend = 0.0
        
        return {
            'avg_sentiment': float(avg_sentiment),
            'sentiment_trend': float(sentiment_trend),
            'confidence': float(avg_confidence),
            'sample_count': len(recent_sentiments),
            'crypto_ratio': float(crypto_ratio)
        }
    
    def get_sentiment_volatility(self, hours_back: float = 2.0) -> float:
        """Calculate sentiment volatility (standard deviation)"""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        recent_sentiments = [
            entry['sentiment'] for entry in self.sentiment_history
            if datetime.fromisoformat(entry['timestamp']) > cutoff_time
        ]
        
        if len(recent_sentiments) < 3:
            return 0.0
        
        return float(np.std(recent_sentiments))

class EnhancedSentimentProcessor:
    """Enhanced sentiment processor with GPU optimization and smart batching"""
    
    def __init__(self, config: FusionIntegrationConfig):
        self.config = config
        
        # Initialize sentiment engine
        sentiment_config = FusionSentimentConfig(
            device="cuda" if torch.cuda.is_available() else "cpu",
            batch_size=config.batch_size,
            mixed_precision=True
        )
        
        self.sentiment_engine = LightweightSentimentEnsemble(sentiment_config)
        self.gpu_optimizer = GPUMemoryOptimizer(sentiment_config)
        self.history_manager = SentimentHistoryManager(config)
        
        # Redis connections
        self.redis_client = None
        
        # Performance tracking
        self.stats = {
            'total_processed': 0,
            'total_errors': 0,
            'avg_processing_time': 0.0,
            'last_batch_size': 0,
            'gpu_memory_usage': 0.0
        }
        
        logger.info("🧠 Enhanced Sentiment Processor initialized!")
    
    async def initialize(self):
        """Initialize all components"""
        # Initialize Redis
        self.redis_client = redis.from_url(self.config.redis_url)
        await self.history_manager.initialize_redis()
        
        logger.info("✅ Sentiment processor fully initialized!")
    
    async def start_processing_loop(self):
        """Start the main sentiment processing loop"""
        logger.info("🚀 Starting sentiment processing loop...")
        
        while True:
            try:
                # Get batch from queue
                batch_start_time = datetime.now()
                texts_batch = await self._get_batch_from_queue()
                
                if texts_batch:
                    # Process sentiment
                    results = await self._process_sentiment_batch(texts_batch)
                    
                    # Store results and update history
                    await self._store_batch_results(results)
                    
                    # Update stats
                    processing_time = (datetime.now() - batch_start_time).total_seconds()
                    self._update_performance_stats(len(texts_batch), processing_time)
                    
                    logger.info(f"📊 Processed {len(texts_batch)} texts in {processing_time:.2f}s")
                
                else:
                    # No data, sleep briefly
                    await asyncio.sleep(self.config.polling_interval)
                
            except Exception as e:
                logger.error(f"❌ Processing loop error: {e}")
                self.stats['total_errors'] += 1
                await asyncio.sleep(self.config.error_backoff_seconds)
    
    async def _get_batch_from_queue(self) -> List[Dict]:
        """Get a batch of texts from Redis queue with smart sizing"""
        batch = []
        
        try:
            # Optimize batch size based on GPU memory
            current_memory = self.gpu_optimizer.get_memory_stats()
            if current_memory['cupy_free_gb'] < 1.0:  # Less than 1GB free
                batch_size = min(self.config.batch_size, 20)
            else:
                batch_size = self.config.batch_size
            
            # Get batch from Redis
            async with self.redis_client.pipeline() as pipe:
                for _ in range(batch_size):
                    await pipe.rpop(self.config.sentiment_queue)
                results = await pipe.execute()
            
            # Parse results
            for result in results:
                if result:
                    try:
                        data = json.loads(result)
                        batch.append(data)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON in queue: {result[:100]}")
                        continue
        
        except Exception as e:
            logger.error(f"Queue read error: {e}")
        
        return batch
    
    async def _process_sentiment_batch(self, batch: List[Dict]) -> List[Dict]:
        """Process sentiment for a batch with GPU optimization"""
        texts = [item['text'] for item in batch]
        
        # Clear GPU cache if memory is getting tight
        memory_stats = self.gpu_optimizer.get_memory_stats()
        if memory_stats['total_used_gb'] > 8.0:  # More than 8GB used
            self.gpu_optimizer.clear_cache()
        
        # Process with sentiment engine
        sentiment_results = await self.sentiment_engine.predict_batch(texts)
        
        # Combine with original data
        processed_results = []
        for original_data, sentiment in zip(batch, sentiment_results):
            result = {
                'text': original_data['text'],
                'timestamp': original_data.get('timestamp', datetime.now().isoformat()),
                'source': original_data.get('source', 'unknown'),
                'processed_at': datetime.now().isoformat(),
                'sentiment': sentiment['sentiment'],
                'confidence': sentiment['confidence'],
                'crypto_related': sentiment.get('crypto_related', False),
                'finbert_score': sentiment.get('finbert_score', 0.0),
                'distilbert_score': sentiment.get('distilbert_score', 0.0),
                'vader_score': sentiment.get('vader_score', 0.0)
            }
            processed_results.append(result)
            
            # Add to history
            await self.history_manager.add_sentiment(sentiment)
        
        return processed_results
    
    async def _store_batch_results(self, results: List[Dict]):
        """Store processed results in Redis"""
        try:
            async with self.redis_client.pipeline() as pipe:
                for result in results:
                    # Store individual result with timestamp key
                    key = f"{self.config.processed_queue}:{result['processed_at']}"
                    await pipe.setex(key, 3600, json.dumps(result))  # 1 hour TTL
                
                await pipe.execute()
        
        except Exception as e:
            logger.error(f"Result storage error: {e}")
    
    def _update_performance_stats(self, batch_size: int, processing_time: float):
        """Update performance statistics"""
        self.stats['total_processed'] += batch_size
        self.stats['last_batch_size'] = batch_size
        
        # Update running average of processing time
        current_avg = self.stats['avg_processing_time']
        total_batches = self.stats['total_processed'] / self.config.batch_size
        self.stats['avg_processing_time'] = (current_avg * (total_batches - 1) + processing_time) / total_batches
        
        # Update GPU memory usage
        memory_stats = self.gpu_optimizer.get_memory_stats()
        self.stats['gpu_memory_usage'] = memory_stats['total_used_gb']
    
    async def get_current_market_sentiment(self) -> Dict[str, float]:
        """Get current aggregated market sentiment"""
        # Get sentiment trend from history
        sentiment_trend = self.history_manager.get_sentiment_trend(hours_back=1.0)
        
        # Get volatility
        sentiment_volatility = self.history_manager.get_sentiment_volatility(hours_back=2.0)
        
        # Calculate market sentiment signal
        market_sentiment = {
            'avg_sentiment': sentiment_trend['avg_sentiment'],
            'sentiment_trend': sentiment_trend['sentiment_trend'],
            'confidence': sentiment_trend['confidence'],
            'volatility': sentiment_volatility,
            'sample_count': sentiment_trend['sample_count'],
            'crypto_ratio': sentiment_trend['crypto_ratio'],
            'last_updated': datetime.now().isoformat()
        }
        
        return market_sentiment

class TradeMonkeySignalEnhancer:
    """Enhanced signal processing with sentiment integration"""
    
    def __init__(self, config: FusionIntegrationConfig, sentiment_processor: EnhancedSentimentProcessor):
        self.config = config
        self.sentiment_processor = sentiment_processor
        
        # Performance tracking
        self.enhancement_stats = {
            'total_signals_processed': 0,
            'sentiment_boosts': 0,
            'sentiment_dampens': 0,
            'avg_enhancement_magnitude': 0.0,
            'high_confidence_enhancements': 0
        }
        
        logger.info("🎯 TradeMonkey Signal Enhancer initialized!")
    
    async def enhance_trading_signal(self, original_signal: Dict, symbol: str = None) -> Dict:
        """Enhance trading signal with real-time sentiment analysis"""
        try:
            # Get current market sentiment
            market_sentiment = await self.sentiment_processor.get_current_market_sentiment()
            
            # Check if we have sufficient sentiment data
            if market_sentiment['sample_count'] < 5:
                logger.debug("Insufficient sentiment data for enhancement")
                return self._add_sentiment_metadata(original_signal, market_sentiment, applied=False)
            
            # Calculate sentiment adjustments
            enhancement = self._calculate_signal_enhancement(original_signal, market_sentiment, symbol)
            
            # Apply enhancements
            enhanced_signal = self._apply_enhancements(original_signal, enhancement, market_sentiment)
            
            # Update stats
            self._update_enhancement_stats(enhancement)
            
            logger.info(f"🚀 Signal enhanced! Sentiment: {market_sentiment['avg_sentiment']:.3f} | "
                       f"Boost: {enhancement['confidence_multiplier']:.3f}")
            
            return enhanced_signal
            
        except Exception as e:
            logger.error(f"Signal enhancement error: {e}")
            return original_signal
    
    def _calculate_signal_enhancement(self, signal: Dict, market_sentiment: Dict, symbol: str = None) -> Dict:
        """Calculate how sentiment should enhance the trading signal"""
        avg_sentiment = market_sentiment['avg_sentiment']
        sentiment_confidence = market_sentiment['confidence']
        sentiment_trend = market_sentiment['sentiment_trend']
        crypto_ratio = market_sentiment['crypto_ratio']
        
        # Determine signal direction
        signal_action = signal.get('action', 'hold')
        if signal_action == 'buy':
            signal_direction = 1
        elif signal_action == 'sell':
            signal_direction = -1
        else:
            signal_direction = 0
        
        # Calculate sentiment alignment
        sentiment_alignment = avg_sentiment * signal_direction
        
        # Base confidence multiplier
        if sentiment_alignment > self.config.sentiment_alignment_threshold:
            # Strong positive alignment
            confidence_multiplier = 1.0 + (sentiment_alignment * self.config.max_confidence_boost)
        elif sentiment_alignment < -self.config.sentiment_alignment_threshold:
            # Strong negative alignment - reduce confidence
            confidence_multiplier = 1.0 + (sentiment_alignment * self.config.max_confidence_boost * 0.5)
        else:
            # Weak alignment - minimal effect
            confidence_multiplier = 1.0 + (sentiment_alignment * 0.1)
        
        # Trend boost
        if abs(sentiment_trend) > 0.1:
            trend_multiplier = 1.0 + (abs(sentiment_trend) * 0.2)
        else:
            trend_multiplier = 1.0
        
        # Crypto boost
        crypto_multiplier = 1.0 + (crypto_ratio * self.config.crypto_boost_multiplier * 0.1)
        
        # Position size adjustment
        if sentiment_alignment > 0.5:
            position_size_multiplier = 1.0 + (sentiment_alignment * self.config.max_position_size_boost)
        else:
            position_size_multiplier = 1.0
        
        # Overall enhancement strength
        enhancement_strength = abs(avg_sentiment) * sentiment_confidence * crypto_multiplier
        
        return {
            'confidence_multiplier': float(np.clip(confidence_multiplier * trend_multiplier, 0.5, 2.0)),
            'position_size_multiplier': float(np.clip(position_size_multiplier, 0.7, 1.5)),
            'enhancement_strength': float(enhancement_strength),
            'sentiment_alignment': float(sentiment_alignment),
            'trend_multiplier': float(trend_multiplier),
            'crypto_multiplier': float(crypto_multiplier)
        }
    
    def _apply_enhancements(self, original_signal: Dict, enhancement: Dict, market_sentiment: Dict) -> Dict:
        """Apply calculated enhancements to the trading signal"""
        enhanced_signal = original_signal.copy()
        
        # Enhance confidence
        original_confidence = enhanced_signal.get('confidence', 0.5)
        enhanced_signal['confidence'] = float(np.clip(
            original_confidence * enhancement['confidence_multiplier'], 
            0.0, 1.0
        ))
        
        # Enhance position size if present
        if 'position_size' in enhanced_signal:
            original_size = enhanced_signal['position_size']
            enhanced_signal['position_size'] = float(np.clip(
                original_size * enhancement['position_size_multiplier'],
                original_size * 0.5,  # Minimum 50% of original
                original_size * 2.0   # Maximum 200% of original
            ))
        
        # Add sentiment metadata
        enhanced_signal['sentiment_analysis'] = {
            'market_sentiment': market_sentiment['avg_sentiment'],
            'sentiment_trend': market_sentiment['sentiment_trend'],
            'sentiment_confidence': market_sentiment['confidence'],
            'sample_count': market_sentiment['sample_count'],
            'crypto_ratio': market_sentiment['crypto_ratio'],
            'enhancement_applied': True,
            'confidence_boost': enhancement['confidence_multiplier'] - 1.0,
            'position_boost': enhancement['position_size_multiplier'] - 1.0,
            'alignment_score': enhancement['sentiment_alignment'],
            'enhancement_strength': enhancement['enhancement_strength']
        }
        
        return enhanced_signal
    
    def _add_sentiment_metadata(self, signal: Dict, market_sentiment: Dict, applied: bool = False) -> Dict:
        """Add sentiment metadata without applying enhancements"""
        enhanced_signal = signal.copy()
        enhanced_signal['sentiment_analysis'] = {
            'market_sentiment': market_sentiment.get('avg_sentiment', 0.0),
            'sentiment_trend': market_sentiment.get('sentiment_trend', 0.0),
            'sentiment_confidence': market_sentiment.get('confidence', 0.0),
            'sample_count': market_sentiment.get('sample_count', 0),
            'crypto_ratio': market_sentiment.get('crypto_ratio', 0.0),
            'enhancement_applied': applied,
            'reason': 'insufficient_data' if not applied else 'applied'
        }
        return enhanced_signal
    
    def _update_enhancement_stats(self, enhancement: Dict):
        """Update enhancement statistics"""
        self.enhancement_stats['total_signals_processed'] += 1
        
        if enhancement['confidence_multiplier'] > 1.1:
            self.enhancement_stats['sentiment_boosts'] += 1
        elif enhancement['confidence_multiplier'] < 0.9:
            self.enhancement_stats['sentiment_dampens'] += 1
        
        if enhancement['enhancement_strength'] > 0.7:
            self.enhancement_stats['high_confidence_enhancements'] += 1
        
        # Update running average
        total = self.enhancement_stats['total_signals_processed']
        current_avg = self.enhancement_stats['avg_enhancement_magnitude']
        enhancement_mag = abs(enhancement['confidence_multiplier'] - 1.0)
        self.enhancement_stats['avg_enhancement_magnitude'] = (
            (current_avg * (total - 1) + enhancement_mag) / total
        )
    
    def get_enhancement_stats(self) -> Dict:
        """Get enhancement performance statistics"""
        total = self.enhancement_stats['total_signals_processed']
        if total == 0:
            return self.enhancement_stats
        
        stats = self.enhancement_stats.copy()
        stats['boost_rate'] = self.enhancement_stats['sentiment_boosts'] / total
        stats['dampen_rate'] = self.enhancement_stats['sentiment_dampens'] / total
        stats['high_confidence_rate'] = self.enhancement_stats['high_confidence_enhancements'] / total
        
        return stats

# Main integration class
class TradeMonkeyFusionIntegration:
    """Complete integration of sentiment analysis with TradeMonkey Fusion"""
    
    def __init__(self, config: FusionIntegrationConfig = None):
        self.config = config or FusionIntegrationConfig()
        
        # Initialize components
        self.sentiment_processor = EnhancedSentimentProcessor(self.config)
        self.signal_enhancer = TradeMonkeySignalEnhancer(self.config, self.sentiment_processor)
        
        # System state
        self.is_running = False
        self.background_tasks = []
        
        logger.info("🦍 TradeMonkey Fusion Integration ready for market domination!")
    
    async def initialize(self):
        """Initialize all components"""
        await self.sentiment_processor.initialize()
        logger.info("✅ TradeMonkey Fusion Integration fully initialized!")
    
    async def start(self):
        """Start all background processing"""
        if self.is_running:
            logger.warning("Integration already running!")
            return
        
        self.is_running = True
        
        # Start sentiment processing loop
        sentiment_task = asyncio.create_task(self.sentiment_processor.start_processing_loop())
        self.background_tasks.append(sentiment_task)
        
        logger.info("🚀 TradeMonkey Fusion Integration started!")
    
    async def enhance_signal(self, original_signal: Dict, symbol: str = None) -> Dict:
        """Main method to enhance trading signals with sentiment"""
        return await self.signal_enhancer.enhance_trading_signal(original_signal, symbol)
    
    async def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        # Get sentiment processor stats
        sentiment_stats = self.sentiment_processor.stats
        
        # Get market sentiment
        market_sentiment = await self.sentiment_processor.get_current_market_sentiment()
        
        # Get enhancement stats
        enhancement_stats = self.signal_enhancer.get_enhancement_stats()
        
        # Get GPU memory stats
        gpu_stats = self.sentiment_processor.gpu_optimizer.get_memory_stats()
        
        return {
            'system_status': {
                'is_running': self.is_running,
                'background_tasks': len(self.background_tasks),
                'last_updated': datetime.now().isoformat()
            },
            'sentiment_processing': sentiment_stats,
            'current_market_sentiment': market_sentiment,
            'signal_enhancement': enhancement_stats,
            'gpu_performance': gpu_stats,
            'health_indicators': {
                'sentiment_data_fresh': market_sentiment['sample_count'] > 5,
                'gpu_memory_healthy': gpu_stats['total_used_gb'] < 10.0,
                'processing_speed_good': sentiment_stats['avg_processing_time'] < 5.0,
                'overall_health': 'excellent' if (
                    market_sentiment['sample_count'] > 10 and 
                    gpu_stats['total_used_gb'] < 8.0 and
                    sentiment_stats['avg_processing_time'] < 3.0
                ) else 'good'
            }
        }
    
    async def shutdown(self):
        """Graceful shutdown"""
        self.is_running = False
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Close Redis connections
        if self.sentiment_processor.redis_client:
            await self.sentiment_processor.redis_client.close()
        
        logger.info("🛑 TradeMonkey Fusion Integration shutdown complete")

# Example usage and testing
async def main():
    """Complete integration testing"""
    logger.info("🔥 Testing Complete TradeMonkey Fusion Integration!")
    
    # Initialize integration
    config = FusionIntegrationConfig(
        sentiment_weight=0.3,
        confidence_threshold=0.6,
        crypto_boost_multiplier=1.5
    )
    
    integration = TradeMonkeyFusionIntegration(config)
    await integration.initialize()
    
    # Start background processing
    await integration.start()
    
    # Wait a moment for processing to start
    await asyncio.sleep(2)
    
    # Test signal enhancement
    test_signals = [
        {
            'action': 'buy',
            'confidence': 0.7,
            'position_size': 0.25,
            'symbol': 'BTC/USD',
            'predicted_return': 0.03
        },
        {
            'action': 'sell', 
            'confidence': 0.6,
            'position_size': 0.20,
            'symbol': 'ETH/USD',
            'predicted_return': -0.02
        }
    ]
    
    logger.info("🎯 Testing signal enhancement...")
    for i, signal in enumerate(test_signals):
        enhanced = await integration.enhance_signal(signal, signal['symbol'])
        
        logger.info(f"📊 Signal {i+1}:")
        logger.info(f"  Original: {signal}")
        logger.info(f"  Enhanced: {enhanced}")
        logger.info(f"  Sentiment: {enhanced.get('sentiment_analysis', {})}")
        logger.info("-" * 80)
    
    # Get system status
    status = await integration.get_system_status()
    logger.info(f"💻 System Status:")
    logger.info(f"  Health: {status['health_indicators']['overall_health']}")
    logger.info(f"  Market Sentiment: {status['current_market_sentiment']['avg_sentiment']:.3f}")
    logger.info(f"  GPU Memory: {status['gpu_performance']['total_used_gb']:.2f} GB")
    logger.info(f"  Processing Speed: {status['sentiment_processing']['avg_processing_time']:.2f}s")
    
    # Cleanup
    await integration.shutdown()
    
    logger.info("✅