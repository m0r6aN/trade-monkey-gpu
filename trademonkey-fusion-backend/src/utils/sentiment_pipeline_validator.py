# File: trademonkey-fusion-backend/src/utils/sentiment_pipeline_validator.py
#!/usr/bin/env python3
"""
TradeMonkey Fusion - Sentiment Pipeline Validator
"Making sure our market telepathy is CRYSTAL CLEAR!" 🔮

Validates sentiment data flow from collection → Redis → API → UI
"""

import asyncio
import json
import redis.asyncio as redis
import logging
from datetime import datetime, timedelta
import aiohttp
from typing import Dict, List, Optional
import numpy as np
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('SentimentValidator')

class SentimentPipelineValidator:
    """Validates the complete sentiment pipeline end-to-end"""
    
    def __init__(self):
        self.redis_client = None
        self.api_base_url = "http://localhost:8080/api"
        
        # Validation metrics
        self.metrics = {
            'redis_sentiment_count': 0,
            'api_response_time': 0.0,
            'websocket_latency': 0.0,
            'sentiment_freshness': 0.0,
            'signal_enhancement_active': False,
            'pipeline_health_score': 0
        }
        
        logger.info("🔮 Sentiment Pipeline Validator initialized!")
    
    async def initialize(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.from_url("redis://localhost:6379")
            await self.redis_client.ping()
            logger.info("✅ Redis connection validated")
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            raise
    
    async def validate_redis_sentiment_queue(self) -> Dict:
        """Validate sentiment data in Redis queue"""
        try:
            # Check queue length
            queue_length = await self.redis_client.llen("trademonkey:sentiment:queue")
            
            # Check current sentiment data
            current_sentiment = await self.redis_client.get("trademonkey:sentiment:current")
            
            if current_sentiment:
                sentiment_data = json.loads(current_sentiment)
                freshness = datetime.now() - datetime.fromisoformat(sentiment_data.get('last_updated', datetime.now().isoformat()))
                freshness_minutes = freshness.total_seconds() / 60
                
                self.metrics['redis_sentiment_count'] = queue_length
                self.metrics['sentiment_freshness'] = freshness_minutes
                
                logger.info(f"📊 Redis Queue: {queue_length} items, Freshness: {freshness_minutes:.1f}m")
                
                return {
                    'status': 'healthy' if freshness_minutes < 5 else 'stale',
                    'queue_length': queue_length,
                    'freshness_minutes': freshness_minutes,
                    'sentiment_data': sentiment_data
                }
            else:
                logger.warning("⚠️ No current sentiment data in Redis")
                return {'status': 'missing', 'queue_length': queue_length}
                
        except Exception as e:
            logger.error(f"❌ Redis validation error: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def validate_api_endpoints(self) -> Dict:
        """Validate API endpoints are serving sentiment data"""
        try:
            start_time = time.time()
            
            async with aiohttp.ClientSession() as session:
                # Test sentiment endpoint
                async with session.get(f"{self.api_base_url}/sentiment/current") as response:
                    api_response_time = time.time() - start_time
                    self.metrics['api_response_time'] = api_response_time
                    
                    if response.status == 200:
                        sentiment_data = await response.json()
                        
                        # Validate sentiment data structure
                        required_fields = ['sentiment', 'confidence', 'sample_count']
                        missing_fields = [field for field in required_fields if field not in sentiment_data]
                        
                        if not missing_fields:
                            logger.info(f"✅ API endpoint healthy - Response time: {api_response_time:.3f}s")
                            
                            # Check if sentiment enhancement should be active
                            sentiment_val = sentiment_data.get('sentiment', 0)
                            confidence_val = sentiment_data.get('confidence', 0)
                            enhancement_active = abs(sentiment_val) > 0.5 and confidence_val > 0.7
                            self.metrics['signal_enhancement_active'] = enhancement_active
                            
                            return {
                                'status': 'healthy',
                                'response_time': api_response_time,
                                'sentiment': sentiment_val,
                                'confidence': confidence_val,
                                'enhancement_active': enhancement_active,
                                'data': sentiment_data
                            }
                        else:
                            logger.error(f"❌ API data missing fields: {missing_fields}")
                            return {'status': 'invalid_data', 'missing_fields': missing_fields}
                    else:
                        logger.error(f"❌ API endpoint error: {response.status}")
                        return {'status': 'api_error', 'status_code': response.status}
                        
        except Exception as e:
            logger.error(f"❌ API validation error: {e}")
            return {'status': 'connection_error', 'error': str(e)}
    
    async def validate_websocket_stream(self) -> Dict:
        """Validate WebSocket is streaming sentiment updates"""
        try:
            import websockets
            
            start_time = time.time()
            ws_uri = "ws://localhost:8080/ws"
            
            async with websockets.connect(ws_uri) as websocket:
                # Send subscription message
                await websocket.send(json.dumps({
                    'type': 'subscribe',
                    'channels': ['sentiment']
                }))
                
                # Wait for sentiment update
                received_update = False
                timeout = 10  # 10 second timeout
                
                while time.time() - start_time < timeout:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=2)
                        data = json.loads(message)
                        
                        if data.get('type') == 'sentiment_update':
                            websocket_latency = time.time() - start_time
                            self.metrics['websocket_latency'] = websocket_latency
                            received_update = True
                            
                            logger.info(f"✅ WebSocket streaming - Latency: {websocket_latency:.3f}s")
                            return {
                                'status': 'streaming',
                                'latency': websocket_latency,
                                'update_received': True
                            }
                    except asyncio.TimeoutError:
                        continue
                
                if not received_update:
                    logger.warning("⚠️ WebSocket connected but no sentiment updates received")
                    return {'status': 'no_updates', 'connected': True}
                    
        except Exception as e:
            logger.error(f"❌ WebSocket validation error: {e}")
            return {'status': 'connection_failed', 'error': str(e)}
    
    async def test_signal_enhancement(self, test_sentiment: float = 0.75) -> Dict:
        """Test signal enhancement logic with mock data"""
        try:
            # Push test sentiment to Redis
            test_data = {
                'avg_sentiment': test_sentiment,
                'confidence': 0.85,
                'sentiment_trend': 0.2,
                'sample_count': 50,
                'crypto_ratio': 0.8,
                'sources': {'twitter': 30, 'reddit': 15, 'news': 5},
                'last_updated': datetime.now().isoformat()
            }
            
            await self.redis_client.setex(
                "trademonkey:sentiment:test",
                60,  # 1 minute TTL
                json.dumps(test_data)
            )
            
            # Calculate expected enhancement
            expected_boost = test_sentiment * 0.85 * 0.5  # sentiment * confidence * multiplier
            
            logger.info(f"🧪 Test signal enhancement: {test_sentiment} sentiment → {expected_boost:.3f} boost")
            
            return {
                'status': 'tested',
                'test_sentiment': test_sentiment,
                'expected_boost': expected_boost,
                'enhancement_threshold': 0.5,
                'should_enhance': abs(test_sentiment) > 0.5
            }
            
        except Exception as e:
            logger.error(f"❌ Signal enhancement test error: {e}")
            return {'status': 'test_failed', 'error': str(e)}
    
    async def calculate_pipeline_health_score(self) -> int:
        """Calculate overall pipeline health score (0-100)"""
        score = 0
        
        # Redis health (25 points)
        if self.metrics['redis_sentiment_count'] > 0:
            score += 15
        if self.metrics['sentiment_freshness'] < 5:  # Less than 5 minutes old
            score += 10
        
        # API health (25 points)
        if self.metrics['api_response_time'] > 0 and self.metrics['api_response_time'] < 1.0:
            score += 25
        
        # WebSocket health (25 points)
        if self.metrics['websocket_latency'] > 0 and self.metrics['websocket_latency'] < 2.0:
            score += 25
        
        # Enhancement logic (25 points)
        if self.metrics['signal_enhancement_active']:
            score += 25
        
        self.metrics['pipeline_health_score'] = score
        return score
    
    async def run_full_validation(self) -> Dict:
        """Run complete pipeline validation"""
        logger.info("🚀 Starting full sentiment pipeline validation...")
        
        validation_results = {
            'timestamp': datetime.now().isoformat(),
            'redis_validation': {},
            'api_validation': {},
            'websocket_validation': {},
            'enhancement_test': {},
            'health_score': 0,
            'overall_status': 'unknown'
        }
        
        try:
            # Validate Redis
            logger.info("📊 Validating Redis sentiment queue...")
            validation_results['redis_validation'] = await self.validate_redis_sentiment_queue()
            
            # Validate API
            logger.info("🌐 Validating API endpoints...")
            validation_results['api_validation'] = await self.validate_api_endpoints()
            
            # Validate WebSocket
            logger.info("🔌 Validating WebSocket stream...")
            validation_results['websocket_validation'] = await self.validate_websocket_stream()
            
            # Test signal enhancement
            logger.info("🧪 Testing signal enhancement logic...")
            validation_results['enhancement_test'] = await self.test_signal_enhancement()
            
            # Calculate health score
            health_score = await self.calculate_pipeline_health_score()
            validation_results['health_score'] = health_score
            
            # Determine overall status
            if health_score >= 90:
                validation_results['overall_status'] = 'excellent'
            elif health_score >= 70:
                validation_results['overall_status'] = 'good'
            elif health_score >= 50:
                validation_results['overall_status'] = 'fair'
            else:
                validation_results['overall_status'] = 'poor'
            
            logger.info(f"🎯 Pipeline validation complete! Health score: {health_score}/100 ({validation_results['overall_status']})")
            
        except Exception as e:
            logger.error(f"❌ Validation error: {e}")
            validation_results['overall_status'] = 'error'
            validation_results['error'] = str(e)
        
        return validation_results
    
    async def monitor_pipeline_continuous(self, interval_seconds: int = 30):
        """Continuously monitor pipeline health"""
        logger.info(f"🔄 Starting continuous pipeline monitoring (interval: {interval_seconds}s)")
        
        while True:
            try:
                results = await self.run_full_validation()
                
                # Log key metrics
                health_score = results['health_score']
                redis_status = results['redis_validation'].get('status', 'unknown')
                api_status = results['api_validation'].get('status', 'unknown')
                ws_status = results['websocket_validation'].get('status', 'unknown')
                
                logger.info(f"💓 Health: {health_score}/100 | Redis: {redis_status} | API: {api_status} | WS: {ws_status}")
                
                # Alert on critical issues
                if health_score < 50:
                    logger.error(f"🚨 CRITICAL: Pipeline health dropped to {health_score}/100!")
                
                await asyncio.sleep(interval_seconds)
                
            except KeyboardInterrupt:
                logger.info("👋 Pipeline monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"❌ Monitoring error: {e}")
                await asyncio.sleep(10)

async def main():
    """Main entry point for validation"""
    validator = SentimentPipelineValidator()
    
    try:
        await validator.initialize()
        
        # Run single validation
        results = await validator.run_full_validation()
        
        # Pretty print results
        print("\n" + "="*60)
        print("🔮 TRADEMONKEY FUSION - SENTIMENT PIPELINE VALIDATION")
        print("="*60)
        print(f"Overall Status: {results['overall_status'].upper()}")
        print(f"Health Score: {results['health_score']}/100")
        print(f"Timestamp: {results['timestamp']}")
        print("\nDetailed Results:")
        print(json.dumps(results, indent=2, default=str))
        
        # Offer continuous monitoring
        response = input("\nStart continuous monitoring? (y/n): ")
        if response.lower() == 'y':
            await validator.monitor_pipeline_continuous()
            
    except KeyboardInterrupt:
        logger.info("👋 Validation stopped by user")
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
    finally:
        if validator.redis_client:
            await validator.redis_client.close()

if __name__ == "__main__":
    asyncio.run(main())