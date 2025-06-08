# File: src/utils/sentiment_redis_integration.py
"""
TradeMonkey Fusion - Redis Client Integration
"Connecting the sentiment pipeline to the Redis fortress!" 🚀
"""

import redis.asyncio as redis
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import asyncio

logger = logging.getLogger('TradeMonkeyRedis')

class RedisClient:
    """Enhanced Redis client for TradeMonkey Fusion sentiment and data pipeline"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", max_retries: int = 3):
        self.redis_url = redis_url
        self.max_retries = max_retries
        self.client: Optional[redis.Redis] = None
        self.is_connected = False
        
    async def connect(self):
        """Connect to Redis with retry logic"""
        for attempt in range(self.max_retries):
            try:
                self.client = redis.from_url(self.redis_url)
                await self.client.ping()
                self.is_connected = True
                logger.info(f"✅ Redis connected successfully on attempt {attempt + 1}")
                return
            except Exception as e:
                logger.warning(f"Redis connection attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error("❌ Failed to connect to Redis after all retries")
                    raise
    
    async def ping(self) -> bool:
        """Ping Redis to check connection"""
        try:
            if not self.client:
                await self.connect()
            await self.client.ping()
            return True
        except Exception as e:
            logger.error(f"Redis ping failed: {e}")
            self.is_connected = False
            return False
    
    async def get(self, key: str) -> Optional[str]:
        """Get value from Redis"""
        try:
            if not self.client:
                await self.connect()
            result = await self.client.get(key)
            return result.decode('utf-8') if result else None
        except Exception as e:
            logger.error(f"Redis GET error for key {key}: {e}")
            return None
    
    async def set(self, key: str, value: str, expire_seconds: Optional[int] = None):
        """Set value in Redis with optional expiration"""
        try:
            if not self.client:
                await self.connect()
            await self.client.set(key, value, ex=expire_seconds)
            logger.debug(f"Redis SET: {key}")
        except Exception as e:
            logger.error(f"Redis SET error for key {key}: {e}")
    
    async def publish(self, channel: str, message: str):
        """Publish message to Redis channel"""
        try:
            if not self.client:
                await self.connect()
            await self.client.publish(channel, message)
            logger.debug(f"Published to channel {channel}")
        except Exception as e:
            logger.error(f"Redis PUBLISH error for channel {channel}: {e}")
    
    async def get_sentiment_data(self) -> Optional[Dict]:
        """Get current sentiment data from Redis"""
        try:
            data = await self.get("trademonkey:sentiment:current")
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"Error getting sentiment data: {e}")
            return None
    
    async def set_sentiment_data(self, sentiment_data: Dict):
        """Store sentiment data in Redis"""
        try:
            data_json = json.dumps(sentiment_data)
            await self.set("trademonkey:sentiment:current", data_json, expire_seconds=300)
            
            # Also publish to sentiment channel for real-time updates
            await self.publish("sentiment_updates", data_json)
            
            logger.info(f"📊 Stored sentiment data: {sentiment_data.get('avg_sentiment', 'N/A')}")
        except Exception as e:
            logger.error(f"Error setting sentiment data: {e}")
    
    async def get_gpu_metrics(self) -> Optional[Dict]:
        """Get GPU performance metrics"""
        try:
            data = await self.get("trademonkey:gpu:metrics")
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"Error getting GPU metrics: {e}")
            return None
    
    async def set_gpu_metrics(self, gpu_data: Dict):
        """Store GPU metrics in Redis"""
        try:
            data_json = json.dumps(gpu_data)
            await self.set("trademonkey:gpu:metrics", data_json, expire_seconds=120)
            await self.publish("gpu_updates", data_json)
            logger.debug(f"⚡ Stored GPU metrics: {gpu_data.get('memory_usage_pct', 'N/A')}%")
        except Exception as e:
            logger.error(f"Error setting GPU metrics: {e}")
    
    async def get_recent_events(self, limit: int = 50) -> List[Dict]:
        """Get recent activity events"""
        try:
            events_data = await self.get("trademonkey:events:recent")
            if events_data:
                events = json.loads(events_data)
                return events[-limit:] if len(events) > limit else events
            return []
        except Exception as e:
            logger.error(f"Error getting recent events: {e}")
            return []
    
    async def add_event(self, event_type: str, description: str, metadata: Optional[Dict] = None):
        """Add new activity event"""
        try:
            event = {
                "event_id": f"evt_{int(datetime.now().timestamp())}",
                "type": event_type,
                "timestamp": datetime.now().isoformat(),
                "description": description,
                "metadata": metadata or {}
            }
            
            # Get existing events
            events = await self.get_recent_events(100)
            events.append(event)
            
            # Keep only last 100 events
            if len(events) > 100:
                events = events[-100:]
            
            # Store updated events
            events_json = json.dumps(events)
            await self.set("trademonkey:events:recent", events_json, expire_seconds=3600)
            await self.publish("activity_updates", json.dumps(event))
            
            logger.info(f"📝 Added event: {event_type} - {description}")
        except Exception as e:
            logger.error(f"Error adding event: {e}")
    
    async def get_uptime(self) -> float:
        """Get system uptime percentage"""
        try:
            uptime_data = await self.get("trademonkey:system:uptime")
            if uptime_data:
                data = json.loads(uptime_data)
                return data.get("uptime_percent", 99.9)
            return 99.9  # Default high uptime
        except Exception as e:
            logger.error(f"Error getting uptime: {e}")
            return 99.9
    
    async def update_uptime(self, uptime_percent: float):
        """Update system uptime"""
        try:
            uptime_data = {
                "uptime_percent": uptime_percent,
                "last_updated": datetime.now().isoformat()
            }
            await self.set("trademonkey:system:uptime", json.dumps(uptime_data), expire_seconds=3600)
        except Exception as e:
            logger.error(f"Error updating uptime: {e}")
    
    async def get_training_status(self, training_id: str) -> Optional[Dict]:
        """Get ML training status"""
        try:
            data = await self.get(f"trademonkey:training:{training_id}")
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"Error getting training status: {e}")
            return None
    
    async def set_training_status(self, training_id: str, status_data: Dict):
        """Set ML training status"""
        try:
            await self.set(f"trademonkey:training:{training_id}", json.dumps(status_data), expire_seconds=86400)
            await self.publish("training_updates", json.dumps({"training_id": training_id, **status_data}))
        except Exception as e:
            logger.error(f"Error setting training status: {e}")
    
    async def log_error(self, error_type: str, error_message: str, context: Optional[Dict] = None):
        """Log error to Redis for monitoring"""
        try:
            error_data = {
                "error_id": f"err_{int(datetime.now().timestamp())}",
                "type": error_type,
                "message": error_message,
                "context": context or {},
                "timestamp": datetime.now().isoformat()
            }
            
            # Get existing errors
            errors_data = await self.get("trademonkey:errors:recent")
            errors = json.loads(errors_data) if errors_data else []
            errors.append(error_data)
            
            # Keep only last 50 errors
            if len(errors) > 50:
                errors = errors[-50:]
            
            await self.set("trademonkey:errors:recent", json.dumps(errors), expire_seconds=86400)
            await self.publish("error_alerts", json.dumps(error_data))
            
            logger.error(f"🚨 Logged error: {error_type} - {error_message}")
        except Exception as e:
            logger.error(f"Error logging error (meta!): {e}")
    
    async def get_portfolio_cache(self, portfolio_id: str) -> Optional[Dict]:
        """Get cached portfolio data"""
        try:
            data = await self.get(f"trademonkey:portfolio:{portfolio_id}")
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"Error getting portfolio cache: {e}")
            return None
    
    async def set_portfolio_cache(self, portfolio_id: str, portfolio_data: Dict):
        """Cache portfolio data"""
        try:
            await self.set(f"trademonkey:portfolio:{portfolio_id}", json.dumps(portfolio_data), expire_seconds=300)
        except Exception as e:
            logger.error(f"Error setting portfolio cache: {e}")
    
    async def get_signal_cache(self, symbol: str) -> Optional[Dict]:
        """Get cached signal data for symbol"""
        try:
            data = await self.get(f"trademonkey:signals:{symbol}")
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"Error getting signal cache: {e}")
            return None
    
    async def set_signal_cache(self, symbol: str, signal_data: Dict):
        """Cache signal data for symbol"""
        try:
            await self.set(f"trademonkey:signals:{symbol}", json.dumps(signal_data), expire_seconds=60)
            await self.publish("signal_updates", json.dumps({"symbol": symbol, **signal_data}))
        except Exception as e:
            logger.error(f"Error setting signal cache: {e}")
    
    async def cleanup_expired_data(self):
        """Clean up expired data and optimize Redis usage"""
        try:
            if not self.client:
                return
            
            # Get memory info
            info = await self.client.info('memory')
            used_memory_mb = info.get('used_memory', 0) / 1024 / 1024
            
            logger.info(f"🧹 Redis cleanup - Memory usage: {used_memory_mb:.1f}MB")
            
            # Clean up old keys if memory usage is high
            if used_memory_mb > 100:  # If using more than 100MB
                patterns_to_clean = [
                    "trademonkey:signals:*",
                    "trademonkey:portfolio:*",
                    "trademonkey:training:*"
                ]
                
                for pattern in patterns_to_clean:
                    keys = await self.client.keys(pattern)
                    if keys:
                        # Delete oldest keys first (simple strategy)
                        keys_to_delete = keys[:len(keys)//2]  # Delete half
                        if keys_to_delete:
                            await self.client.delete(*keys_to_delete)
                            logger.info(f"🗑️ Cleaned up {len(keys_to_delete)} keys for pattern {pattern}")
                            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    async def close(self):
        """Close Redis connection"""
        try:
            if self.client:
                await self.client.close()
                self.is_connected = False
                logger.info("🔌 Redis connection closed")
        except Exception as e:
            logger.error(f"Error closing Redis connection: {e}")

# Singleton instance for global use
redis_client = RedisClient()

# Convenience functions for backward compatibility
async def get_sentiment_data() -> Optional[Dict]:
    """Get current sentiment data"""
    return await redis_client.get_sentiment_data()

async def set_sentiment_data(data: Dict):
    """Set sentiment data"""
    await redis_client.set_sentiment_data(data)

async def publish_sentiment_update(data: Dict):
    """Publish sentiment update"""
    await redis_client.publish("sentiment_updates", json.dumps(data))

async def get_gpu_metrics() -> Optional[Dict]:
    """Get GPU metrics"""
    return await redis_client.get_gpu_metrics()

async def set_gpu_metrics(data: Dict):
    """Set GPU metrics"""
    await redis_client.set_gpu_metrics(data)

async def add_activity_event(event_type: str, description: str, metadata: Optional[Dict] = None):
    """Add activity event"""
    await redis_client.add_event(event_type, description, metadata)

async def log_trading_error(error_type: str, message: str, context: Optional[Dict] = None):
    """Log trading error"""
    await redis_client.log_error(error_type, message, context)