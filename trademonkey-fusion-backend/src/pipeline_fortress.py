# TradeMonkey Fusion - Pipeline Fortress
# Authors: The Legendary Bros
# Purpose: Multi-source data ingestion with bulletproof error handling
# Date: June 5, 2025

import asyncio
import aiohttp
import redis.asyncio as redis
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import json
from circuitbreaker import circuit
from tenacity import retry, stop_after_attempt, wait_exponential

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('PipelineFortress')

class PipelineFortress:
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_client = redis.from_url(redis_url)
        self.session = None
        self.data_sources = {
            'twitter_api': {
                'endpoint': 'https://api.twitter.com/2/tweets/search/stream',
                'rate_limit': 300/900,  # 300 requests per 15 minutes
                'failover': ['twitter_scraper', 'cached_feed']
            },
            'reddit_praw': {
                'subreddits': ['cryptocurrency', 'bitcoin', 'ethtrader', 'wallstreetbets'],
                'rate_limit': 60/60,  # 60 requests per minute
                'failover': ['reddit_api', 'pushshift']
            },
            'discord_webhooks': {
                'channels': ['crypto_calls', 'degen_plays', 'moonshots'],
                'rate_limit': 50/60,  # 50 requests per minute
                'failover': ['telegram_backup', 'cached_messages']
            },
            'telegram_api': {
                'channels': ['crypto_signals', 'pump_groups'],
                'rate_limit': 30/60,  # 30 requests per minute
                'failover': ['telegram_scraper', 'cached_feed']
            },
            'news_api': {
                'endpoint': 'https://newsapi.org/v2/everything',
                'rate_limit': 100/86400,  # 100 requests per day
                'failover': ['cached_news']
            }
        }

    async def initialize(self):
        """Initialize HTTP session and Redis connection."""
        self.session = aiohttp.ClientSession()
        await self.redis_client.ping()
        logger.info("Pipeline Fortress initialized!")

    @circuit(failure_threshold=5, recovery_timeout=60)
    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=60))
    async def fetch_from_source(self, source: str, params: Dict = None) -> List[Dict]:
        """Fetch data from a single source with circuit breaker and retry logic."""
        config = self.data_sources[source]
        endpoint = config.get('endpoint', '')
        
        try:
            async with self.session.get(endpoint, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    texts = self._extract_texts(data, source)
                    await self._store_in_queue(texts)
                    return texts
                elif response.status == 429:
                    raise Exception("Rate limit exceeded")
                else:
                    raise Exception(f"HTTP {response.status}")
        except Exception as e:
            logger.error(f"Fetch failed for {source}: {e}")
            raise

    async def execute_with_fallback(self, source: str, params: Dict = None) -> List[Dict]:
        """Execute fetch with automatic failover to backup sources."""
        sources = [source] + self.data_sources[source]['failover']
        for src in sources:
            try:
                return await self.fetch_from_source(src, params)
            except Exception as e:
                logger.warning(f"Source {src} failed: {e}, trying next...")
        raise Exception("All sources failed - DEFCON 1!")

    async def _store_in_queue(self, texts: List[Dict]):
        """Store fetched texts in Redis queue with timestamp."""
        async with self.redis_client.pipeline() as pipe:
            for text in texts:
                data = {
                    'text': text['content'],
                    'source': text['source'],
                    'timestamp': datetime.now().isoformat()
                }
                await pipe.lpush("trademonkey:sentiment:queue", json.dumps(data))
            await pipe.execute()

    def _extract_texts(self, data: Dict, source: str) -> List[Dict]:
        """Extract relevant text content from source data."""
        # Placeholder logic - to be customized per source
        texts = []
        if source.startswith('twitter'):
            for tweet in data.get('data', []):
                texts.append({'content': tweet['text'], 'source': source})
        elif source.startswith('reddit'):
            for post in data.get('posts', []):
                texts.append({'content': post['title'] + ' ' + post['selftext'], 'source': source})
        return texts

    async def monitor_queue(self):
        """Monitor queue stats and log performance."""
        while True:
            queue_size = await self.redis_client.llen("trademonkey:sentiment:queue")
            logger.info(f"Queue size: {queue_size} | Throughput: {queue_size/60:.2f} texts/sec")
            await asyncio.sleep(60)

async def main():
    fortress = PipelineFortress()
    await fortress.initialize()
    tasks = [
        fortress.execute_with_fallback('twitter_api', params={'query': 'bitcoin'}),
        fortress.monitor_queue()
    ]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())