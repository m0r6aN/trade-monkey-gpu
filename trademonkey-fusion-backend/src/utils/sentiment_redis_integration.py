# File: trademonkey-fusion-backend/src/utils/sentiment_redis_pusher.py
#!/usr/bin/env python3
"""
TradeMonkey Fusion - Sentiment Redis Data Pusher
"Pushing market emotions straight to the trading beast!" 🧠

This module connects sentiment_engine_v2.py to Redis for real-time sentiment streaming
"""

import asyncio
import json
import redis.asyncio as redis
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np
from dataclasses import dataclass, asdict
import tweepy
import praw
import aiohttp
from textblob import TextBlob
import re

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('SentimentRedisPusher')

@dataclass
class SentimentReading:
    text: str
    sentiment: float
    confidence: float
    source: str
    crypto_related: bool
    timestamp: datetime
    processed_by: str = "fusion_engine_v2"

class CryptoSentimentCollector:
    """Collects sentiment from multiple sources and pushes to Redis"""
    
    def __init__(self):
        # Redis connection
        self.redis_client = None
        
        # Data sources
        self.twitter_api = None
        self.reddit_api = None
        
        # Sentiment processing
        self.crypto_keywords = [
            'bitcoin', 'btc', 'ethereum', 'eth', 'crypto', 'cryptocurrency',
            'blockchain', 'defi', 'nft', 'solana', 'sol', 'cardano', 'ada',
            'dogecoin', 'doge', 'shiba', 'matic', 'polygon', 'luna', 'terra',
            'wagmi', 'ngmi', 'hodl', 'diamond hands', 'paper hands', 'ape',
            'moon', 'lambo', 'rekt', 'pump', 'dump', 'fud', 'fomo'
        ]
        
        # Sentiment buffer for processing
        self.sentiment_buffer: List[SentimentReading] = []
        self.max_buffer_size = 100
        
        logger.info("🧠 Crypto Sentiment Collector initialized!")
    
    async def initialize(self):
        """Initialize all connections and APIs"""
        try:
            # Initialize Redis
            self.redis_client = redis.from_url("redis://localhost:6379")
            await self.redis_client.ping()
            logger.info("✅ Redis connected successfully")
            
            # Initialize Twitter API (if credentials available)
            try:
                import os
                bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
                if bearer_token:
                    self.twitter_api = tweepy.Client(bearer_token=bearer_token)
                    logger.info("✅ Twitter API initialized")
                else:
                    logger.warning("⚠️ Twitter credentials not found - skipping Twitter integration")
            except Exception as e:
                logger.warning(f"⚠️ Twitter API initialization failed: {e}")
            
            # Initialize Reddit API (if credentials available)
            try:
                reddit_client_id = os.getenv('REDDIT_CLIENT_ID')
                reddit_secret = os.getenv('REDDIT_CLIENT_SECRET')
                if reddit_client_id and reddit_secret:
                    self.reddit_api = praw.Reddit(
                        client_id=reddit_client_id,
                        client_secret=reddit_secret,
                        user_agent="TradeMonkey:v2.0 (by u/trademonkey)"
                    )
                    logger.info("✅ Reddit API initialized")
                else:
                    logger.warning("⚠️ Reddit credentials not found - skipping Reddit integration")
            except Exception as e:
                logger.warning(f"⚠️ Reddit API initialization failed: {e}")
            
        except Exception as e:
            logger.error(f"❌ Initialization error: {e}")
            raise
    
    def is_crypto_related(self, text: str) -> bool:
        """Check if text is crypto-related"""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.crypto_keywords)
    
    def analyze_sentiment_simple(self, text: str) -> tuple[float, float]:
        """Simple sentiment analysis using TextBlob (fallback method)"""
        try:
            blob = TextBlob(text)
            sentiment = blob.sentiment.polarity  # -1 to 1
            confidence = abs(blob.sentiment.polarity) + 0.3  # Base confidence
            return sentiment, min(confidence, 1.0)
        except Exception as e:
            logger.warning(f"TextBlob sentiment analysis failed: {e}")
            return 0.0, 0.0
    
    async def collect_twitter_sentiment(self) -> List[SentimentReading]:
        """Collect sentiment from Twitter"""
        if not self.twitter_api:
            return []
        
        sentiments = []
        try:
            # Search for recent crypto tweets
            query = "bitcoin OR ethereum OR crypto OR btc OR eth -is:retweet lang:en"
            tweets = self.twitter_api.search_recent_tweets(
                query=query,
                max_results=50,
                tweet_fields=['created_at', 'public_metrics']
            )
            
            if tweets.data:
                for tweet in tweets.data:
                    text = tweet.text
                    sentiment_score, confidence = self.analyze_sentiment_simple(text)
                    
                    reading = SentimentReading(
                        text=text[:200],  # Truncate for storage
                        sentiment=sentiment_score,
                        confidence=confidence,
                        source='twitter',
                        crypto_related=self.is_crypto_related(text),
                        timestamp=datetime.now()
                    )
                    sentiments.append(reading)
            
            logger.info(f"📱 Collected {len(sentiments)} Twitter sentiments")
            
        except Exception as e:
            logger.error(f"Twitter collection error: {e}")
        
        return sentiments
    
    async def collect_reddit_sentiment(self) -> List[SentimentReading]:
        """Collect sentiment from Reddit"""
        if not self.reddit_api:
            return []
        
        sentiments = []
        try:
            # Get hot posts from crypto subreddits
            subreddits = ['cryptocurrency', 'bitcoin', 'ethereum', 'CryptoMarkets']
            
            for subreddit_name in subreddits:
                subreddit = self.reddit_api.subreddit(subreddit_name)
                
                for post in subreddit.hot(limit=25):
                    # Combine title and text
                    text = f"{post.title} {post.selftext}"[:500]
                    if len(text.strip()) < 10:
                        continue
                    
                    sentiment_score, confidence = self.analyze_sentiment_simple(text)
                    
                    reading = SentimentReading(
                        text=text,
                        sentiment=sentiment_score,
                        confidence=confidence,
                        source=f'reddit_{subreddit_name}',
                        crypto_related=True,  # All from crypto subreddits
                        timestamp=datetime.now()
                    )
                    sentiments.append(reading)
            
            logger.info(f"🔴 Collected {len(sentiments)} Reddit sentiments")
            
        except Exception as e:
            logger.error(f"Reddit collection error: {e}")
        
        return sentiments
    
    async def collect_news_sentiment(self) -> List[SentimentReading]:
        """Collect sentiment from crypto news sources"""
        sentiments = []
        try:
            # Mock news sentiment for demo - replace with real news API
            news_headlines = [
                "Bitcoin breaks through resistance, analysts bullish on Q2",
                "Ethereum network upgrade successful, gas fees dropping",
                "Major institution adds Bitcoin to treasury holdings",
                "Regulatory clarity brings optimism to crypto markets",
                "DeFi protocol launches innovative yield farming strategy"
            ]
            
            for headline in news_headlines:
                sentiment_score, confidence = self.analyze_sentiment_simple(headline)
                
                reading = SentimentReading(
                    text=headline,
                    sentiment=sentiment_score,
                    confidence=confidence + 0.2,  # News gets confidence boost
                    source='crypto_news',
                    crypto_related=True,
                    timestamp=datetime.now()
                )
                sentiments.append(reading)
            
            logger.info(f"📰 Collected {len(sentiments)} news sentiments")
            
        except Exception as e:
            logger.error(f"News collection error: {e}")
        
        return sentiments
    
    async def push_to_redis(self, sentiments: List[SentimentReading]):
        """Push sentiment readings to Redis for real-time processing"""
        if not self.redis_client or not sentiments:
            return
        
        try:
            # Add to sentiment queue for processing
            for sentiment in sentiments:
                sentiment_json = json.dumps(asdict(sentiment), default=str)
                await self.redis_client.lpush("trademonkey:sentiment:queue", sentiment_json)
            
            # Calculate aggregated sentiment
            crypto_sentiments = [s for s in sentiments if s.crypto_related]
            if crypto_sentiments:
                avg_sentiment = np.mean([s.sentiment for s in crypto_sentiments])
                avg_confidence = np.mean([s.confidence for s in crypto_sentiments])
                
                # Source breakdown
                sources = {}
                for sentiment in crypto_sentiments:
                    source = sentiment.source.split('_')[0]  # twitter, reddit, etc.
                    sources[source] = sources.get(source, 0) + 1
                
                # Calculate trend (simplified)
                recent_sentiments = [s.sentiment for s in crypto_sentiments[-20:]]
                if len(recent_sentiments) >= 10:
                    early_avg = np.mean(recent_sentiments[:len(recent_sentiments)//2])
                    late_avg = np.mean(recent_sentiments[len(recent_sentiments)//2:])
                    trend = late_avg - early_avg
                else:
                    trend = 0.0
                
                # Store current market sentiment
                current_sentiment = {
                    'avg_sentiment': float(avg_sentiment),
                    'confidence': float(avg_confidence),
                    'sentiment_trend': float(trend),
                    'sample_count': len(crypto_sentiments),
                    'crypto_ratio': len(crypto_sentiments) / len(sentiments) if sentiments else 0,
                    'sources': sources,
                    'last_updated': datetime.now().isoformat()
                }
                
                await self.redis_client.setex(
                    "trademonkey:sentiment:current",
                    300,  # 5 minute TTL
                    json.dumps(current_sentiment)
                )
                
                logger.info(f"📊 Pushed aggregated sentiment: {avg_sentiment:.3f} confidence: {avg_confidence:.3f}")
            
        except Exception as e:
            logger.error(f"Redis push error: {e}")
    
    async def run_collection_cycle(self):
        """Run one complete sentiment collection cycle"""
        logger.info("🔄 Starting sentiment collection cycle...")
        
        # Collect from all sources in parallel
        tasks = [
            self.collect_twitter_sentiment(),
            self.collect_reddit_sentiment(),
            self.collect_news_sentiment()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine all sentiments
        all_sentiments = []
        for result in results:
            if isinstance(result, list):
                all_sentiments.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"Collection task error: {result}")
        
        # Push to Redis
        if all_sentiments:
            await self.push_to_redis(all_sentiments)
            logger.info(f"✅ Collected {len(all_sentiments)} total sentiments")
        else:
            logger.warning("⚠️ No sentiments collected this cycle")
    
    async def start_continuous_collection(self, interval_seconds: int = 30):
        """Start continuous sentiment collection"""
        logger.info(f"🚀 Starting continuous sentiment collection (interval: {interval_seconds}s)")
        
        while True:
            try:
                await self.run_collection_cycle()
                await asyncio.sleep(interval_seconds)
            except KeyboardInterrupt:
                logger.info("👋 Sentiment collection stopped by user")
                break
            except Exception as e:
                logger.error(f"Collection cycle error: {e}")
                await asyncio.sleep(10)  # Short backoff on error

async def main():
    """Main entry point for sentiment collection"""
    collector = CryptoSentimentCollector()
    
    try:
        await collector.initialize()
        await collector.start_continuous_collection(interval_seconds=30)
    except KeyboardInterrupt:
        logger.info("👋 Shutting down sentiment collector...")
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
    finally:
        if collector.redis_client:
            await collector.redis_client.close()

if __name__ == "__main__":
    asyncio.run(main())