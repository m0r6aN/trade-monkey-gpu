#!/usr/bin/env python3
"""
TradeMonkey Fusion - Enhanced Sentiment Analysis Engine
"When AI meets market psychology, legends are born" 🧠💎

This module gives TradeMonkey psychic powers to read market sentiment
from news, tweets, and social media before the market even knows what's coming!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer,
    BertForSequenceClassification,
    BertTokenizer
)
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import pickle
from pathlib import Path

# Setup legendary logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('TradeMonkeySentiment')

@dataclass
class SentimentConfig:
    """Configuration for the sentiment analysis system"""
    # Model settings
    model_name: str = "ProsusAI/finbert"  # Financial BERT for market text
    backup_model: str = "nlptown/bert-base-multilingual-uncased-sentiment"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_length: int = 512
    batch_size: int = 16
    
    # Sentiment scoring
    sentiment_weight: float = 0.3  # How much sentiment affects overall confidence
    decay_hours: float = 2.0  # How long sentiment influence lasts
    min_confidence_threshold: float = 0.1  # Minimum confidence to consider sentiment
    
    # Cache settings
    cache_size: int = 1000
    cache_ttl_minutes: int = 30
    
    # Advanced features
    use_ensemble: bool = True  # Use multiple models for better accuracy
    crypto_keywords: List[str] = None
    
    def __post_init__(self):
        if self.crypto_keywords is None:
            self.crypto_keywords = [
                'bitcoin', 'btc', 'ethereum', 'eth', 'crypto', 'cryptocurrency',
                'blockchain', 'defi', 'nft', 'altcoin', 'pump', 'dump', 'hodl',
                'moon', 'lambo', 'diamond hands', 'paper hands', 'fud', 'fomo'
            ]

class CryptoSentimentClassifier(nn.Module):
    """Custom sentiment classifier fine-tuned for crypto language"""
    
    def __init__(self, base_model_name: str, num_classes: int = 3):
        super().__init__()
        self.bert = BertForSequenceClassification.from_pretrained(
            base_model_name, 
            num_labels=num_classes
        )
        
        # Add crypto-specific layers
        hidden_size = self.bert.config.hidden_size
        self.crypto_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        self.sentiment_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        # Apply crypto-specific attention
        attended_output, _ = self.crypto_attention(
            sequence_output, sequence_output, sequence_output,
            key_padding_mask=~attention_mask.bool() if attention_mask is not None else None
        )
        
        # Pool and classify
        pooled_output = torch.mean(attended_output, dim=1)
        logits = self.sentiment_head(pooled_output)
        
        return {'logits': logits, 'hidden_states': attended_output}

class SentimentCache:
    """Fast caching system for sentiment scores"""
    
    def __init__(self, max_size: int = 1000, ttl_minutes: int = 30):
        self.cache = {}
        self.timestamps = {}
        self.max_size = max_size
        self.ttl = timedelta(minutes=ttl_minutes)
        
    def get(self, text_hash: str) -> Optional[float]:
        if text_hash in self.cache:
            if datetime.now() - self.timestamps[text_hash] < self.ttl:
                return self.cache[text_hash]
            else:
                # Expired
                del self.cache[text_hash]
                del self.timestamps[text_hash]
        return None
    
    def set(self, text_hash: str, sentiment: float):
        # Evict oldest if cache is full
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.timestamps.keys(), key=lambda k: self.timestamps[k])
            del self.cache[oldest_key]
            del self.timestamps[oldest_key]
        
        self.cache[text_hash] = sentiment
        self.timestamps[text_hash] = datetime.now()
    
    def clear(self):
        self.cache.clear()
        self.timestamps.clear()

class EnhancedSentimentAnalyzer:
    """The legendary sentiment engine that gives TradeMonkey psychic powers! 🔮"""
    
    def __init__(self, config: SentimentConfig = None):
        self.config = config or SentimentConfig()
        self.device = self.config.device
        self.cache = SentimentCache(self.config.cache_size, self.config.cache_ttl_minutes)
        
        # Initialize models
        self._load_models()
        
        # Sentiment history for temporal analysis
        self.sentiment_history = []
        self.max_history = 1000
        
        logger.info(f"🧠 Enhanced Sentiment Analyzer initialized on {self.device}")
        logger.info(f"🎯 Using ensemble: {self.config.use_ensemble}")
    
    def _load_models(self):
        """Load and initialize sentiment models"""
        try:
            # Primary FinBERT model for financial text
            logger.info(f"🔄 Loading primary model: {self.config.model_name}")
            self.primary_tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self.primary_model = AutoModelForSequenceClassification.from_pretrained(
                self.config.model_name
            ).to(self.device)
            
            if self.config.use_ensemble:
                # Secondary model for broader sentiment
                logger.info(f"🔄 Loading secondary model: {self.config.backup_model}")
                self.secondary_tokenizer = AutoTokenizer.from_pretrained(self.config.backup_model)
                self.secondary_model = AutoModelForSequenceClassification.from_pretrained(
                    self.config.backup_model
                ).to(self.device)
            
            logger.info("✅ All sentiment models loaded successfully!")
            
        except Exception as e:
            logger.error(f"❌ Failed to load models: {e}")
            raise
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for better sentiment analysis"""
        # Remove URLs
        import re
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'www\.\S+', '', text)
        
        # Handle crypto slang
        crypto_replacements = {
            '🚀': ' bullish rocket ',
            '💎': ' diamond hands hold ',
            '📈': ' price up ',
            '📉': ' price down ',
            '🔥': ' fire hot ',
            'hodl': 'hold',
            'rekt': 'destroyed',
            'wen moon': 'when moon price increase',
            'diamond hands': 'strong holder',
            'paper hands': 'weak seller',
            'to the moon': 'price increase dramatically'
        }
        
        for slang, replacement in crypto_replacements.items():
            text = text.replace(slang, replacement)
        
        # Clean whitespace
        text = ' '.join(text.split())
        
        return text
    
    def _calculate_text_hash(self, text: str) -> str:
        """Calculate hash for caching"""
        import hashlib
        return hashlib.md5(text.encode()).hexdigest()
    
    def _analyze_with_model(self, text: str, model, tokenizer) -> float:
        """Analyze sentiment with a specific model"""
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=self.config.max_length, 
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            
            # Handle different model output formats
            if probs.shape[1] == 3:  # [negative, neutral, positive] or [negative, positive, neutral]
                if 'finbert' in str(model.__class__).lower():
                    # FinBERT: [negative, positive, neutral]
                    sentiment_score = probs[0, 1].item() - probs[0, 0].item()
                else:
                    # Other models: [negative, neutral, positive]
                    sentiment_score = probs[0, 2].item() - probs[0, 0].item()
            elif probs.shape[1] == 5:  # 5-star rating
                # Convert 5-star to sentiment (-1 to 1)
                weighted_score = torch.sum(probs[0] * torch.arange(1, 6, device=self.device)).item()
                sentiment_score = (weighted_score - 3) / 2  # Normalize to [-1, 1]
            else:
                # Binary classification
                sentiment_score = probs[0, 1].item() - probs[0, 0].item()
        
        return sentiment_score
    
    def analyze_single(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of a single text"""
        # Preprocess
        processed_text = self._preprocess_text(text)
        
        # Check cache
        text_hash = self._calculate_text_hash(processed_text)
        cached_sentiment = self.cache.get(text_hash)
        if cached_sentiment is not None:
            return {
                'sentiment': cached_sentiment,
                'confidence': 0.8,  # Cached values get moderate confidence
                'source': 'cache'
            }
        
        try:
            # Primary model analysis
            primary_sentiment = self._analyze_with_model(
                processed_text, 
                self.primary_model, 
                self.primary_tokenizer
            )
            
            if self.config.use_ensemble and hasattr(self, 'secondary_model'):
                # Secondary model analysis
                secondary_sentiment = self._analyze_with_model(
                    processed_text, 
                    self.secondary_model, 
                    self.secondary_tokenizer
                )
                
                # Ensemble combination (weighted average)
                final_sentiment = 0.7 * primary_sentiment + 0.3 * secondary_sentiment
                confidence = 1.0 - abs(primary_sentiment - secondary_sentiment) / 2.0
            else:
                final_sentiment = primary_sentiment
                confidence = 0.85  # Single model confidence
            
            # Apply crypto keyword boost
            if any(keyword in processed_text.lower() for keyword in self.config.crypto_keywords):
                confidence *= 1.1  # Boost confidence for crypto-relevant text
            
            # Clamp values
            final_sentiment = np.clip(final_sentiment, -1.0, 1.0)
            confidence = np.clip(confidence, 0.0, 1.0)
            
            # Cache result
            self.cache.set(text_hash, final_sentiment)
            
            # Store in history
            self._update_history(final_sentiment, confidence)
            
            return {
                'sentiment': final_sentiment,
                'confidence': confidence,
                'source': 'ensemble' if self.config.use_ensemble else 'primary'
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {
                'sentiment': 0.0,
                'confidence': 0.0,
                'source': 'error'
            }
    
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """Analyze sentiment for multiple texts efficiently"""
        results = []
        
        # Process in batches for memory efficiency
        for i in range(0, len(texts), self.config.batch_size):
            batch_texts = texts[i:i + self.config.batch_size]
            
            # For now, process individually (can optimize with true batching later)
            batch_results = [self.analyze_single(text) for text in batch_texts]
            results.extend(batch_results)
        
        return results
    
    def _update_history(self, sentiment: float, confidence: float):
        """Update sentiment history for temporal analysis"""
        entry = {
            'sentiment': sentiment,
            'confidence': confidence,
            'timestamp': datetime.now()
        }
        
        self.sentiment_history.append(entry)
        
        # Keep only recent history
        if len(self.sentiment_history) > self.max_history:
            self.sentiment_history = self.sentiment_history[-self.max_history:]
    
    def get_temporal_sentiment(self, hours_back: float = 1.0) -> Dict[str, float]:
        """Get aggregated sentiment over a time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        recent_sentiments = [
            entry for entry in self.sentiment_history 
            if entry['timestamp'] > cutoff_time
        ]
        
        if not recent_sentiments:
            return {
                'avg_sentiment': 0.0,
                'sentiment_trend': 0.0,
                'confidence': 0.0,
                'sample_count': 0
            }
        
        # Calculate metrics
        sentiments = [entry['sentiment'] for entry in recent_sentiments]
        confidences = [entry['confidence'] for entry in recent_sentiments]
        
        avg_sentiment = np.mean(sentiments)
        avg_confidence = np.mean(confidences)
        
        # Calculate trend (recent vs older)
        if len(sentiments) >= 4:
            recent_half = sentiments[len(sentiments)//2:]
            older_half = sentiments[:len(sentiments)//2]
            sentiment_trend = np.mean(recent_half) - np.mean(older_half)
        else:
            sentiment_trend = 0.0
        
        return {
            'avg_sentiment': float(avg_sentiment),
            'sentiment_trend': float(sentiment_trend),
            'confidence': float(avg_confidence),
            'sample_count': len(recent_sentiments)
        }
    
    def get_market_sentiment_signal(self, current_confidence: float = 1.0) -> Dict[str, float]:
        """Generate trading signal adjustment based on sentiment"""
        # Get recent sentiment trends
        temporal_sentiment = self.get_temporal_sentiment(self.config.decay_hours)
        
        if temporal_sentiment['sample_count'] < 3:
            # Not enough data
            return {
                'sentiment_adjustment': 0.0,
                'confidence_multiplier': 1.0,
                'signal_strength': 0.0
            }
        
        # Calculate sentiment strength
        sentiment_magnitude = abs(temporal_sentiment['avg_sentiment'])
        sentiment_confidence = temporal_sentiment['confidence']
        
        # Only apply sentiment if we have minimum confidence
        if sentiment_confidence < self.config.min_confidence_threshold:
            return {
                'sentiment_adjustment': 0.0,
                'confidence_multiplier': 1.0,
                'signal_strength': 0.0
            }
        
        # Calculate adjustments
        sentiment_adjustment = temporal_sentiment['avg_sentiment'] * self.config.sentiment_weight
        
        # Boost confidence if sentiment aligns with trend
        trend_boost = 1.0 + (sentiment_magnitude * 0.2)  # Up to 20% boost
        confidence_multiplier = trend_boost if sentiment_confidence > 0.7 else 1.0
        
        # Overall signal strength from sentiment
        signal_strength = sentiment_magnitude * sentiment_confidence
        
        return {
            'sentiment_adjustment': float(sentiment_adjustment),
            'confidence_multiplier': float(confidence_multiplier),
            'signal_strength': float(signal_strength),
            'raw_sentiment': temporal_sentiment['avg_sentiment'],
            'sentiment_trend': temporal_sentiment['sentiment_trend']
        }
    
    def save_state(self, filepath: str):
        """Save analyzer state"""
        state = {
            'config': self.config,
            'sentiment_history': self.sentiment_history[-100:],  # Save recent history
            'cache': dict(self.cache.cache),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"💾 Sentiment analyzer state saved to {filepath}")
    
    def load_state(self, filepath: str):
        """Load analyzer state"""
        if not Path(filepath).exists():
            logger.warning(f"State file {filepath} not found")
            return
        
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            
            if 'sentiment_history' in state:
                self.sentiment_history = state['sentiment_history']
            
            logger.info(f"📂 Sentiment analyzer state loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load state: {e}")

# Integration with TradeMonkey Fusion
class TradeMonkeySentimentIntegration:
    """Integration class to connect sentiment analysis with TradeMonkey Fusion"""
    
    def __init__(self, sentiment_analyzer: EnhancedSentimentAnalyzer):
        self.sentiment_analyzer = sentiment_analyzer
        logger.info("🔌 TradeMonkey sentiment integration initialized")
    
    async def process_live_sentiment(self, texts: List[str]) -> Dict[str, float]:
        """Process live sentiment data and return trading adjustments"""
        if not texts:
            return {'sentiment_adjustment': 0.0, 'confidence_multiplier': 1.0}
        
        # Analyze all texts
        sentiment_results = self.sentiment_analyzer.analyze_batch(texts)
        
        # Get current market sentiment signal
        market_signal = self.sentiment_analyzer.get_market_sentiment_signal()
        
        logger.info(f"📊 Processed {len(texts)} sentiment texts")
        logger.info(f"🎯 Market sentiment: {market_signal['raw_sentiment']:.3f}")
        logger.info(f"⚡ Signal strength: {market_signal['signal_strength']:.3f}")
        
        return market_signal
    
    def enhance_trading_signal(self, original_signal: Dict, sentiment_data: List[str] = None) -> Dict:
        """Enhance trading signal with sentiment analysis"""
        if sentiment_data:
            # Process new sentiment data
            asyncio.create_task(self.process_live_sentiment(sentiment_data))
        
        # Get current sentiment adjustments
        sentiment_signal = self.sentiment_analyzer.get_market_sentiment_signal(
            original_signal.get('confidence', 1.0)
        )
        
        # Apply sentiment adjustments
        enhanced_signal = original_signal.copy()
        
        # Adjust confidence based on sentiment
        enhanced_signal['confidence'] *= sentiment_signal['confidence_multiplier']
        
        # Add sentiment to signal strength
        enhanced_signal['signal_strength'] = enhanced_signal.get('signal_strength', 0) + sentiment_signal['signal_strength']
        
        # Add sentiment metadata
        enhanced_signal['sentiment'] = {
            'raw_sentiment': sentiment_signal['raw_sentiment'],
            'sentiment_trend': sentiment_signal['sentiment_trend'],
            'adjustment': sentiment_signal['sentiment_adjustment']
        }
        
        logger.info(f"🚀 Enhanced signal with sentiment: {sentiment_signal['raw_sentiment']:.3f}")
        
        return enhanced_signal

# Example usage and testing
async def main():
    """Test the sentiment analysis engine"""
    logger.info("🔥 Testing TradeMonkey Sentiment Analysis Engine!")
    
    # Initialize analyzer
    config = SentimentConfig(use_ensemble=True)
    analyzer = EnhancedSentimentAnalyzer(config)
    
    # Test texts
    test_texts = [
        "Bitcoin to the moon! 🚀🚀🚀 This bull run is incredible!",
        "Market crash incoming, everything is dumping hard 📉",
        "Hodling strong with these diamond hands 💎🙌",
        "Another day, another pump and dump scheme",
        "Ethereum breaking all resistance levels! Bullish AF!",
        "FUD everywhere but I'm buying the dip",
        "When lambo? When moon? 🌙",
        "Paper hands selling at the bottom again 🤦‍♂️"
    ]
    
    logger.info("📊 Analyzing test sentiments...")
    
    # Analyze sentiments
    for text in test_texts:
        result = analyzer.analyze_single(text)
        logger.info(f"Text: '{text[:50]}...'")
        logger.info(f"Sentiment: {result['sentiment']:.3f} | Confidence: {result['confidence']:.3f}")
        logger.info("-" * 60)
    
    # Test temporal analysis
    await asyncio.sleep(1)  # Small delay to build history
    temporal = analyzer.get_temporal_sentiment(1.0)
    logger.info(f"🕐 Temporal sentiment (1h): {temporal}")
    
    # Test market signal
    market_signal = analyzer.get_market_sentiment_signal()
    logger.info(f"📈 Market signal: {market_signal}")
    
    logger.info("✅ Sentiment analysis testing complete!")

if __name__ == "__main__":
    asyncio.run(main())
