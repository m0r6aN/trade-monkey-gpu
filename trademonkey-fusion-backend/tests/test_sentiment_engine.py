import pytest
import pytest_asyncio
from unittest.mock import patch, Mock
from src.utils.sentiment_engine import EnhancedSentimentAnalyzer, SentimentConfig, SentimentCache, TradeMonkeySentimentIntegration
import torch
from datetime import datetime, timedelta
import numpy as np
import asyncio

@pytest.fixture
def sentiment_config():
    return SentimentConfig(
        model_name="ProsusAI/finbert",
        backup_model="nlptown/bert-base-multilingual-uncased-sentiment",
        use_ensemble=False,  # Disable ensemble for simpler testing
        device="cpu",
        cache_size=10,
        cache_ttl_minutes=5
    )

@pytest.fixture
def mock_tokenizer():
    tokenizer = Mock()
    tokenizer.return_value = {
        "input_ids": torch.tensor([[101, 102]]),
        "attention_mask": torch.tensor([[1, 1]])
    }
    return tokenizer

@pytest.fixture
def mock_model():
    model = Mock()
    model.return_value.logits = torch.tensor([[0.1, 0.8, 0.1]])  # [negative, positive, neutral]
    model.bert.return_value.last_hidden_state = torch.zeros(1, 2, 768)
    return model

@pytest_asyncio.fixture
async def sentiment_analyzer(sentiment_config, mock_tokenizer, mock_model):
    with patch("src.utils.sentiment_engine.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
         patch("src.utils.sentiment_engine.AutoModelForSequenceClassification.from_pretrained", return_value=mock_model):
        analyzer = EnhancedSentimentAnalyzer(sentiment_config)
        return analyzer

@pytest_asyncio.fixture
async def sentiment_integration(sentiment_analyzer):
    return TradeMonkeySentimentIntegration(sentiment_analyzer)

def test_sentiment_cache():
    cache = SentimentCache(max_size=2, ttl_minutes=1)
    
    # Test setting and getting
    cache.set("hash1", 0.5)
    assert cache.get("hash1") == 0.5
    
    # Test max size
    cache.set("hash2", 0.6)
    cache.set("hash3", 0.7)
    assert cache.get("hash1") is None  # Evicted
    assert cache.get("hash2") == 0.6
    assert cache.get("hash3") == 0.7
    
    # Test TTL
    cache.timestamps["hash2"] = datetime.now() - timedelta(minutes=2)
    assert cache.get("hash2") is None  # Expired

@pytest.mark.asyncio
async def test_analyze_single(sentiment_analyzer):
    result = sentiment_analyzer.analyze_single("Bitcoin to the moon!")
    assert isinstance(result, dict)
    assert "sentiment" in result
    assert "confidence" in result
    assert "source" in result
    assert -1.0 <= result["sentiment"] <= 1.0
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["source"] == "primary"

@pytest.mark.asyncio
async def test_analyze_batch(sentiment_analyzer):
    texts = ["Bitcoin is pumping!", "Market crash incoming."]
    results = sentiment_analyzer.analyze_batch(texts)
    assert len(results) == 2
    for result in results:
        assert isinstance(result, dict)
        assert "sentiment" in result
        assert "confidence" in result
        assert -1.0 <= result["sentiment"] <= 1.0
        assert 0.0 <= result["confidence"] <= 1.0

@pytest.mark.asyncio
async def test_get_temporal_sentiment(sentiment_analyzer):
    # Add mock history
    sentiment_analyzer.sentiment_history = [
        {"sentiment": 0.8, "confidence": 0.9, "timestamp": datetime.now()},
        {"sentiment": 0.7, "confidence": 0.85, "timestamp": datetime.now() - timedelta(minutes=30)},
        {"sentiment": 0.6, "confidence": 0.8, "timestamp": datetime.now() - timedelta(hours=2)}
    ]
    
    result = sentiment_analyzer.get_temporal_sentiment(hours_back=1.0)
    assert isinstance(result, dict)
    assert result["sample_count"] == 2  # Only within last hour
    assert result["avg_sentiment"] == pytest.approx(0.75, 0.01)
    assert result["confidence"] == pytest.approx(0.875, 0.01)

@pytest.mark.asyncio
async def test_get_market_sentiment_signal(sentiment_analyzer):
    # Add mock history for sufficient sample count
    sentiment_analyzer.sentiment_history = [
        {"sentiment": 0.8, "confidence": 0.9, "timestamp": datetime.now()} for _ in range(5)
    ]
    
    result = sentiment_analyzer.get_market_sentiment_signal()
    assert isinstance(result, dict)
    assert "sentiment_adjustment" in result
    assert "confidence_multiplier" in result
    assert "signal_strength" in result
    assert result["sentiment_adjustment"] == pytest.approx(0.24, 0.01)  # 0.8 * 0.3
    assert result["confidence_multiplier"] > 1.0
    assert result["signal_strength"] > 0.0

@pytest.mark.asyncio
async def test_process_live_sentiment(sentiment_integration):
    texts = ["Bitcoin is pumping!"]
    result = await sentiment_integration.process_live_sentiment(texts)
    assert isinstance(result, dict)
    assert "sentiment_adjustment" in result
    assert "confidence_multiplier" in result
    assert -1.0 <= result["sentiment_adjustment"] <= 1.0
    assert result["confidence_multiplier"] >= 1.0

@pytest.mark.asyncio
async def test_enhance_trading_signal(sentiment_integration):
    original_signal = {"symbol": "BTC/USD", "signal_strength": 0.7, "confidence": 0.9}
    result = sentiment_integration.enhance_trading_signal(original_signal, ["Bitcoin is pumping!"])
    assert isinstance(result, dict)
    assert result["symbol"] == "BTC/USD"
    assert result["signal_strength"] > 0.7
    assert result["confidence"] >= 0.9
    assert "sentiment" in result
    assert "raw_sentiment" in result["sentiment"]

@pytest.mark.asyncio
async def test_save_load_state(sentiment_analyzer, tmp_path):
    state_file = tmp_path / "sentiment_state.pkl"
    sentiment_analyzer.sentiment_history = [
        {"sentiment": 0.8, "confidence": 0.9, "timestamp": datetime.now()}
    ]
    
    sentiment_analyzer.save_state(str(state_file))
    assert state_file.exists()
    
    new_analyzer = EnhancedSentimentAnalyzer(sentiment_analyzer.config)
    new_analyzer.load_state(str(state_file))
    assert len(new_analyzer.sentiment_history) == 1
    assert new_analyzer.sentiment_history[0]["sentiment"] == 0.8