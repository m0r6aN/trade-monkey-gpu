# File: tests/test_api_server.py
import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
import json
from datetime import datetime
from api_server import app, SentimentData, GPUData, HealthData, ActivityEvent, AgentStatus, MLTrainingStatus, Position, RiskAdjustRequest, RebalanceRequest
import ccxt.async_support as ccxt

@pytest_asyncio.fixture
async def client():
    return TestClient(app)

@pytest_asyncio.fixture
async def mock_kraken():
    kraken = AsyncMock(spec=ccxt.kraken)
    kraken.fetch_ticker.return_value = {"last": 65000.0}
    return kraken

@pytest_asyncio.fixture
async def mock_sentiment_engine():
    sentiment_engine = AsyncMock()
    sentiment_engine.get_market_sentiment_signal.return_value = {
        "sentiment_adjustment": 0.1,
        "confidence_multiplier": 1.1,
        "signal_strength": 0.85,
        "raw_sentiment": 0.85,
        "sentiment_trend": 0.05
    }
    sentiment_engine.get_temporal_sentiment.return_value = {
        "avg_sentiment": 0.75,
        "sentiment_trend": 0.05,
        "confidence": 0.92,
        "sample_count": 10
    }
    return sentiment_engine

@pytest_asyncio.fixture
async def mock_position_manager():
    position_manager = AsyncMock()
    position_manager.open_position.return_value = {
        "id": "pos_001",
        "symbol": "BTC/USD",
        "side": "long",
        "size": 0.25,
        "entry_price": 64500.0,
        "current_price": 65000.0,
        "pnl": 125.0,
        "pnl_percent": 0.78,
        "sentiment_enhanced": True,
        "enhancement_multiplier": 1.1,
        "open_time": "2024-01-01T12:00:00Z"
    }
    position_manager.get_live_positions.return_value = [
        {
            "id": "pos_001",
            "symbol": "BTC/USD",
            "side": "long",
            "size": 0.25,
            "entry_price": 64500.0,
            "current_price": 65000.0,
            "pnl": 125.0,
            "pnl_percent": 0.78,
            "sentiment_enhanced": True,
            "enhancement_multiplier": 1.1,
            "open_time": "2024-01-01T12:00:00Z"
        }
    ]
    position_manager.close_position.return_value = {"status": "closed"}
    position_manager.get_portfolio_metrics.return_value = {
        "total_pnl": 125.0,
        "risk_level": 35.0
    }
    position_manager.update_position.return_value = None
    return position_manager

@pytest_asyncio.fixture
async def mock_redis_client():
    redis_client = AsyncMock()
    redis_client.get_recent_events.return_value = [
        {
            "event_id": "evt_001",
            "type": "position_opened",
            "timestamp": "2025-06-08T00:00:00Z",
            "description": "Opened long position on BTC/USD",
            "metadata": {"size": 0.25}
        }
    ]
    redis_client.get_uptime.return_value = 99.9
    redis_client.get_training_status.return_value = {
        "training_id": "train_001",
        "status": "started",
        "progress": 0.5,
        "metrics": {}
    }
    redis_client.publish.return_value = None
    redis_client.log_error.return_value = None
    return redis_client

@pytest_asyncio.fixture
async def mock_risk_manager():
    risk_manager = AsyncMock()
    risk_manager.calculate_portfolio_risk.return_value = 35.0
    return risk_manager

@pytest.mark.asyncio
async def test_open_position(client, mock_kraken, mock_sentiment_engine, mock_position_manager, mock_redis_client):
    with patch("api_server.kraken_exchange", mock_kraken), \
         patch("api_server.sentiment_engine", mock_sentiment_engine), \
         patch("api_server.position_manager", mock_position_manager), \
         patch("api_server.redis_client", mock_redis_client):
        response = client.post("/api/positions/open?symbol=BTC/USD&size=0.25&side=long")
        assert response.status_code == 200
        data = response.json()
        assert Position(**data)
        assert data["symbol"] == "BTC/USD"
        assert data["side"] == "long"
        assert data["size"] == 0.25
        assert data["pnl_percent"] == 0.78
        assert data["sentiment_enhanced"] == True
        assert data["enhancement_multiplier"] == 1.1
        assert data["open_time"] == "2024-01-01T12:00:00Z"
        mock_redis_client.publish.assert_called()

@pytest.mark.asyncio
async def test_get_live_positions(client, mock_kraken, mock_position_manager, mock_sentiment_engine, mock_redis_client):
    with patch("api_server.kraken_exchange", mock_kraken), \
         patch("api_server.position_manager", mock_position_manager), \
         patch("api_server.sentiment_engine", mock_sentiment_engine), \
         patch("api_server.redis_client", mock_redis_client):
        response = client.get("/api/positions/live")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0
        assert Position(**data[0])
        assert data[0]["current_price"] == 65000.0
        assert data[0]["pnl"] == 125.0
        mock_redis_client.publish.assert_called()

@pytest.mark.asyncio
async def test_close_position(client, mock_position_manager, mock_redis_client):
    with patch("api_server.position_manager", mock_position_manager), \
         patch("api_server.redis_client", mock_redis_client):
        response = client.delete("/api/positions/pos_001")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "closed"
        mock_redis_client.publish.assert_called()

@pytest.mark.asyncio
async def test_enhance_signal(client, mock_sentiment_engine, mock_redis_client):
    with patch("api_server.sentiment_engine", mock_sentiment_engine), \
         patch("api_server.redis_client", mock_redis_client):
        response = client.post("/api/signals/enhance", json={"symbol": "BTC/USD", "signal_strength": 0.7})
        assert response.status_code == 200
        data = response.json()
        assert data["symbol"] == "BTC/USD"
        assert data["signal_strength"] == 1.55  # 0.7 + 0.85
        assert "sentiment" in data
        mock_redis_client.publish.assert_called()

@pytest.mark.asyncio
async def test_get_portfolio_metrics(client, mock_position_manager, mock_risk_manager, mock_redis_client):
    with patch("api_server.position_manager", mock_position_manager), \
         patch("api_server.risk_manager", mock_risk_manager), \
         patch("api_server.redis_client", mock_redis_client):
        response = client.get("/api/portfolio/metrics")
        assert response.status_code == 200
        data = response.json()
        assert data["total_pnl"] == 125.0
        assert data["risk_level"] == 35.0
        mock_redis_client.publish.assert_called()

@pytest.mark.asyncio
async def test_get_current_sentiment(client, mock_sentiment_engine, mock_redis_client):
    with patch("api_server.sentiment_engine", mock_sentiment_engine), \
         patch("api_server.redis_client", mock_redis_client):
        response = client.get("/api/sentiment/current")
        assert response.status_code == 200
        data = response.json()
        assert SentimentData(**data)
        assert data["score"] == 0.85
        assert data["confidence"] == 1.1
        mock_redis_client.publish.assert_called()

@pytest.mark.asyncio
async def test_get_sentiment_history(client, mock_sentiment_engine):
    with patch("api_server.sentiment_engine", mock_sentiment_engine):
        response = client.get("/api/sentiment/history")
        assert response.status_code == 200
        data = response.json()
        assert "history" in data
        assert len(data["history"]) > 0
        assert data["history"][0]["avg_sentiment"] == -0.665

@pytest.mark.asyncio
async def test_get_gpu_metrics(client, mock_redis_client):
    with patch("api_server.redis_client", mock_redis_client):
        response = client.get("/api/gpu/metrics")
        assert response.status_code == 200
        data = response.json()
        assert GPUData(**data) # Validates structure
        assert 65.0 <= data["memory_usage"] < 85.0 # Check if value is in expected random range
        assert data["memory_total"] == 11264.0 # Check another static mock value
        mock_redis_client.publish.assert_called()

@pytest.mark.asyncio
async def test_get_system_health(client, mock_redis_client):
    with patch("api_server.redis_client", mock_redis_client):
        response = client.get("/api/system/health")
        assert response.status_code == 200
        data = response.json()
        assert HealthData(**data)
        assert data["uptime"] == 99.9
        mock_redis_client.publish.assert_called()

@pytest.mark.asyncio
async def test_get_activity_feed(client, mock_redis_client):
    with patch("api_server.redis_client", mock_redis_client):
        response = client.get("/api/activities/feed?limit=10")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0
        assert ActivityEvent(**data[0])
        assert data[0]["event_id"] == "evt_001"

@pytest.mark.asyncio
async def test_get_omega_agents_status(client, mock_redis_client):
    with patch("api_server.redis_client", mock_redis_client):
        response = client.get("/api/agents/status")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0
        assert AgentStatus(**data[0])
        assert data[0]["agent_id"].startswith("agent_")
        mock_redis_client.publish.assert_called()

@pytest.mark.asyncio
async def test_start_ml_training(client, mock_sentiment_engine, mock_redis_client):
    with patch("api_server.sentiment_engine", mock_sentiment_engine), \
         patch("api_server.redis_client", mock_redis_client):
        response = client.post("/api/ml/training/start", json={
            "model_type": "sentiment",
            "parameters": {},
            "dataset_id": "ds_001"
        })
        assert response.status_code == 200
        data = response.json()
        assert MLTrainingStatus(**data)
        assert data["status"] == "started"
        mock_redis_client.publish.assert_called()

@pytest.mark.asyncio
async def test_get_training_status(client, mock_redis_client):
    with patch("api_server.redis_client", mock_redis_client):
        response = client.get("/api/ml/training/status?training_id=train_001")
        assert response.status_code == 200
        data = response.json()
        assert MLTrainingStatus(**data)
        assert data["training_id"] == "train_001"
        assert data["progress"] == 0.5

@pytest.mark.asyncio
async def test_adjust_risk(client, mock_position_manager, mock_redis_client):
    with patch("api_server.position_manager", mock_position_manager), \
         patch("api_server.redis_client", mock_redis_client):
        response = client.post("/api/risk/adjust", json={
            "action": "adjust_stops",
            "risk_level": 50.0
        })
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["adjusted_positions"] == 1
        mock_position_manager.update_position.assert_called()
        mock_redis_client.publish.assert_called()

@pytest.mark.asyncio
async def test_adjust_risk_invalid_action(client):
    response = client.post("/api/risk/adjust", json={
        "action": "invalid",
        "risk_level": 50.0
    })
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid action"

@pytest.mark.asyncio
async def test_rebalance_portfolio(client, mock_position_manager, mock_sentiment_engine, mock_redis_client):
    with patch("api_server.position_manager", mock_position_manager), \
         patch("api_server.sentiment_engine", mock_sentiment_engine), \
         patch("api_server.redis_client", mock_redis_client):
        response = client.post("/api/portfolio/rebalance", json={
            "strategy": "sentiment_weighted"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["rebalanced_positions"] == 1
        mock_position_manager.update_position.assert_called()
        mock_redis_client.publish.assert_called()

@pytest.mark.asyncio
async def test_rebalance_portfolio_invalid_strategy(client):
    response = client.post("/api/portfolio/rebalance", json={
        "strategy": "invalid"
    })
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid strategy"

@pytest.mark.asyncio
async def test_close_all_positions(client, mock_position_manager, mock_redis_client):
    with patch("api_server.position_manager", mock_position_manager), \
         patch("api_server.redis_client", mock_redis_client):
        response = client.delete("/api/positions/close-all")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["closed_positions"] == 1
        mock_position_manager.close_position.assert_called()
        mock_redis_client.publish.assert_called()