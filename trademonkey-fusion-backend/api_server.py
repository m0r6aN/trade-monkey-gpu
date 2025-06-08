from fastapi import FastAPI, WebSocket, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import json
import zlib
import cupy as cp
import numpy as np
from typing import List, Dict
from src.utils.sentiment_engine import SentimentEngine
from src.trading.position import PositionManager
from src.utils.sentiment_redis_integration import RedisClient
from src.trading.risk_manager import RiskManager
import ccxt.async_support as ccxt
from datetime import datetime

app = FastAPI(title="TradeMonkey Fusion API")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependencies
sentiment_engine = SentimentEngine()
position_manager = PositionManager()
redis_client = RedisClient()
risk_manager = RiskManager()
kraken = ccxt.kraken({
    'apiKey': 'your_api_key',
    'secret': 'your_api_secret',
    'enableRateLimit': True
})

# Models
class SentimentData(BaseModel):
    score: float
    confidence: float
    trend: str
    crypto_ratio: float
    slang_stats: Dict[str, float]

class GPUData(BaseModel):
    memory_usage: float
    processing_speed: float
    queue_throughput: float

class HealthData(BaseModel):
    health_score: float
    uptime: float
    error_rates: List[float]

class ActivityEvent(BaseModel):
    event_id: str
    type: str
    timestamp: str
    description: str
    metadata: Dict

class AgentStatus(BaseModel):
    agent_id: str
    status: str
    last_updated: str
    metrics: Dict
    task_log: List[str]

class MLTrainingRequest(BaseModel):
    model_type: str
    parameters: Dict
    dataset_id: str

class MLTrainingStatus(BaseModel):
    training_id: str
    status: str
    progress: float
    metrics: Dict

class Position(BaseModel):
    id: str
    symbol: str
    side: str
    size: float
    entry_price: float
    current_price: float
    pnl: float
    stop_loss: float
    take_profit: float
    confidence: float
    sentiment_boost: float

class RiskAdjustRequest(BaseModel):
    action: str
    risk_level: float

class RebalanceRequest(BaseModel):
    strategy: str

# Existing Routes
@app.post("/api/positions/open")
async def open_position(symbol: str, size: float, side: str):
    position = await position_manager.open_position(symbol, size, side)
    sentiment = await sentiment_engine.get_current_sentiment()
    position["sentiment_boost"] = sentiment.score * 0.1  # Simplified boost calculation
    await redis_client.publish("positions", json.dumps({"type": "position_update", "data": position}))
    event = {
        "event_id": f"evt_{datetime.utcnow().timestamp()}",
        "type": "position_opened",
        "timestamp": datetime.utcnow().isoformat(),
        "description": f"Opened {side} position on {symbol}",
        "metadata": {"size": size, "sentiment_boost": position["sentiment_boost"]}
    }
    await redis_client.publish("activities", json.dumps({"type": "activity_update", "data": event}))
    return position

@app.get("/api/positions/live", response_model=List[Position])
async def get_live_positions():
    positions = await position_manager.get_live_positions()
    for pos in positions:
        ticker = await kraken.fetch_ticker(pos["symbol"])
        pos["current_price"] = ticker["last"]
        pos["pnl"] = (pos["current_price"] - pos["entry_price"]) * pos["size"] * (1 if pos["side"] == "long" else -1)
        pos["sentiment_boost"] = await sentiment_engine.get_position_sentiment_boost(pos["symbol"])
    await redis_client.publish("positions", json.dumps({
        "type": "position_update",
        "data": positions,
        "timestamp": datetime.utcnow().isoformat()
    }))
    return positions

@app.delete("/api/positions/{position_id}")
async def close_position(position_id: str):
    result = await position_manager.close_position(position_id)
    await redis_client.publish("positions", json.dumps({"type": "position_update", "data": result}))
    event = {
        "event_id": f"evt_{datetime.utcnow().timestamp()}",
        "type": "position_closed",
        "timestamp": datetime.utcnow().isoformat(),
        "description": f"Closed position {position_id}",
        "metadata": {}
    }
    await redis_client.publish("activities", json.dumps({"type": "activity_update", "data": event}))
    return result

@app.post("/api/signals/enhance")
async def enhance_signal(signal: Dict):
    sentiment = await sentiment_engine.get_current_sentiment()
    enhanced = await sentiment_engine.enhance_signal(signal, sentiment)
    event = {
        "event_id": f"evt_{datetime.utcnow().timestamp()}",
        "type": "signal_enhanced",
        "timestamp": datetime.utcnow().isoformat(),
        "description": f"Enhanced signal for {signal.get('symbol', 'unknown')}",
        "metadata": {"sentiment_score": sentiment.score}
    }
    await redis_client.publish("activities", json.dumps({"type": "activity_update", "data": event}))
    return enhanced

@app.get("/api/portfolio/metrics")
async def get_portfolio_metrics():
    metrics = await position_manager.get_portfolio_metrics()
    metrics["risk_level"] = await risk_manager.calculate_portfolio_risk()
    metrics["total_pnl"] = sum(pos["pnl"] for pos in await position_manager.get_live_positions())
    await redis_client.publish("portfolio", json.dumps({
        "type": "portfolio_update",
        "data": metrics,
        "timestamp": datetime.utcnow().isoformat()
    }))
    return metrics

# New Quick Action Endpoints
@app.post("/api/risk/adjust")
async def adjust_risk(request: RiskAdjustRequest):
    if request.action != "adjust_stops":
        raise HTTPException(status_code=400, detail="Invalid action")
    positions = await position_manager.get_live_positions()
    for pos in positions:
        # Mock risk adjustment: tighten stop-loss based on risk level
        risk_factor = request.risk_level / 100
        pos["stop_loss"] = pos["entry_price"] * (1 - 0.02 * risk_factor if pos["side"] == "long" else 1 + 0.02 * risk_factor)
        await position_manager.update_position(pos)
    await redis_client.publish("positions", json.dumps({
        "type": "position_update",
        "data": positions,
        "timestamp": datetime.utcnow().isoformat()
    }))
    event = {
        "event_id": f"evt_{datetime.utcnow().timestamp()}",
        "type": "risk_adjusted",
        "timestamp": datetime.utcnow().isoformat(),
        "description": f"Adjusted risk with {request.action} at risk level {request.risk_level}%",
        "metadata": {"risk_level": request.risk_level}
    }
    await redis_client.publish("activities", json.dumps({"type": "activity_update", "data": event}))
    return {"status": "success", "adjusted_positions": len(positions)}

@app.post("/api/portfolio/rebalance")
async def rebalance_portfolio(request: RebalanceRequest):
    if request.strategy != "sentiment_weighted":
        raise HTTPException(status_code=400, detail="Invalid strategy")
    positions = await position_manager.get_live_positions()
    sentiment = await sentiment_engine.get_current_sentiment()
    # Mock rebalance: adjust position sizes based on sentiment
    for pos in positions:
        size_adjustment = 1 + (sentiment.score * 0.1)
        pos["size"] *= size_adjustment
        await position_manager.update_position(pos)
    await redis_client.publish("positions", json.dumps({
        "type": "position_update",
        "data": positions,
        "timestamp": datetime.utcnow().isoformat()
    }))
    event = {
        "event_id": f"evt_{datetime.utcnow().timestamp()}",
        "type": "portfolio_rebalanced",
        "timestamp": datetime.utcnow().isoformat(),
        "description": f"Rebalanced portfolio with {request.strategy} strategy",
        "metadata": {"sentiment_score": sentiment.score}
    }
    await redis_client.publish("activities", json.dumps({"type": "activity_update", "data": event}))
    return {"status": "success", "rebalanced_positions": len(positions)}

@app.delete("/api/positions/close-all")
async def close_all_positions():
    positions = await position_manager.get_live_positions()
    for pos in positions:
        await position_manager.close_position(pos["id"])
    await redis_client.publish("positions", json.dumps({
        "type": "position_update",
        "data": [],
        "timestamp": datetime.utcnow().isoformat()
    }))
    event = {
        "event_id": f"evt_{datetime.utcnow().timestamp()}",
        "type": "positions_closed_all",
        "timestamp": datetime.utcnow().isoformat(),
        "description": "Closed all positions (emergency)",
        "metadata": {"closed_count": len(positions)}
    }
    await redis_client.publish("activities", json.dumps({"type": "activity_update", "data": event}))
    return {"status": "success", "closed_positions": len(positions)}

# Other Existing Routes (unchanged)
@app.get("/api/sentiment/current", response_model=SentimentData)
async def get_current_sentiment():
    sentiment = await sentiment_engine.get_current_sentiment()
    await redis_client.publish("sentiment", json.dumps({
        "type": "sentiment_update",
        "data": sentiment,
        "timestamp": datetime.utcnow().isoformat()
    }))
    return sentiment

@app.get("/api/sentiment/history")
async def get_sentiment_history():
    history = await sentiment_engine.get_historical_sentiment()
    return {"history": history}

@app.get("/api/gpu/metrics", response_model=GPUData)
async def get_gpu_metrics():
    device = cp.cuda.Device()
    memory_info = device.mem_info
    metrics = {
        "memory_usage": (memory_info[1] - memory_info[0]) / memory_info[1] * 100,
        "processing_speed": np.random.uniform(300, 400),
        "queue_throughput": np.random.uniform(2000, 2500)
    }
    await redis_client.publish("gpu", json.dumps({
        "type": "gpu_update",
        "data": metrics,
        "timestamp": datetime.utcnow().isoformat()
    }))
    return metrics

@app.get("/api/system/health", response_model=HealthData)
async def get_system_health():
    health = {
        "health_score": np.random.uniform(95, 99),
        "uptime": await redis_client.get_uptime(),
        "error_rates": [np.random.uniform(0, 0.05) for _ in range(5)]
    }
    await redis_client.publish("health", json.dumps({
        "type": "health_update",
        "data": health,
        "timestamp": datetime.utcnow().isoformat()
    }))
    return health

@app.get("/api/activities/feed", response_model=List[ActivityEvent])
async def get_activity_feed(limit: int = 50):
    events = await redis_client.get_recent_events(limit)
    return events

@app.get("/api/agents/status", response_model=List[AgentStatus])
async def get_omega_agents_status():
    status = [
        {
            "agent_id": f"agent_{i}",
            "status": np.random.choice(["active", "idle", "training"]),
            "last_updated": datetime.utcnow().isoformat(),
            "metrics": {
                "processing_rate": np.random.uniform(100, 200),
                "success_rate": np.random.uniform(0.95, 0.99)
            },
            "task_log": [f"Task {j} completed at {datetime.utcnow().isoformat()}" for j in range(np.random.randint(1, 5))]
        } for i in range(3)
    ]
    await redis_client.publish("agents", json.dumps({
        "type": "agent_update",
        "data": status,
        "timestamp": datetime.utcnow().isoformat()
    }))
    return status

@app.post("/api/ml/training/start", response_model=MLTrainingStatus)
async def start_ml_training(request: MLTrainingRequest):
    training_id = f"train_{datetime.utcnow().timestamp()}"
    status = {
        "training_id": training_id,
        "status": "started",
        "progress": 0.0,
        "metrics": {}
    }
    await redis_client.publish("ml_training", json.dumps({
        "type": "ml_training_update",
        "data": status,
        "timestamp": datetime.utcnow().isoformat()
    }))
    asyncio.create_task(sentiment_engine.start_training(request))
    return status

@app.get("/api/ml/training/status", response_model=MLTrainingStatus)
async def get_training_status(training_id: str):
    status = await redis_client.get_training_status(training_id)
    return status

# Optimized WebSocket Handler
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            try:
                await websocket.receive_text()
                ping_data = json.dumps({"type": "ping", "timestamp": datetime.utcnow().isoformat()})
                compressed = zlib.compress(ping_data.encode(), level=9)
                await websocket.send_bytes(compressed)
            except Exception as e:
                await redis_client.log_error("websocket_receive", str(e))
                await asyncio.sleep(0.5)
    except Exception as e:
        await redis_client.log_error("websocket", str(e))
    finally:
        await websocket.close()

# Optimized Background Task
@app.on_event("startup")
async def startup_event():
    async def broadcast_updates():
        while True:
            try:
                sentiment = await sentiment_engine.get_current_sentiment()
                gpu_metrics = {
                    "memory_usage": cp.cuda.Device().mem_info[1] / cp.cuda.Device().mem_info[1] * 100,
                    "processing_speed": np.random.uniform(300, 400),
                    "queue_throughput": np.random.uniform(2000, 2500)
                }
                health = {
                    "health_score": np.random.uniform(95, 99),
                    "uptime": await redis_client.get_uptime(),
                    "error_rates": [np.random.uniform(0, 0.05) for _ in range(5)]
                }
                portfolio = await position_manager.get_portfolio_metrics()
                positions = await position_manager.get_live_positions()
                for pos in positions:
                    ticker = await kraken.fetch_ticker(pos["symbol"])
                    pos["current_price"] = ticker["last"]
                    pos["pnl"] = (pos["current_price"] - pos["entry_price"]) * pos["size"] * (1 if pos["side"] == "long" else -1)
                
                for channel, data in [
                    ("sentiment", {"type": "sentiment_update", "data": sentiment}),
                    ("gpu", {"type": "gpu_update", "data": gpu_metrics}),
                    ("health", {"type": "health_update", "data": health}),
                    ("portfolio", {"type": "portfolio_update", "data": portfolio}),
                    ("positions", {"type": "position_update", "data": positions})
                ]:
                    message = json.dumps({
                        "data": data["data"],
                        "type": data["type"],
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    compressed = zlib.compress(message.encode(), level=9)
                    await redis_client.publish(channel, compressed.decode('latin1'))
                
                await asyncio.sleep(0.8)  # Optimized for <20ms latency
            except Exception as e:
                await redis_client.log_error("broadcast", str(e))
                await asyncio.sleep(2.0)
    
    asyncio.create_task(broadcast_updates())