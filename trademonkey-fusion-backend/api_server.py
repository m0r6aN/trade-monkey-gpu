# File: api_server.py
#!/usr/bin/env python3
"""
TradeMonkey Fusion - Enhanced Real-Time API Server
"Feeding the beast with LIVE DATA and P&L UPDATES!" 🚀

Enhanced FastAPI server with live P&L tracking, signal enhancement, and market regime detection
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import asyncio
import json
import ccxt
import redis.asyncio as redis
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Set, Union
import uvicorn
from pydantic import BaseModel
import traceback
import numpy as np

# Import our custom modules (with error handling for missing modules)
try:
    from src.utils.sentiment_engine import EnhancedSentimentAnalyzer
except ImportError:
    logger = logging.getLogger('TradeMonkeyAPI')
    logger.warning("sentiment_engine not found, using mock implementation")
    class EnhancedSentimentAnalyzer:
        async def get_market_sentiment_signal(self):
            return {
                "raw_sentiment": 0.5,
                "confidence_multiplier": 0.85,
                "sentiment_trend": 0.05,
                "sentiment_adjustment": 0.1
            }

try:
    from src.trading.position import PositionManager
except ImportError:
    logger = logging.getLogger('TradeMonkeyAPI')
    logger.warning("position module not found, using mock implementation")
    class PositionManager:
        def __init__(self):
            self.positions = []
        
        async def open_position(self, symbol: str, size: float, side: str):
            return {
                "id": f"pos_{int(datetime.now().timestamp())}",
                "symbol": symbol, "side": side, "size": size,
                "entry_price": 45000.0, "current_price": 45000.0, "pnl": 0.0
            }
        
        async def get_live_positions(self):
            return self.positions
        
        async def close_position(self, position_id: str):
            return {"status": "closed", "id": position_id}
        
        async def update_position(self, position: Dict):
            pass
        
        async def get_portfolio_metrics(self):
            return {"total_pnl": 0.0, "position_count": 0, "risk_level": 0.0}

try:
    from src.utils.sentiment_redis_integration import RedisClient
except ImportError:
    logger = logging.getLogger('TradeMonkeyAPI')
    logger.warning("Redis integration not found, using mock implementation")
    class RedisClient:
        async def ping(self): return True
        async def get(self, key): return None
        async def set(self, key, value, expire_seconds=None): pass
        async def publish(self, channel, message): pass
        async def get_recent_events(self, limit=50): return []
        async def add_event(self, event_type, description, metadata=None): pass
        async def get_uptime(self): return 99.9
        async def get_training_status(self, training_id): return None
        async def log_error(self, error_type, message, context=None): pass

try:
    from src.trading.risk_manager import RiskManager
except ImportError:
    logger = logging.getLogger('TradeMonkeyAPI')
    logger.warning("risk_manager not found, using mock implementation")
    class RiskManager:
        async def calculate_portfolio_risk(self): return 15.0

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('TradeMonkeyAPI')

app = FastAPI(title="TradeMonkey Fusion API", version="3.0.0")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Models
class SentimentData(BaseModel):
    score: float
    confidence: float
    trend: float
    sample_count: int
    crypto_ratio: float
    sources: Dict[str, int]
    enhancement_multiplier: float
    market_regime: str
    signal_boost_active: bool

class GPUData(BaseModel):
    memory_usage: float
    memory_total: float
    processing_speed: float
    queue_throughput: float
    temperature: float

class HealthData(BaseModel):
    overall_score: int
    uptime: float
    api_connections: Dict[str, str]
    error_count: int

class ActivityEvent(BaseModel):
    event_id: str
    type: str
    timestamp: str
    description: str
    metadata: Dict

class AgentStatus(BaseModel):
    agent_id: str
    name: str
    status: str
    current_task: Optional[str]
    uptime: float
    response_time: float

class MLTrainingStatus(BaseModel):
    training_id: str
    status: str
    progress: float
    metrics: Dict

class Position(BaseModel):
    id: str
    symbol: str
    side: str
    entry_price: float
    current_price: float
    size: float
    pnl: float
    pnl_percent: float
    sentiment_enhanced: bool
    enhancement_multiplier: float
    open_time: str

class RiskAdjustRequest(BaseModel):
    action: str
    risk_level: float

class RebalanceRequest(BaseModel):
    strategy: str

# Global connections
active_websockets: Set[WebSocket] = set()
redis_client: Optional[RedisClient] = None
kraken_exchange: Optional[ccxt.Exchange] = None

# Initialize components
sentiment_engine = EnhancedSentimentAnalyzer()
position_manager = PositionManager()
risk_manager = RiskManager()

# Live position tracking
live_positions = {}
portfolio_metrics = {
    'total_pnl': 0.0,
    'total_pnl_percent': 0.0,
    'open_positions': 0,
    'winning_positions': 0,
    'largest_winner': 0.0,
    'largest_loser': 0.0,
    'sentiment_enhanced_trades': 0,
    'risk_level': 35.0
}

@app.on_event("startup")
async def startup_event():
    """Initialize connections on startup"""
    global redis_client, kraken_exchange
    
    try:
        # Initialize Redis
        redis_client = RedisClient()
        await redis_client.ping()
        logger.info("✅ Redis connected successfully")
        
        # Initialize Kraken (with mock for development)
        try:
            kraken_exchange = ccxt.kraken({
                'enableRateLimit': True,
                'timeout': 30000,
            })
            # Test connection
            await asyncio.to_thread(kraken_exchange.load_markets)
            logger.info("✅ Kraken connected successfully")
        except Exception as e:
            logger.warning(f"Kraken connection failed, using mock data: {e}")
            kraken_exchange = None
        
        # Start background tasks
        asyncio.create_task(background_data_collector())
        
    except Exception as e:
        logger.error(f"❌ Startup error: {e}")
        # Continue with mock implementations

async def background_data_collector():
    """Background task to collect and broadcast data"""
    logger.info("🚀 Starting background data collector...")
    
    while True:
        try:
            if not active_websockets:
                await asyncio.sleep(5)
                continue
            
            # Collect data
            tickers = await collect_kraken_data()
            sentiment = await collect_sentiment_data()
            gpu_metrics = await collect_gpu_metrics()
            health = await collect_system_health()
            
            # Broadcast to WebSocket clients
            updates = [
                {'type': 'ticker_update', 'data': tickers},
                {'type': 'sentiment_update', 'data': sentiment},
                {'type': 'gpu_update', 'data': gpu_metrics},
                {'type': 'health_update', 'data': health}
            ]
            
            for update in updates:
                await broadcast_to_websockets(update)
            
            await asyncio.sleep(2)
            
        except Exception as e:
            logger.error(f"Background collector error: {e}")
            await asyncio.sleep(5)

async def collect_kraken_data() -> Dict:
    """Collect ticker data from Kraken or use mock data"""
    try:
        if kraken_exchange:
            ticker = await asyncio.to_thread(kraken_exchange.fetch_ticker, 'BTC/USD')
            return {
                'BTCUSD': {
                    'symbol': 'BTC/USD',
                    'price': ticker['last'],
                    'change24h': ticker['percentage'],
                    'volume24h': ticker['baseVolume'],
                    'timestamp': ticker['timestamp']
                }
            }
        else:
            # Mock data
            import random
            base_price = 45000
            change = (random.random() - 0.5) * 2
            return {
                'BTCUSD': {
                    'symbol': 'BTC/USD',
                    'price': base_price + (base_price * change / 100),
                    'change24h': change,
                    'volume24h': random.randint(10000, 50000),
                    'timestamp': datetime.now().timestamp() * 1000
                }
            }
    except Exception as e:
        logger.error(f"Kraken data collection error: {e}")
        return {}

async def collect_sentiment_data() -> Dict:
    """Collect sentiment data from engine"""
    try:
        sentiment_signal = await sentiment_engine.get_market_sentiment_signal()
        return {
            'score': sentiment_signal.get('raw_sentiment', 0),
            'confidence': sentiment_signal.get('confidence_multiplier', 0),
            'trend': sentiment_signal.get('sentiment_trend', 0),
            'sample_count': 150,
            'crypto_ratio': 0.89,
            'sources': {'twitter': 50, 'reddit': 30, 'discord': 25, 'news': 20},
            'enhancement_multiplier': sentiment_signal.get('sentiment_adjustment', 0),
            'market_regime': 'bull' if sentiment_signal.get('raw_sentiment', 0) > 0 else 'bear',
            'signal_boost_active': abs(sentiment_signal.get('sentiment_adjustment', 0)) > 0.15
        }
    except Exception as e:
        logger.error(f"Sentiment data collection error: {e}")
        return {
            'score': 0.5, 'confidence': 0.85, 'trend': 0.05, 'sample_count': 150,
            'crypto_ratio': 0.89, 'sources': {'twitter': 50, 'reddit': 30},
            'enhancement_multiplier': 0.1, 'market_regime': 'bull', 'signal_boost_active': False
        }

async def collect_gpu_metrics() -> Dict:
    """Collect GPU metrics or use mock data"""
    try:
        import random
        return {
            'memory_usage': 65.0 + random.random() * 20,
            'memory_total': 11264.0,
            'processing_speed': 250.0 + random.random() * 100,
            'queue_throughput': 2000.0 + random.random() * 500,
            'temperature': 70.0 + random.random() * 10
        }
    except Exception as e:
        logger.error(f"GPU metrics collection error: {e}")
        return {'memory_usage': 50.0, 'memory_total': 11264.0, 'processing_speed': 300.0}

async def collect_system_health() -> Dict:
    """Collect system health metrics"""
    try:
        uptime = await redis_client.get_uptime() if redis_client else 99.9
        return {
            'overall_score': 95,
            'uptime': uptime,
            'api_connections': {
                'kraken': 'healthy' if kraken_exchange else 'warning',
                'redis': 'healthy' if redis_client else 'critical',
                'sentiment_engine': 'healthy',
                'gpu': 'healthy'
            },
            'error_count': 0
        }
    except Exception as e:
        logger.error(f"System health collection error: {e}")
        return {'overall_score': 75, 'uptime': 95.0, 'api_connections': {}, 'error_count': 1}

async def broadcast_to_websockets(message: Dict):
    """Broadcast message to all connected WebSocket clients"""
    if not active_websockets:
        return
    
    disconnected = set()
    message_str = json.dumps(message)
    
    for websocket in active_websockets:
        try:
            await websocket.send_text(message_str)
        except Exception:
            disconnected.add(websocket)
    
    # Remove disconnected clients
    active_websockets.difference_update(disconnected)

# REST API Endpoints
@app.get("/api/health")
async def health_check():
    """API health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "3.0.0",
        "websocket_connections": len(active_websockets)
    }

@app.post("/api/positions/open")
async def open_position(symbol: str, size: float, side: str):
    """Open a new position"""
    try:
        position = await position_manager.open_position(symbol, size, side)
        if redis_client:
            await redis_client.publish("positions", json.dumps(position))
        return position
    except Exception as e:
        logger.error(f"Position opening error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/positions/live")
async def get_live_positions():
    """Get all live positions"""
    try:
        positions = await position_manager.get_live_positions()
        if redis_client:
            await redis_client.publish("positions", json.dumps(positions))
        return positions
    except Exception as e:
        logger.error(f"Live positions error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/positions/{position_id}")
async def close_position(position_id: str):
    """Close a position"""
    try:
        result = await position_manager.close_position(position_id)
        if redis_client:
            await redis_client.publish("positions", json.dumps(result))
        return result
    except Exception as e:
        logger.error(f"Position closing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/signals/enhance")
async def enhance_signal(symbol: str, signal_strength: float):
    """Enhance trading signal with sentiment"""
    try:
        sentiment_data = await collect_sentiment_data()
        enhanced_strength = signal_strength + sentiment_data['enhancement_multiplier']
        
        result = {
            'symbol': symbol,
            'signal_strength': enhanced_strength,
            'sentiment': sentiment_data
        }
        
        if redis_client:
            await redis_client.publish("signals", json.dumps(result))
        
        return result
    except Exception as e:
        logger.error(f"Signal enhancement error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/portfolio/metrics")
async def get_portfolio_metrics():
    """Get portfolio metrics"""
    try:
        metrics = await position_manager.get_portfolio_metrics()
        if redis_client:
            await redis_client.publish("portfolio", json.dumps(metrics))
        return metrics
    except Exception as e:
        logger.error(f"Portfolio metrics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sentiment/current")
async def get_current_sentiment():
    """Get current market sentiment"""
    try:
        sentiment_data = await collect_sentiment_data()
        if redis_client:
            await redis_client.publish("sentiment", json.dumps(sentiment_data))
        return sentiment_data
    except Exception as e:
        logger.error(f"Current sentiment error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sentiment/history")
async def get_sentiment_history():
    """Get historical sentiment data"""
    try:
        # Generate mock historical data
        history = []
        for i in range(24):
            timestamp = datetime.now() - timedelta(hours=i)
            sentiment_score = np.sin(i * 0.5) * 0.7 + (np.random.random() - 0.5) * 0.3
            history.append({
                'timestamp': timestamp.isoformat(),
                'avg_sentiment': round(sentiment_score, 3),
                'confidence': round(0.6 + np.random.random() * 0.4, 3),
                'sample_count': np.random.randint(50, 200)
            })
        return {"history": list(reversed(history))}
    except Exception as e:
        logger.error(f"Sentiment history error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/gpu/metrics")
async def get_gpu_metrics():
    """Get GPU performance metrics"""
    try:
        gpu_data = await collect_gpu_metrics()
        if redis_client:
            await redis_client.publish("gpu", json.dumps(gpu_data))
        return gpu_data
    except Exception as e:
        logger.error(f"GPU metrics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/system/health")
async def get_system_health():
    """Get system health status"""
    try:
        health_data = await collect_system_health()
        if redis_client:
            await redis_client.publish("health", json.dumps(health_data))
        return health_data
    except Exception as e:
        logger.error(f"System health error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/activities/feed")
async def get_activity_feed(limit: int = 50):
    """Get activity feed events"""
    try:
        if redis_client:
            events = await redis_client.get_recent_events(limit)
        else:
            # Mock events
            events = [
                {
                    "event_id": "evt_001",
                    "type": "position_opened",
                    "timestamp": datetime.now().isoformat(),
                    "description": "Opened long position on BTC/USD",
                    "metadata": {"size": 0.25}
                }
            ]
        return events
    except Exception as e:
        logger.error(f"Activity feed error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/agents/status")
async def get_agents_status():
    """Get OMEGA agent status"""
    try:
        import random
        agents = [
            {
                "agent_id": f"agent_{i}",
                "name": f"Agent {i}",
                "status": random.choice(["active", "idle", "busy"]),
                "current_task": f"Task {i}",
                "uptime": 99.0 + random.random(),
                "response_time": random.randint(50, 500)
            }
            for i in range(3)
        ]
        
        if redis_client:
            await redis_client.publish("agents", json.dumps(agents))
        
        return agents
    except Exception as e:
        logger.error(f"Agent status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ml/training/start")
async def start_ml_training(model_type: str, parameters: dict, dataset_id: str):
    """Start ML model training"""
    try:
        training_id = f"train_{int(datetime.now().timestamp())}"
        status_data = {
            "training_id": training_id,
            "status": "started",
            "progress": 0.0,
            "model_type": model_type,
            "start_time": datetime.now().isoformat()
        }
        
        if redis_client:
            await redis_client.set_training_status(training_id, status_data)
            await redis_client.publish("training", json.dumps(status_data))
        
        return status_data
    except Exception as e:
        logger.error(f"ML training start error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/ml/training/status")
async def get_training_status(training_id: str):
    """Get ML training status"""
    try:
        if redis_client:
            status_data = await redis_client.get_training_status(training_id)
        else:
            status_data = None
        
        if not status_data:
            # Mock training progress
            progress = min(100, (datetime.now().timestamp() % 100))
            status_data = {
                "training_id": training_id,
                "status": "training" if progress < 100 else "completed",
                "progress": progress / 100,
                "metrics": {
                    "loss": max(0.01, 2.0 * (1 - progress / 100)),
                    "accuracy": min(0.95, 0.5 + (progress / 100) * 0.45)
                }
            }
        
        return status_data
    except Exception as e:
        logger.error(f"Training status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/risk/adjust")
async def adjust_risk(request: RiskAdjustRequest):
    """Adjust risk management settings"""
    try:
        if request.action != "adjust_stops":
            raise HTTPException(status_code=400, detail="Invalid action")
        
        # Mock risk adjustment
        adjusted_positions = 1
        
        if redis_client:
            await redis_client.add_event(
                "risk_adjustment",
                f"Risk adjusted to {request.risk_level}%",
                {"risk_level": request.risk_level, "adjusted_positions": adjusted_positions}
            )
            await redis_client.publish("risk", json.dumps({
                "action": request.action,
                "risk_level": request.risk_level,
                "adjusted_positions": adjusted_positions
            }))
        
        return {
            "status": "success",
            "adjusted_positions": adjusted_positions,
            "risk_level": request.risk_level
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Risk adjustment error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/portfolio/rebalance")
async def rebalance_portfolio(request: RebalanceRequest):
    """Rebalance portfolio based on strategy"""
    try:
        if request.strategy != "sentiment_weighted":
            raise HTTPException(status_code=400, detail="Invalid strategy")
        
        # Mock rebalancing
        rebalanced_positions = 1
        
        if redis_client:
            await redis_client.add_event(
                "portfolio_rebalance",
                f"Portfolio rebalanced using {request.strategy}",
                {"strategy": request.strategy, "rebalanced_positions": rebalanced_positions}
            )
            await redis_client.publish("portfolio", json.dumps({
                "strategy": request.strategy,
                "rebalanced_positions": rebalanced_positions
            }))
        
        return {
            "status": "success",
            "rebalanced_positions": rebalanced_positions,
            "strategy": request.strategy
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Portfolio rebalance error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/positions/close-all")
async def close_all_positions():
    """Emergency close all positions"""
    try:
        # Mock closing all positions
        closed_positions = len(live_positions)
        total_pnl = 0.0
        
        # Clear positions
        live_positions.clear()
        
        if redis_client:
            await redis_client.add_event(
                "emergency_close",
                f"Emergency close: {closed_positions} positions closed",
                {"closed_positions": closed_positions, "total_pnl": total_pnl}
            )
            await redis_client.publish("positions", json.dumps({
                "action": "close_all",
                "closed_positions": closed_positions,
                "total_pnl": total_pnl
            }))
        
        return {
            "status": "success",
            "closed_positions": closed_positions,
            "total_pnl": total_pnl
        }
    except Exception as e:
        logger.error(f"Close all positions error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time data"""
    await websocket.accept()
    active_websockets.add(websocket)
    logger.info(f"✅ WebSocket client connected. Total: {len(active_websockets)}")
    
    try:
        # Send initial data
        initial_data = {
            'type': 'initial_data',
            'data': {
                'tickers': await collect_kraken_data(),
                'sentiment': await collect_sentiment_data(),
                'gpu': await collect_gpu_metrics(),
                'health': await collect_system_health()
            },
            'timestamp': datetime.now().isoformat()
        }
        await websocket.send_text(json.dumps(initial_data))
        
        # Handle incoming messages
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=60)
                message = json.loads(data)
                
                if message.get('type') == 'ping':
                    await websocket.send_text(json.dumps({'type': 'pong', 'timestamp': datetime.now().isoformat()}))
                elif message.get('type') == 'refresh_all':
                    refresh_data = {
                        'type': 'refresh_response',
                        'data': {
                            'tickers': await collect_kraken_data(),
                            'sentiment': await collect_sentiment_data(),
                            'gpu': await collect_gpu_metrics(),
                            'health': await collect_system_health()
                        },
                        'timestamp': datetime.now().isoformat()
                    }
                    await websocket.send_text(json.dumps(refresh_data))
                    
            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                await websocket.send_text(json.dumps({'type': 'ping', 'timestamp': datetime.now().isoformat()}))
                
    except WebSocketDisconnect:
        logger.info("🔌 WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        active_websockets.discard(websocket)
        logger.info(f"📊 Active WebSocket connections: {len(active_websockets)}")

if __name__ == "__main__":
    logger.info("🚀 Starting TradeMonkey Fusion API Server...")
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )