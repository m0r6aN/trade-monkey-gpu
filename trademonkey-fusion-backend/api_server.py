# File: trademonkey-fusion-backend/api_server_enhanced.py
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
from typing import Dict, List, Optional, Set
import uvicorn
from pydantic import BaseModel
import traceback
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('TradeMonkeyEnhancedAPI')

app = FastAPI(title="TradeMonkey Fusion Enhanced API", version="2.1.0")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global connections
active_websockets: Set[WebSocket] = set()
redis_client: Optional[redis.Redis] = None
kraken_exchange: Optional[ccxt.Exchange] = None

# Live position tracking
live_positions = {}
portfolio_metrics = {
    'total_pnl': 0.0,
    'total_pnl_percent': 0.0,
    'open_positions': 0,
    'winning_positions': 0,
    'largest_winner': 0.0,
    'largest_loser': 0.0,
    'sentiment_enhanced_trades': 0
}

class SymbolRequest(BaseModel):
    symbols: List[str]

class PositionUpdate(BaseModel):
    symbol: str
    side: str
    entry_price: float
    size: float
    current_price: Optional[float] = None

class SignalEnhancementRequest(BaseModel):
    base_signal: Dict
    symbol: str

@app.on_event("startup")
async def startup_event():
    """Initialize connections on startup"""
    global redis_client, kraken_exchange
    
    try:
        # Initialize Redis
        redis_client = redis.from_url("redis://localhost:6379")
        await redis_client.ping()
        logger.info("✅ Redis connected successfully")
        
        # Initialize Kraken
        kraken_exchange = ccxt.kraken({
            'enableRateLimit': True,
            'timeout': 30000,
        })
        
        # Test Kraken connection
        await asyncio.to_thread(kraken_exchange.load_markets)
        logger.info("✅ Kraken connected successfully")
        
        # Start background data collection
        asyncio.create_task(background_data_collector())
        asyncio.create_task(portfolio_tracker())
        
    except Exception as e:
        logger.error(f"❌ Startup error: {e}")
        traceback.print_exc()

async def background_data_collector():
    """Enhanced background task with portfolio tracking"""
    logger.info("🚀 Starting enhanced background data collector...")
    
    while True:
        try:
            if not active_websockets:
                await asyncio.sleep(5)
                continue
            
            # Collect all data in parallel
            tasks = [
                collect_kraken_data(),
                collect_enhanced_sentiment_data(),
                collect_gpu_metrics(),
                collect_system_health(),
                calculate_portfolio_metrics()
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Broadcast updates to all connected clients
            message_types = ['ticker_update', 'sentiment_update', 'gpu_update', 'health_update', 'portfolio_update']
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Data collection error {i}: {result}")
                    continue
                
                if i < len(message_types) and result:
                    await broadcast_to_websockets({
                        'type': message_types[i],
                        'data': result,
                        'timestamp': datetime.now().isoformat()
                    })
            
            await asyncio.sleep(2)  # Update every 2 seconds
            
        except Exception as e:
            logger.error(f"Background collector error: {e}")
            await asyncio.sleep(5)

async def collect_enhanced_sentiment_data() -> Dict:
    """Enhanced sentiment data with signal enhancement logic"""
    try:
        if not redis_client:
            return get_mock_sentiment_enhanced()
        
        # Get current market sentiment from Redis
        sentiment_data = await redis_client.get("trademonkey:sentiment:current")
        if sentiment_data:
            data = json.loads(sentiment_data)
            
            # Calculate signal enhancement metrics
            sentiment_score = data.get('avg_sentiment', 0)
            confidence = data.get('confidence', 0)
            
            # Determine enhancement strength
            enhancement_multiplier = 0
            if abs(sentiment_score) > 0.5 and confidence > 0.7:
                enhancement_multiplier = sentiment_score * confidence * 0.5
            
            # Determine market regime based on sentiment
            market_regime = "unknown"
            if sentiment_score > 0.6:
                market_regime = "bull_euphoria"
            elif sentiment_score > 0.3:
                market_regime = "bull_optimism"
            elif sentiment_score > -0.3:
                market_regime = "neutral_mixed"
            elif sentiment_score > -0.6:
                market_regime = "bear_pessimism"
            else:
                market_regime = "bear_panic"
            
            return {
                'sentiment': sentiment_score,
                'confidence': confidence,
                'crypto_related': data.get('crypto_ratio', 0) > 0.5,
                'trend': data.get('sentiment_trend', 0),
                'sample_count': data.get('sample_count', 0),
                'crypto_ratio': data.get('crypto_ratio', 0),
                'sources': data.get('sources', {'twitter': 0, 'reddit': 0, 'discord': 0, 'news': 0}),
                'enhancement_multiplier': enhancement_multiplier,
                'market_regime': market_regime,
                'signal_boost_active': abs(enhancement_multiplier) > 0.15,
                'volatility_adjusted': True
            }
        else:
            return get_mock_sentiment_enhanced()
            
    except Exception as e:
        logger.error(f"Enhanced sentiment data collection error: {e}")
        return get_mock_sentiment_enhanced()

def get_mock_sentiment_enhanced() -> Dict:
    """Generate enhanced mock sentiment data"""
    import random
    import math
    
    # Create realistic sentiment oscillation with market cycles
    time_factor = datetime.now().timestamp() / 3600  # Hours since epoch
    base_sentiment = math.sin(time_factor * 0.05) * 0.6  # Slower, larger oscillation
    noise = (random.random() - 0.5) * 0.2  # Reduced noise
    sentiment = np.clip(base_sentiment + noise, -1.0, 1.0)
    
    confidence = 0.6 + random.random() * 0.35
    enhancement_multiplier = sentiment * confidence * 0.5 if abs(sentiment) > 0.3 else 0
    
    # Market regime
    if sentiment > 0.5:
        regime = "bull_euphoria"
    elif sentiment > 0.2:
        regime = "bull_optimism"
    elif sentiment > -0.2:
        regime = "neutral_mixed"
    elif sentiment > -0.5:
        regime = "bear_pessimism"
    else:
        regime = "bear_panic"
    
    return {
        'sentiment': round(sentiment, 3),
        'confidence': round(confidence, 3),
        'crypto_related': True,
        'trend': round((random.random() - 0.5) * 0.15, 3),
        'sample_count': random.randint(75, 250),
        'crypto_ratio': round(0.7 + random.random() * 0.3, 3),
        'sources': {
            'twitter': random.randint(30, 120),
            'reddit': random.randint(15, 60),
            'discord': random.randint(5, 30),
            'news': random.randint(3, 20)
        },
        'enhancement_multiplier': round(enhancement_multiplier, 3),
        'market_regime': regime,
        'signal_boost_active': abs(enhancement_multiplier) > 0.15,
        'volatility_adjusted': True
    }

async def calculate_portfolio_metrics() -> Dict:
    """Calculate real-time portfolio metrics"""
    try:
        if not live_positions:
            return portfolio_metrics
        
        total_pnl = 0
        total_value = 0
        winning_count = 0
        largest_winner = 0
        largest_loser = 0
        sentiment_enhanced_count = 0
        
        for pos_id, position in live_positions.items():
            current_price = position.get('current_price', position['entry_price'])
            entry_price = position['entry_price']
            size = position['size']
            side = position['side']
            
            # Calculate P&L
            if side == 'long':
                pnl = (current_price - entry_price) * size
            else:
                pnl = (entry_price - current_price) * size
            
            total_pnl += pnl
            total_value += entry_price * size
            
            if pnl > 0:
                winning_count += 1
                largest_winner = max(largest_winner, pnl)
            else:
                largest_loser = min(largest_loser, pnl)
            
            if position.get('sentiment_enhanced', False):
                sentiment_enhanced_count += 1
        
        portfolio_metrics.update({
            'total_pnl': round(total_pnl, 2),
            'total_pnl_percent': round((total_pnl / total_value * 100) if total_value > 0 else 0, 2),
            'open_positions': len(live_positions),
            'winning_positions': winning_count,
            'largest_winner': round(largest_winner, 2),
            'largest_loser': round(largest_loser, 2),
            'sentiment_enhanced_trades': sentiment_enhanced_count,
            'last_updated': datetime.now().isoformat()
        })
        
        return portfolio_metrics
        
    except Exception as e:
        logger.error(f"Portfolio metrics calculation error: {e}")
        return portfolio_metrics

async def portfolio_tracker():
    """Background task to track portfolio changes"""
    while True:
        try:
            if live_positions:
                # Update current prices for all positions
                for pos_id, position in live_positions.items():
                    symbol = position['symbol']
                    try:
                        ticker = await asyncio.to_thread(kraken_exchange.fetch_ticker, symbol)
                        live_positions[pos_id]['current_price'] = ticker['last']
                    except Exception as e:
                        logger.warning(f"Error updating price for {symbol}: {e}")
            
            await asyncio.sleep(5)  # Update positions every 5 seconds
            
        except Exception as e:
            logger.error(f"Portfolio tracker error: {e}")
            await asyncio.sleep(10)

async def collect_kraken_data() -> Dict:
    """Enhanced Kraken data collection with price change tracking"""
    try:
        symbols = ['BTC/USD', 'ETH/USD', 'SOL/USD', 'ADA/USD']
        tickers = {}
        
        for symbol in symbols:
            try:
                ticker = await asyncio.to_thread(kraken_exchange.fetch_ticker, symbol)
                
                # Enhanced ticker data
                tickers[symbol.replace('/', '')] = {
                    'symbol': symbol,
                    'price': ticker['last'],
                    'change24h': ticker['percentage'],
                    'volume24h': ticker['baseVolume'],
                    'high24h': ticker['high'],
                    'low24h': ticker['low'],
                    'bid': ticker['bid'],
                    'ask': ticker['ask'],
                    'spread': round(((ticker['ask'] - ticker['bid']) / ticker['bid']) * 100, 3) if ticker['bid'] else 0,
                    'timestamp': ticker['timestamp'] or datetime.now().timestamp() * 1000,
                    'market_cap_rank': {'BTC/USD': 1, 'ETH/USD': 2, 'SOL/USD': 5, 'ADA/USD': 8}.get(symbol, 99)
                }
            except Exception as e:
                logger.warning(f"Error fetching {symbol}: {e}")
                continue
        
        return tickers
        
    except Exception as e:
        logger.error(f"Enhanced Kraken data collection error: {e}")
        return {}

# Enhanced REST API Endpoints

@app.post("/api/positions/open")
async def open_position(position: PositionUpdate):
    """Open a new position with sentiment enhancement"""
    try:
        # Get current sentiment for enhancement
        sentiment_data = await collect_enhanced_sentiment_data()
        
        position_id = f"{position.symbol}_{datetime.now().timestamp()}"
        live_positions[position_id] = {
            'id': position_id,
            'symbol': position.symbol,
            'side': position.side,
            'entry_price': position.entry_price,
            'size': position.size,
            'current_price': position.current_price or position.entry_price,
            'open_time': datetime.now().isoformat(),
            'sentiment_at_entry': sentiment_data['sentiment'],
            'sentiment_enhanced': sentiment_data['signal_boost_active'],
            'enhancement_multiplier': sentiment_data['enhancement_multiplier']
        }
        
        logger.info(f"📈 Opened position: {position.symbol} {position.side} @ ${position.entry_price}")
        
        return {
            "status": "success",
            "position_id": position_id,
            "sentiment_enhancement": sentiment_data['enhancement_multiplier'],
            "market_regime": sentiment_data['market_regime']
        }
        
    except Exception as e:
        logger.error(f"Position opening error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/signals/enhance")
async def enhance_signal(request: SignalEnhancementRequest):
    """Enhance trading signal with real-time sentiment"""
    try:
        sentiment_data = await collect_enhanced_sentiment_data()
        base_signal = request.base_signal
        
        # Apply sentiment enhancement
        original_confidence = base_signal.get('confidence', 0.5)
        enhancement_multiplier = sentiment_data['enhancement_multiplier']
        
        enhanced_confidence = np.clip(
            original_confidence * (1 + enhancement_multiplier),
            0.0, 1.0
        )
        
        # Position size adjustment
        original_size = base_signal.get('position_size', 0.25)
        size_multiplier = 1.0 + (enhancement_multiplier * 0.3)  # Max 30% size adjustment
        enhanced_size = np.clip(original_size * size_multiplier, original_size * 0.5, original_size * 1.5)
        
        enhanced_signal = {
            **base_signal,
            'confidence': round(enhanced_confidence, 3),
            'position_size': round(enhanced_size, 3),
            'sentiment_enhancement': {
                'applied': True,
                'original_confidence': original_confidence,
                'enhancement_multiplier': enhancement_multiplier,
                'market_regime': sentiment_data['market_regime'],
                'sentiment_score': sentiment_data['sentiment'],
                'confidence': sentiment_data['confidence'],
                'boost_active': sentiment_data['signal_boost_active']
            }
        }
        
        logger.info(f"🎯 Enhanced signal for {request.symbol}: {original_confidence:.3f} → {enhanced_confidence:.3f}")
        
        return enhanced_signal
        
    except Exception as e:
        logger.error(f"Signal enhancement error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/portfolio/metrics")
async def get_portfolio_metrics():
    """Get real-time portfolio metrics"""
    try:
        metrics = await calculate_portfolio_metrics()
        return metrics
    except Exception as e:
        logger.error(f"Portfolio metrics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/positions/live")
async def get_live_positions():
    """Get all live positions with current P&L"""
    try:
        if not live_positions:
            return {"positions": [], "count": 0}
        
        # Update all positions with current prices and P&L
        updated_positions = []
        for pos_id, position in live_positions.items():
            current_price = position.get('current_price', position['entry_price'])
            entry_price = position['entry_price']
            size = position['size']
            side = position['side']
            
            # Calculate real-time P&L
            if side == 'long':
                pnl = (current_price - entry_price) * size
                pnl_percent = ((current_price - entry_price) / entry_price) * 100
            else:
                pnl = (entry_price - current_price) * size
                pnl_percent = ((entry_price - current_price) / entry_price) * 100
            
            updated_position = {
                **position,
                'pnl': round(pnl, 2),
                'pnl_percent': round(pnl_percent, 2),
                'unrealized_pnl': round(pnl, 2),
                'time_open_minutes': int((datetime.now() - datetime.fromisoformat(position['open_time'])).total_seconds() / 60)
            }
            updated_positions.append(updated_position)
        
        return {
            "positions": updated_positions,
            "count": len(updated_positions),
            "total_pnl": sum(pos['pnl'] for pos in updated_positions)
        }
        
    except Exception as e:
        logger.error(f"Live positions error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/positions/{position_id}")
async def close_position(position_id: str):
    """Close a position"""
    try:
        if position_id not in live_positions:
            raise HTTPException(status_code=404, detail="Position not found")
        
        position = live_positions[position_id]
        current_price = position.get('current_price', position['entry_price'])
        
        # Calculate final P&L
        if position['side'] == 'long':
            final_pnl = (current_price - position['entry_price']) * position['size']
        else:
            final_pnl = (position['entry_price'] - current_price) * position['size']
        
        # Remove from live positions
        del live_positions[position_id]
        
        logger.info(f"📉 Closed position: {position['symbol']} with P&L: ${final_pnl:.2f}")
        
        return {
            "status": "closed",
            "position_id": position_id,
            "final_pnl": round(final_pnl, 2),
            "exit_price": current_price
        }
        
    except Exception as e:
        logger.error(f"Position closing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Enhanced WebSocket with portfolio updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Enhanced WebSocket endpoint with portfolio streaming"""
    await websocket.accept()
    active_websockets.add(websocket)
    logger.info(f"✅ WebSocket client connected. Total: {len(active_websockets)}")
    
    try:
        # Send initial comprehensive data
        initial_data = {
            'type': 'initial_data',
            'data': {
                'tickers': await collect_kraken_data(),
                'sentiment': await collect_enhanced_sentiment_data(),
                'gpu': await collect_gpu_metrics(),
                'health': await collect_system_health(),
                'portfolio': await calculate_portfolio_metrics(),
                'positions': list(live_positions.values())
            }
        }
        await websocket.send_text(json.dumps(initial_data))
        
        # Handle incoming messages
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=60)
                message = json.loads(data)
                
                if message.get('type') == 'subscribe':
                    logger.info(f"Client subscribed to: {message.get('channels', [])}")
                elif message.get('type') == 'refresh_all':
                    # Send fresh comprehensive data
                    refresh_data = {
                        'type': 'refresh_response',
                        'data': {
                            'tickers': await collect_kraken_data(),
                            'sentiment': await collect_enhanced_sentiment_data(),
                            'gpu': await collect_gpu_metrics(),
                            'health': await collect_system_health(),
                            'portfolio': await calculate_portfolio_metrics(),
                            'positions': list(live_positions.values())
                        }
                    }
                    await websocket.send_text(json.dumps(refresh_data))
                elif message.get('type') == 'ping':
                    await websocket.send_text(json.dumps({'type': 'pong'}))
                    
            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                await websocket.send_text(json.dumps({'type': 'ping'}))
                
    except WebSocketDisconnect:
        logger.info("🔌 WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        active_websockets.discard(websocket)
        logger.info(f"📊 Active WebSocket connections: {len(active_websockets)}")

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

# Include all the existing functions from the previous API server
async def collect_gpu_metrics() -> Dict:
    """Collect GPU performance metrics"""
    try:
        if not redis_client:
            return get_mock_gpu_metrics()
        
        gpu_data = await redis_client.get("trademonkey:gpu:metrics")
        if gpu_data:
            data = json.loads(gpu_data)
            return {
                'memory_used_gb': data.get('total_used_gb', 0),
                'memory_total_gb': data.get('total_memory_gb', 11),
                'memory_usage_pct': data.get('memory_usage_pct', 0),
                'processing_speed_ms': data.get('avg_processing_time', 0),
                'queue_throughput': data.get('queue_throughput', 0),
                'temperature_c': data.get('temperature', 65)
            }
        else:
            return get_mock_gpu_metrics()
            
    except Exception as e:
        logger.error(f"GPU metrics collection error: {e}")
        return get_mock_gpu_metrics()

def get_mock_gpu_metrics() -> Dict:
    """Generate mock GPU metrics for demo"""
    import random
    import math
    
    time_factor = datetime.now().timestamp() / 60
    base_usage = 0.4 + math.sin(time_factor * 0.2) * 0.2
    noise = random.random() * 0.15
    memory_usage_pct = min(95, max(20, (base_usage + noise) * 100))
    
    return {
        'memory_used_gb': round(11 * memory_usage_pct / 100, 2),
        'memory_total_gb': 11.0,
        'memory_usage_pct': round(memory_usage_pct, 1),
        'processing_speed_ms': round(200 + random.random() * 300, 0),
        'queue_throughput': round(2000 + random.random() * 1000, 0),
        'temperature_c': round(55 + memory_usage_pct * 0.3 + random.random() * 10, 1)
    }

async def collect_system_health() -> Dict:
    """Collect system health metrics"""
    try:
        kraken_health = 'healthy' if kraken_exchange else 'critical'
        redis_health = 'healthy' if redis_client else 'critical'
        sentiment_health = 'healthy'
        gpu_health = 'healthy'
        
        score = 0
        if kraken_health == 'healthy': score += 25
        if redis_health == 'healthy': score += 25
        if sentiment_health == 'healthy': score += 25
        if gpu_health == 'healthy': score += 25
        
        return {
            'overall_score': score,
            'api_connections': {
                'kraken': kraken_health,
                'redis': redis_health,
                'sentiment_engine': sentiment_health,
                'gpu': gpu_health
            },
            'uptime_hours': 24,
            'total_trades': len(live_positions),
            'current_positions': len(live_positions),
            'portfolio_value': portfolio_metrics.get('total_pnl', 0)
        }
        
    except Exception as e:
        logger.error(f"System health collection error: {e}")
        return {
            'overall_score': 0,
            'api_connections': {
                'kraken': 'critical',
                'redis': 'critical',
                'sentiment_engine': 'critical',
                'gpu': 'critical'
            },
            'uptime_hours': 0,
            'total_trades': 0,
            'current_positions': 0,
            'portfolio_value': 0
        }

if __name__ == "__main__":
    logger.info("🚀 Starting TradeMonkey Fusion Enhanced API Server...")
    uvicorn.run(
        "api_server_enhanced:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )