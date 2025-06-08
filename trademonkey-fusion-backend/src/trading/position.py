import asyncio
from typing import List, Dict
from datetime import datetime
import ccxt.async_support as ccxt
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('TradeMonkeyPosition')

class PositionManager:
    """Manages trading positions for TradeMonkey Fusion"""
    
    def __init__(self):
        self.positions: List[Dict] = []
        self.kraken = ccxt.kraken({
            'apiKey': 'your_api_key',
            'secret': 'your_api_secret',
            'enableRateLimit': True
        })
        logger.info("🚀 PositionManager initialized")

    async def open_position(self, symbol: str, size: float, side: str) -> Dict:
        """Open a new trading position"""
        try:
            if side not in ["long", "short"]:
                raise ValueError("Invalid side: must be 'long' or 'short'")
            if size <= 0:
                raise ValueError("Size must be positive")

            ticker = await self.kraken.fetch_ticker(symbol)
            entry_price = ticker["last"]
            
            position = {
                "id": f"pos_{int(datetime.utcnow().timestamp())}",
                "symbol": symbol,
                "side": side,
                "size": size,
                "entry_price": entry_price,
                "current_price": entry_price,
                "pnl": 0.0,
                "stop_loss": entry_price * (0.98 if side == "long" else 1.02),
                "take_profit": entry_price * (1.05 if side == "long" else 0.95),
                "confidence": 0.85,
                "sentiment_boost": 0.0,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            self.positions.append(position)
            logger.info(f"📈 Opened {side} position on {symbol} at {entry_price}")
            return position
        
        except Exception as e:
            logger.error(f"Failed to open position: {e}")
            raise

    async def get_live_positions(self) -> List[Dict]:
        """Get all open positions with updated P&L"""
        try:
            updated_positions = []
            for pos in self.positions:
                ticker = await self.kraken.fetch_ticker(pos["symbol"])
                pos["current_price"] = ticker["last"]
                pos["pnl"] = (pos["current_price"] - pos["entry_price"]) * pos["size"] * (1 if pos["side"] == "long" else -1)
                updated_positions.append(pos)
            
            logger.info(f"📊 Retrieved {len(updated_positions)} live positions")
            return updated_positions
        
        except Exception as e:
            logger.error(f"Failed to get live positions: {e}")
            raise

    async def close_position(self, position_id: str) -> Dict:
        """Close a position by ID"""
        try:
            position = next((pos for pos in self.positions if pos["id"] == position_id), None)
            if not position:
                raise ValueError(f"Position {position_id} not found")
            
            self.positions.remove(position)
            logger.info(f"📉 Closed position {position_id} on {position['symbol']}")
            return {"status": "closed", "id": position_id}
        
        except Exception as e:
            logger.error(f"Failed to close position: {e}")
            raise

    async def update_position(self, position: Dict) -> None:
        """Update an existing position"""
        try:
            for i, pos in enumerate(self.positions):
                if pos["id"] == position["id"]:
                    self.positions[i] = position
                    logger.info(f"🔄 Updated position {position['id']} on {position['symbol']}")
                    return
            raise ValueError(f"Position {position['id']} not found")
        
        except Exception as e:
            logger.error(f"Failed to update position: {e}")
            raise

    async def get_portfolio_metrics(self) -> Dict:
        """Get portfolio-level metrics"""
        try:
            positions = await self.get_live_positions()
            total_pnl = sum(pos["pnl"] for pos in positions)
            metrics = {
                "total_pnl": total_pnl,
                "position_count": len(positions),
                "risk_level": min(100, len(positions) * 15)  # Mock risk calculation
            }
            logger.info(f"📊 Portfolio metrics: {metrics}")
            return metrics
        
        except Exception as e:
            logger.error(f"Failed to get portfolio metrics: {e}")
            raise