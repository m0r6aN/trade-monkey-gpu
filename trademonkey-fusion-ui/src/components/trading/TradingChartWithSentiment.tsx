// File: src/components/trading/TradingChartWithSentiment.tsx
import React, { useEffect, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import { useTickers, useSentiment } from '@/hooks/useRealtimeData';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { TrendingUp, TrendingDown, Activity, Zap } from 'lucide-react';

interface CandleData {
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  sentiment?: number;
}

interface ChartProps {
  symbol?: string;
  timeframe?: string;
}

const TradingChartWithSentiment: React.FC<ChartProps> = ({ 
  symbol = 'BTC/USD', 
  timeframe = '1h' 
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [selectedSymbol, setSelectedSymbol] = useState(symbol);
  const [candleData, setCandleData] = useState<CandleData[]>([]);
  const [isGlitching, setIsGlitching] = useState(false);
  
  const tickers = useTickers();
  const sentiment = useSentiment();

  // Generate mock candle data (to be replaced with real OHLCV data)
  useEffect(() => {
    const generateCandles = () => {
      const candles: CandleData[] = [];
      let basePrice = 45000;
      
      for (let i = 0; i < 100; i++) {
        const timestamp = Date.now() - (100 - i) * 3600000; // 1 hour intervals
        const volatility = 0.02 + (Math.random() * 0.03);
        const change = (Math.random() - 0.5) * volatility;
        
        const open = basePrice;
        const close = basePrice * (1 + change);
        const high = Math.max(open, close) * (1 + Math.random() * 0.01);
        const low = Math.min(open, close) * (1 - Math.random() * 0.01);
        const volume = 100 + Math.random() * 500;
        
        candles.push({
          timestamp,
          open,
          high,
          low,
          close,
          volume,
          sentiment: (Math.random() - 0.5) * 2 // Random sentiment for each candle
        });
        
        basePrice = close;
      }
      
      setCandleData(candles);
    };

    generateCandles();
  }, [selectedSymbol]);

  // Cyberpunk glitch effect
  useEffect(() => {
    const glitchInterval = setInterval(() => {
      if (Math.random() > 0.98) {
        setIsGlitching(true);
        setTimeout(() => setIsGlitching(false), 100);
      }
    }, 2000);

    return () => clearInterval(glitchInterval);
  }, []);

  // Canvas drawing logic
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || candleData.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * devicePixelRatio;
    canvas.height = rect.height * devicePixelRatio;
    ctx.scale(devicePixelRatio, devicePixelRatio);

    const width = rect.width;
    const height = rect.height;
    const padding = 40;
    const chartWidth = width - padding * 2;
    const chartHeight = height - padding * 2;

    // Clear canvas
    ctx.fillStyle = '#111827';
    ctx.fillRect(0, 0, width, height);

    // Calculate price range
    const prices = candleData.flatMap(c => [c.high, c.low]);
    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);
    const priceRange = maxPrice - minPrice;

    // Draw grid
    ctx.strokeStyle = '#374151';
    ctx.lineWidth = 1;
    
    // Horizontal grid lines
    for (let i = 0; i <= 5; i++) {
      const y = padding + (chartHeight * i) / 5;
      ctx.beginPath();
      ctx.moveTo(padding, y);
      ctx.lineTo(width - padding, y);
      ctx.stroke();
    }

    // Vertical grid lines
    for (let i = 0; i <= 10; i++) {
      const x = padding + (chartWidth * i) / 10;
      ctx.beginPath();
      ctx.moveTo(x, padding);
      ctx.lineTo(x, height - padding);
      ctx.stroke();
    }

    // Draw candlesticks with sentiment overlay
    const candleWidth = chartWidth / candleData.length * 0.8;
    
    candleData.forEach((candle, index) => {
      const x = padding + (index * chartWidth) / candleData.length;
      
      // Calculate y positions
      const openY = padding + chartHeight - ((candle.open - minPrice) / priceRange) * chartHeight;
      const closeY = padding + chartHeight - ((candle.close - minPrice) / priceRange) * chartHeight;
      const highY = padding + chartHeight - ((candle.high - minPrice) / priceRange) * chartHeight;
      const lowY = padding + chartHeight - ((candle.low - minPrice) / priceRange) * chartHeight;

      // Determine candle color
      const isGreen = candle.close > candle.open;
      const baseColor = isGreen ? '#10b981' : '#ef4444';
      
      // Apply sentiment enhancement
      const sentimentBoost = Math.abs(candle.sentiment || 0);
      const alpha = 0.7 + (sentimentBoost * 0.3);
      
      // Draw wick
      ctx.strokeStyle = baseColor;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x + candleWidth / 2, highY);
      ctx.lineTo(x + candleWidth / 2, lowY);
      ctx.stroke();

      // Draw body
      ctx.fillStyle = baseColor + Math.floor(alpha * 255).toString(16).padStart(2, '0');
      const bodyTop = Math.min(openY, closeY);
      const bodyHeight = Math.abs(closeY - openY);
      ctx.fillRect(x, bodyTop, candleWidth, Math.max(bodyHeight, 1));

      // Draw sentiment indicator
      if (Math.abs(candle.sentiment || 0) > 0.5) {
        const sentimentColor = (candle.sentiment || 0) > 0 ? '#3b82f6' : '#f59e0b';
        ctx.fillStyle = sentimentColor + '80';
        ctx.fillRect(x, padding, candleWidth, 5);
      }
    });

    // Draw current price line
    const currentTicker = tickers?.[selectedSymbol.replace('/', '')] || tickers?.['BTCUSD'];
    if (currentTicker) {
      const currentPrice = currentTicker.price;
      const currentY = padding + chartHeight - ((currentPrice - minPrice) / priceRange) * chartHeight;
      
      ctx.strokeStyle = '#8b5cf6';
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      ctx.moveTo(padding, currentY);
      ctx.lineTo(width - padding, currentY);
      ctx.stroke();
      ctx.setLineDash([]);

      // Price label
      ctx.fillStyle = '#8b5cf6';
      ctx.font = '12px monospace';
      ctx.fillText(`$${currentPrice.toFixed(2)}`, width - padding + 5, currentY + 4);
    }

    // Draw sentiment heatmap overlay
    if (sentiment?.sentiment !== undefined) {
      const sentimentValue = sentiment.sentiment;
      const overlayAlpha = Math.abs(sentimentValue) * 0.1;
      const overlayColor = sentimentValue > 0 ? '#10b981' : '#ef4444';
      
      ctx.fillStyle = overlayColor + Math.floor(overlayAlpha * 255).toString(16).padStart(2, '0');
      ctx.fillRect(padding, padding, chartWidth, chartHeight);
    }

  }, [candleData, selectedSymbol, tickers, sentiment]);

  const availableSymbols = ['BTC/USD', 'ETH/USD', 'SOL/USD', 'ADA/USD'];
  
  const getCurrentPrice = () => {
    const ticker = tickers?.[selectedSymbol.replace('/', '')] || tickers?.['BTCUSD'];
    return ticker?.price || 0;
  };

  const getPriceChange = () => {
    const ticker = tickers?.[selectedSymbol.replace('/', '')] || tickers?.['BTCUSD'];
    return ticker?.change24h || 0;
  };

  const getSentimentColor = () => {
    if (!sentiment?.sentiment) return 'text-gray-400';
    return sentiment.sentiment > 0 ? 'text-green-400' : 'text-red-400';
  };

  return (
    <Card className="bg-gray-900 border-purple-500/30 quantum-glow">
      <CardHeader className="pb-2">
        <CardTitle className={`flex items-center justify-between text-white transition-colors ${isGlitching ? 'animate-pulse' : ''}`}>
          <div className="flex items-center gap-2">
            <Activity className="w-5 h-5 text-purple-400" />
            <span className="font-mono">TRADING_CHART_&_SENTIMENT</span>
            {sentiment?.signal_boost_active && (
              <Badge className="bg-purple-500/20 text-purple-400 animate-pulse">
                <Zap className="w-3 h-3 mr-1" />
                ENHANCED
              </Badge>
            )}
          </div>
          <div className="flex items-center gap-2">
            {availableSymbols.map((sym) => (
              <Button
                key={sym}
                size="sm"
                variant={selectedSymbol === sym ? "default" : "outline"}
                onClick={() => setSelectedSymbol(sym)}
                className={`font-mono text-xs ${
                  selectedSymbol === sym 
                    ? 'bg-purple-600 text-white' 
                    : 'border-gray-600 text-gray-300 hover:bg-gray-800'
                }`}
              >
                {sym}
              </Button>
            ))}
          </div>
        </CardTitle>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Price Header */}
        <div className="flex items-center justify-between">
          <div>
            <div className="text-3xl font-mono font-bold text-white">
              ${getCurrentPrice().toFixed(2)}
            </div>
            <div className={`flex items-center gap-1 text-sm font-mono ${
              getPriceChange() >= 0 ? 'text-green-400' : 'text-red-400'
            }`}>
              {getPriceChange() >= 0 ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
              {getPriceChange() >= 0 ? '+' : ''}{getPriceChange().toFixed(2)}% (24H)
            </div>
          </div>
          
          {sentiment && (
            <div className="text-right">
              <div className="text-sm text-gray-400 font-mono">SENTIMENT</div>
              <div className={`text-2xl font-mono font-bold ${getSentimentColor()}`}>
                {sentiment.sentiment >= 0 ? '+' : ''}{sentiment.sentiment.toFixed(3)}
              </div>
              <div className="text-xs text-gray-400 font-mono">
                {(sentiment.confidence * 100).toFixed(0)}% CONFIDENCE
              </div>
            </div>
          )}
        </div>

        {/* Chart Canvas */}
        <div className="relative h-96 bg-gray-800 rounded border border-gray-600">
          <canvas
            ref={canvasRef}
            className="w-full h-full"
            style={{ imageRendering: 'pixelated' }}
          />
          
          {/* Sentiment Enhancement Overlay */}
          {sentiment?.enhancement_multiplier && Math.abs(sentiment.enhancement_multiplier) > 0.1 && (
            <motion.div
              className="absolute top-2 right-2 bg-purple-900/90 p-2 rounded border border-purple-500/50"
              animate={{ 
                boxShadow: [
                  '0 0 10px rgba(139, 92, 246, 0.3)',
                  '0 0 20px rgba(139, 92, 246, 0.6)',
                  '0 0 10px rgba(139, 92, 246, 0.3)'
                ]
              }}
              transition={{ duration: 2, repeat: Infinity }}
            >
              <div className="text-purple-400 text-sm font-mono">
                SIGNAL BOOST: {(sentiment.enhancement_multiplier * 100).toFixed(0)}%
              </div>
            </motion.div>
          )}
        </div>

        {/* Chart Controls */}
        <div className="flex items-center justify-between">
          <div className="flex gap-2">
            {['1m', '5m', '15m', '1h', '4h', '1d'].map((tf) => (
              <Button
                key={tf}
                size="sm"
                variant={timeframe === tf ? "default" : "outline"}
                className={`font-mono text-xs ${
                  timeframe === tf 
                    ? 'bg-blue-600 text-white' 
                    : 'border-gray-600 text-gray-300 hover:bg-gray-800'
                }`}
              >
                {tf}
              </Button>
            ))}
          </div>
          
          <div className="flex items-center gap-2">
            <Badge className="bg-gray-800 text-gray-300 font-mono">
              VOL: {tickers?.[selectedSymbol.replace('/', '')]?.volume24h?.toFixed(0) || '0'}
            </Badge>
            <Badge className="bg-gray-800 text-gray-300 font-mono">
              SPREAD: {tickers?.[selectedSymbol.replace('/', '')]?.spread?.toFixed(3) || '0.000'}%
            </Badge>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default TradingChartWithSentiment;