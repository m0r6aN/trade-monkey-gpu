// File: src/components/trading/TradingChartWithSentiment.tsx
import React, { useState, useEffect, useRef, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { useRealtimeData } from '@/hooks/useRealtimeData';
import { 
  formatPrice, 
  formatPercentage, 
  getSentimentColor, 
  getChangeColor,
  generateTimeLabels,
  generateMockPriceData,
  cyberCard,
  TIMEFRAMES 
} from '@/lib/utils';
import type { SentimentData, Ticker } from '@/types/trading';

interface TradingChartProps {
  symbol?: string;
  timeframe?: string;
  demoMode?: boolean;
  tickers?: Record<string, Ticker>;
  sentiment?: SentimentData;
}

interface ChartPoint {
  timestamp: number;
  price: number;
  sentiment: number;
  volume: number;
}

const TradingChartWithSentiment: React.FC<TradingChartProps> = ({
  symbol = 'BTC/USD',
  timeframe = '1h',
  demoMode = false,
  tickers,
  sentiment
}) => {
  const [selectedSymbol, setSelectedSymbol] = useState(symbol);
  const [selectedTimeframe, setSelectedTimeframe] = useState(timeframe);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [showSentimentOverlay, setShowSentimentOverlay] = useState(true);
  const [chartData, setChartData] = useState<ChartPoint[]>([]);
  const [priceHistory, setPriceHistory] = useState<number[]>([]);
  
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>();
  
  const { tickers: realtimeTickers, sentiment: realtimeSentiment } = useRealtimeData(
    'tickers',
    '/api/realtime',
    {
      symbol: selectedSymbol,
      updateInterval: 1000
    }
  );

  // Use provided data or fallback to realtime data
  const currentTickers = tickers || realtimeTickers;
  const currentSentiment = sentiment || realtimeSentiment;
  
  const currentTicker = useMemo(() => {
    const tickerKey = selectedSymbol.replace('/', '');
    return currentTickers?.[tickerKey];
  }, [currentTickers, selectedSymbol]);

  // Generate mock chart data for demo
  useEffect(() => {
    if (demoMode || !currentTicker) {
      const basePrice = { 'BTC/USD': 45000, 'ETH/USD': 3200, 'SOL/USD': 85, 'ADA/USD': 0.45 }[selectedSymbol] || 100;
      const prices = generateMockPriceData(basePrice, 50, 0.015);
      const newChartData: ChartPoint[] = prices.map((price, index) => ({
        timestamp: Date.now() - (50 - index) * 60000,
        price,
        sentiment: (Math.sin(index * 0.1) + Math.random() * 0.4 - 0.2) * 0.8,
        volume: Math.random() * 1000000
      }));
      setChartData(newChartData);
      setPriceHistory(prices);
    }
  }, [demoMode, selectedSymbol, currentTicker]);

  // Update chart data with real price movements
  useEffect(() => {
    if (currentTicker && !demoMode) {
      const newPoint: ChartPoint = {
        timestamp: Date.now(),
        price: currentTicker.price,
        sentiment: currentSentiment?.sentiment || 0,
        volume: currentTicker.volume24h || 0
      };
      
      setChartData(prev => {
        const updated = [...prev.slice(-49), newPoint];
        return updated;
      });
      
      setPriceHistory(prev => {
        const updated = [...prev.slice(-49), currentTicker.price];
        return updated;
      });
    }
  }, [currentTicker, currentSentiment, demoMode]);

  // Canvas drawing logic
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || chartData.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const draw = () => {
      const { width, height } = canvas.getBoundingClientRect();
      canvas.width = width * window.devicePixelRatio;
      canvas.height = height * window.devicePixelRatio;
      ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

      // Clear canvas
      ctx.fillStyle = '#111827';
      ctx.fillRect(0, 0, width, height);

      if (chartData.length < 2) return;

      const padding = 40;
      const chartWidth = width - padding * 2;
      const chartHeight = height - padding * 2;

      // Calculate price bounds
      const prices = chartData.map(d => d.price);
      const minPrice = Math.min(...prices);
      const maxPrice = Math.max(...prices);
      const priceRange = maxPrice - minPrice || 1;

      // Draw sentiment background heatmap
      if (showSentimentOverlay) {
        const gradient = ctx.createLinearGradient(0, padding, 0, height - padding);
        chartData.forEach((point, index) => {
          const position = index / (chartData.length - 1);
          const sentiment = point.sentiment;
          const alpha = Math.abs(sentiment) * 0.3;
          const color = sentiment > 0 
            ? `rgba(34, 197, 94, ${alpha})` 
            : `rgba(239, 68, 68, ${alpha})`;
          gradient.addColorStop(position, color);
        });
        
        ctx.fillStyle = gradient;
        ctx.fillRect(padding, padding, chartWidth, chartHeight);
      }

      // Draw price line
      ctx.beginPath();
      ctx.strokeStyle = '#3b82f6';
      ctx.lineWidth = 2;
      
      chartData.forEach((point, index) => {
        const x = padding + (index / (chartData.length - 1)) * chartWidth;
        const y = padding + (1 - (point.price - minPrice) / priceRange) * chartHeight;
        
        if (index === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });
      ctx.stroke();

      // Draw sentiment indicators
      if (showSentimentOverlay) {
        chartData.forEach((point, index) => {
          const x = padding + (index / (chartData.length - 1)) * chartWidth;
          const sentimentHeight = Math.abs(point.sentiment) * 30;
          const y = point.sentiment > 0 
            ? padding - sentimentHeight 
            : height - padding;
          
          ctx.fillStyle = point.sentiment > 0 
            ? 'rgba(34, 197, 94, 0.6)' 
            : 'rgba(239, 68, 68, 0.6)';
          ctx.fillRect(x - 1, y, 2, sentimentHeight);
        });
      }

      // Draw grid lines
      ctx.strokeStyle = 'rgba(75, 85, 99, 0.3)';
      ctx.lineWidth = 1;
      
      // Horizontal grid lines
      for (let i = 0; i <= 4; i++) {
        const y = padding + (i / 4) * chartHeight;
        ctx.beginPath();
        ctx.moveTo(padding, y);
        ctx.lineTo(width - padding, y);
        ctx.stroke();
        
        // Price labels
        const price = maxPrice - (i / 4) * priceRange;
        ctx.fillStyle = '#9ca3af';
        ctx.font = '12px monospace';
        ctx.textAlign = 'right';
        ctx.fillText(formatPrice(price), padding - 10, y + 4);
      }

      // Vertical grid lines
      const timeLabels = generateTimeLabels(selectedTimeframe, 5);
      for (let i = 0; i < 5; i++) {
        const x = padding + (i / 4) * chartWidth;
        ctx.beginPath();
        ctx.moveTo(x, padding);
        ctx.lineTo(x, height - padding);
        ctx.stroke();
        
        // Time labels
        ctx.fillStyle = '#9ca3af';
        ctx.font = '12px monospace';
        ctx.textAlign = 'center';
        ctx.fillText(timeLabels[i] || '', x, height - 10);
      }

      // Current price indicator
      if (currentTicker) {
        const currentY = padding + (1 - (currentTicker.price - minPrice) / priceRange) * chartHeight;
        
        // Price line
        ctx.strokeStyle = currentTicker.change24h >= 0 ? '#10b981' : '#ef4444';
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        ctx.moveTo(padding, currentY);
        ctx.lineTo(width - padding, currentY);
        ctx.stroke();
        ctx.setLineDash([]);

        // Price label
        ctx.fillStyle = currentTicker.change24h >= 0 ? '#10b981' : '#ef4444';
        ctx.font = 'bold 14px monospace';
        ctx.textAlign = 'right';
        ctx.fillText(formatPrice(currentTicker.price), width - padding - 10, currentY - 10);
      }

      // Draw crosshair on hover (simplified for demo)
      // This would be enhanced with proper mouse tracking
    };

    draw();
    animationRef.current = requestAnimationFrame(draw);

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [chartData, currentTicker, selectedTimeframe, showSentimentOverlay]);

  const handleSymbolChange = (newSymbol: string) => {
    setSelectedSymbol(newSymbol);
    setChartData([]);
    setPriceHistory([]);
  };

  const priceChange = currentTicker ? currentTicker.change24h || 0 : 0;
  const currentPrice = currentTicker ? currentTicker.price : (priceHistory[priceHistory.length - 1] || 0);

  return (
    <motion.div
      className={`${cyberCard} ${isFullscreen ? 'fixed inset-0 z-50' : ''}`}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <CardHeader className="pb-2">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div className="flex items-center gap-3">
            <CardTitle className="text-white text-lg sm:text-xl">
              📈 {selectedSymbol} Trading Chart
            </CardTitle>
            {currentSentiment && (
              <Badge 
                className={`${getSentimentColor(currentSentiment.sentiment)} border-current`}
                variant="outline"
              >
                Sentiment: {(currentSentiment.sentiment * 100).toFixed(1)}%
              </Badge>
            )}
          </div>
          
          <div className="flex flex-wrap items-center gap-2">
            <Select value={selectedSymbol} onValueChange={handleSymbolChange}>
              <SelectTrigger className="w-32 bg-gray-800 border-gray-600 text-white">
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="bg-gray-800 border-gray-600">
                <SelectItem value="BTC/USD">BTC/USD</SelectItem>
                <SelectItem value="ETH/USD">ETH/USD</SelectItem>
                <SelectItem value="SOL/USD">SOL/USD</SelectItem>
                <SelectItem value="ADA/USD">ADA/USD</SelectItem>
              </SelectContent>
            </Select>

            <Select value={selectedTimeframe} onValueChange={setSelectedTimeframe}>
              <SelectTrigger className="w-24 bg-gray-800 border-gray-600 text-white">
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="bg-gray-800 border-gray-600">
                {TIMEFRAMES.map(tf => (
                  <SelectItem key={tf.value} value={tf.value}>
                    {tf.value}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>

            <Button
              variant="outline"
              size="sm"
              onClick={() => setShowSentimentOverlay(!showSentimentOverlay)}
              className={`border-gray-600 ${showSentimentOverlay ? 'bg-blue-600' : 'bg-gray-800'} text-white`}
            >
              🧠
            </Button>

            <Button
              variant="outline"
              size="sm"
              onClick={() => setIsFullscreen(!isFullscreen)}
              className="border-gray-600 bg-gray-800 text-white"
            >
              {isFullscreen ? '📥' : '📤'}
            </Button>
          </div>
        </div>
        
        {/* Price Display */}
        <div className="flex flex-wrap items-center gap-4 mt-2">
          <div className="text-2xl sm:text-3xl font-mono text-white">
            {formatPrice(currentPrice)}
          </div>
          <div className={`text-lg font-medium ${getChangeColor(priceChange)}`}>
            {formatPercentage(priceChange)}
          </div>
          {currentTicker && (
            <>
              <div className="text-sm text-gray-400">
                H: {formatPrice(currentTicker.high24h)}
              </div>
              <div className="text-sm text-gray-400">
                L: {formatPrice(currentTicker.low24h)}
              </div>
              <div className="text-sm text-gray-400">
                Vol: {currentTicker.volume24h ? (currentTicker.volume24h / 1000000).toFixed(1) + 'M' : 'N/A'}
              </div>
            </>
          )}
        </div>
      </CardHeader>

      <CardContent className="p-0">
        <div className="relative">
          <canvas
            ref={canvasRef}
            className="w-full h-64 sm:h-80 lg:h-96 cursor-crosshair"
            style={{ display: 'block' }}
          />
          
          {/* Sentiment Legend */}
          <AnimatePresence>
            {showSentimentOverlay && (
              <motion.div
                className="absolute top-4 right-4 bg-black/70 rounded p-2 text-xs text-white"
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.8 }}
              >
                <div className="flex items-center gap-2 mb-1">
                  <div className="w-3 h-3 bg-green-500/60 rounded"></div>
                  <span>Bullish Sentiment</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 bg-red-500/60 rounded"></div>
                  <span>Bearish Sentiment</span>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Loading Overlay */}
          {!currentTicker && !demoMode && (
            <div className="absolute inset-0 flex items-center justify-center bg-gray-900/80 backdrop-blur-sm">
              <motion.div
                className="text-white text-center"
                animate={{ opacity: [0.5, 1, 0.5] }}
                transition={{ duration: 1.5, repeat: Infinity }}
              >
                <div className="text-2xl mb-2">📡</div>
                <div>Loading market data...</div>
              </motion.div>
            </div>
          )}

          {/* Demo Mode Indicator */}
          {demoMode && (
            <div className="absolute top-4 left-4 bg-purple-600/80 rounded px-2 py-1 text-xs text-white font-semibold">
              🎭 DEMO MODE
            </div>
          )}
        </div>

        {/* Chart Controls */}
        <div className="p-4 border-t border-gray-700">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-center gap-2">
              <span className="text-sm text-gray-400">Chart Style:</span>
              <div className="flex rounded border border-gray-600 overflow-hidden">
                <button className="px-3 py-1 bg-blue-600 text-white text-xs">Line</button>
                <button className="px-3 py-1 bg-gray-700 text-gray-300 text-xs hover:bg-gray-600">Candle</button>
                <button className="px-3 py-1 bg-gray-700 text-gray-300 text-xs hover:bg-gray-600">Area</button>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <span className="text-sm text-gray-400">Indicators:</span>
              <div className="flex gap-1">
                <Button
                  variant="outline"
                  size="sm"
                  className="border-gray-600 bg-gray-800 text-white text-xs px-2 py-1"
                >
                  RSI
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  className="border-gray-600 bg-gray-800 text-white text-xs px-2 py-1"
                >
                  MACD
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  className="border-gray-600 bg-gray-800 text-white text-xs px-2 py-1"
                >
                  BB
                </Button>
              </div>
            </div>
          </div>
        </div>
      </CardContent>
    </motion.div>
  );
};

export default TradingChartWithSentiment;
