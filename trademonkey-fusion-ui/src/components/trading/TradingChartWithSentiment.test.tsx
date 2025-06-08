import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { TrendingUp, TrendingDown, Brain, Zap, Target } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';

interface SentimentSignal {
  confidence: number;
  sentiment: number;
  boost: number;
  alignment: string;
}

interface ChartData {
  timestamp: number;
  price: number;
  volume: number;
  sentiment: number;
  signals: Array<{
    type: 'buy' | 'sell';
    price: number;
    confidence: number;
    boost: number;
  }>;
}

interface TradingChartProps {
  symbol?: string;
  timeframe?: string;
  demoMode?: boolean;
}

const TradingChartWithSentiment: React.FC<TradingChartProps> = ({ 
  symbol = 'BTC/USD', 
  timeframe = '1h',
  demoMode = false 
}) => {
  const [chartData, setChartData] = useState<ChartData[]>([]);
  const [currentSentiment, setCurrentSentiment] = useState<SentimentSignal>({
    confidence: 0.85,
    sentiment: 0.67,
    boost: 0.35,
    alignment: 'strong_bullish'
  });
  const [activeSignals, setActiveSignals] = useState<any[]>([]);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Generate demo data
  useEffect(() => {
    if (demoMode) {
      const generateDemoData = () => {
        const data: ChartData[] = [];
        let basePrice = 65000;
        
        for (let i = 0; i < 50; i++) {
          const sentiment = Math.sin(i * 0.3) * 0.4 + Math.random() * 0.4;
          const priceChange = sentiment * 1000 + (Math.random() - 0.5) * 500;
          basePrice += priceChange;
          
          data.push({
            timestamp: Date.now() - (50 - i) * 3600000,
            price: basePrice,
            volume: Math.random() * 100000 + 50000,
            sentiment: sentiment,
            signals: Math.random() > 0.8 ? [{
              type: sentiment > 0 ? 'buy' : 'sell',
              price: basePrice,
              confidence: 0.7 + Math.random() * 0.3,
              boost: sentiment * 0.5
            }] : []
          });
        }
        setChartData(data);
      };

      generateDemoData();
      const interval = setInterval(generateDemoData, 5000);
      return () => clearInterval(interval);
    }
  }, [demoMode]);

  // Draw chart
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !chartData.length) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Calculate scales
    const prices = chartData.map(d => d.price);
    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);
    const priceRange = maxPrice - minPrice;

    // Draw sentiment heatmap background
    chartData.forEach((data, i) => {
      const x = (i / (chartData.length - 1)) * width;
      const hue = data.sentiment > 0 ? 142 : 0; // Green for bull, red for bear
      const saturation = Math.abs(data.sentiment) * 85;
      const lightness = 50 + Math.abs(data.sentiment) * 30;
      
      ctx.fillStyle = `hsla(${hue}, ${saturation}%, ${lightness}%, 0.3)`;
      ctx.fillRect(x - 5, 0, 10, height);
    });

    // Draw price line
    ctx.strokeStyle = '#00d4ff';
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    chartData.forEach((data, i) => {
      const x = (i / (chartData.length - 1)) * width;
      const y = height - ((data.price - minPrice) / priceRange) * height;
      
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();

    // Draw signal badges
    chartData.forEach((data, i) => {
      data.signals.forEach(signal => {
        const x = (i / (chartData.length - 1)) * width;
        const y = height - ((signal.price - minPrice) / priceRange) * height;
        
        // Signal circle
        ctx.fillStyle = signal.type === 'buy' ? '#10b981' : '#ef4444';
        ctx.beginPath();
        ctx.arc(x, y, 8, 0, 2 * Math.PI);
        ctx.fill();
        
        // Signal arrow
        ctx.fillStyle = '#ffffff';
        ctx.font = '12px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(signal.type === 'buy' ? '↑' : '↓', x, y + 4);
      });
    });

  }, [chartData]);

  // Demo scenario updates
  useEffect(() => {
    if (demoMode) {
      const scenarios = [
        { sentiment: 0.85, confidence: 0.92, boost: 0.35, alignment: 'strong_bullish' },
        { sentiment: -0.75, confidence: 0.88, boost: -0.20, alignment: 'strong_bearish' },
        { sentiment: 0.45, confidence: 0.65, boost: 0.15, alignment: 'moderate_bullish' }
      ];
      
      let scenarioIndex = 0;
      const interval = setInterval(() => {
        setCurrentSentiment(scenarios[scenarioIndex]);
        scenarioIndex = (scenarioIndex + 1) % scenarios.length;
      }, 15000);
      
      return () => clearInterval(interval);
    }
  }, [demoMode]);

  const getSentimentColor = (sentiment: number) => {
    if (sentiment > 0.5) return 'text-green-400';
    if (sentiment < -0.5) return 'text-red-400';
    return 'text-yellow-400';
  };

  const getBoostBadgeColor = (boost: number) => {
    if (boost > 0.2) return 'bg-green-500';
    if (boost < -0.1) return 'bg-red-500';
    return 'bg-blue-500';
  };

  return (
    <Card className="w-full h-96 bg-gray-900 border-blue-500/30">
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center justify-between text-white">
          <div className="flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-blue-400" />
            {symbol} - {timeframe}
          </div>
          <div className="flex items-center gap-2">
            <AnimatePresence>
              <motion.div
                key={currentSentiment.boost}
                initial={{ scale: 0, rotate: -180 }}
                animate={{ scale: 1, rotate: 0 }}
                exit={{ scale: 0, rotate: 180 }}
                transition={{ duration: 0.5 }}
              >
                <Badge 
                  className={`${getBoostBadgeColor(currentSentiment.boost)} text-white font-mono`}
                >
                  <Brain className="w-3 h-3 mr-1" />
                  {currentSentiment.boost > 0 ? '+' : ''}{(currentSentiment.boost * 100).toFixed(0)}%
                </Badge>
              </motion.div>
            </AnimatePresence>
            <Badge variant="outline" className="text-blue-400 border-blue-400">
              <Target className="w-3 h-3 mr-1" />
              {(currentSentiment.confidence * 100).toFixed(0)}%
            </Badge>
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent className="p-4">
        <div className="relative">
          <canvas
            ref={canvasRef}
            width={800}
            height={240}
            className="w-full h-60 border border-gray-700 rounded"
          />
          
          {/* Sentiment overlay */}
          <motion.div
            className="absolute top-2 left-2 bg-black/70 rounded p-2 backdrop-blur-sm"
            animate={{ 
              boxShadow: currentSentiment.sentiment > 0 
                ? '0 0 20px rgba(16, 185, 129, 0.5)' 
                : '0 0 20px rgba(239, 68, 68, 0.5)' 
            }}
          >
            <div className="text-xs text-gray-300 mb-1">Market Sentiment</div>
            <div className={`text-lg font-mono ${getSentimentColor(currentSentiment.sentiment)}`}>
              {currentSentiment.sentiment > 0 ? '+' : ''}{currentSentiment.sentiment.toFixed(3)}
            </div>
            <div className="text-xs text-gray-400">{currentSentiment.alignment.replace('_', ' ')}</div>
          </motion.div>

          {/* Price display */}
          <div className="absolute top-2 right-2 bg-black/70 rounded p-2 backdrop-blur-sm">
            <div className="text-xs text-gray-300 mb-1">Current Price</div>
            <div className="text-lg font-mono text-blue-400">
              ${chartData.length ? chartData[chartData.length - 1].price.toFixed(2) : '65,420.00'}
            </div>
            <div className="text-xs text-green-400">+2.34%</div>
          </div>

          {/* Signal indicators */}
          <AnimatePresence>
            {activeSignals.map((signal, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, scale: 0 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0 }}
                className="absolute bottom-2 left-2 bg-yellow-500/20 border border-yellow-500 rounded p-2"
              >
                <div className="flex items-center gap-1 text-yellow-400 text-sm">
                  <Zap className="w-4 h-4" />
                  Signal: {signal.type} | Boost: +{(signal.boost * 100).toFixed(0)}%
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
        </div>
      </CardContent>
    </Card>
  );
};

export default TradingChartWithSentiment;