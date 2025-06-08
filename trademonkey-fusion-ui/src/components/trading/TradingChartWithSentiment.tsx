// File: src/components/trading/TradingChartWithSentiment.tsx
import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { TrendingUp, Brain, Zap, Target, Activity } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { SentimentData, TickerData, ChartData, SentimentSignal } from '@/types/trading';

interface TradingChartProps {
  symbol?: string;
  timeframe?: string;
  demoMode?: boolean;
  tickers?: TickerData;
  sentiment?: SentimentData;
}

const TradingChartWithSentiment: React.FC<TradingChartProps> = ({ 
  symbol = 'BTC/USD', 
  timeframe = '1h',
  demoMode = false,
  tickers,
  sentiment: propSentiment
}) => {
  const [chartData, setChartData] = useState<ChartData[]>([]);
  const [currentSentiment, setCurrentSentiment] = useState<SentimentSignal>({
    confidence: propSentiment?.confidence || 0.85,
    sentiment: propSentiment?.sentiment || 0.67,  // Using 'sentiment' field now
    boost: propSentiment?.boost || 0.35,
    alignment: propSentiment?.alignment || 'strong_bullish'
  });
  const [activeSignals, setActiveSignals] = useState<any[]>([]);
  const [isGlitching, setIsGlitching] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Update sentiment when props change
  useEffect(() => {
    if (propSentiment) {
      setCurrentSentiment({
        confidence: propSentiment.confidence || 0.85,
        sentiment: propSentiment.sentiment || 0.67,  // Using 'sentiment' field now
        boost: propSentiment.boost || 0.35,
        alignment: propSentiment.alignment || 'strong_bullish'
      });
    }
  }, [propSentiment]);

  // Generate demo data with cyberpunk flair
  useEffect(() => {
    if (demoMode) {
      const generateDemoData = () => {
        const data: ChartData[] = [];
        let basePrice = tickers?.[symbol]?.price || 65000;
        
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
  }, [demoMode, symbol, tickers]);

  // Cyberpunk glitch effect trigger
  useEffect(() => {
    const glitchInterval = setInterval(() => {
      if (Math.random() > 0.95) { // 5% chance every 2 seconds
        setIsGlitching(true);
        setTimeout(() => setIsGlitching(false), 200);
      }
    }, 2000);

    return () => clearInterval(glitchInterval);
  }, []);

  // Enhanced chart drawing with cyberpunk effects
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !chartData.length) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;

    // Clear canvas with dark cyberpunk background
    ctx.fillStyle = 'rgba(10, 10, 20, 0.95)';
    ctx.fillRect(0, 0, width, height);

    // Calculate scales
    const prices = chartData.map(d => d.price);
    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);
    const priceRange = maxPrice - minPrice;

    // Draw cyberpunk grid
    ctx.strokeStyle = 'rgba(0, 255, 255, 0.1)';
    ctx.lineWidth = 0.5;
    
    // Vertical grid lines
    for (let i = 0; i <= 10; i++) {
      const x = (i / 10) * width;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    
    // Horizontal grid lines
    for (let i = 0; i <= 8; i++) {
      const y = (i / 8) * height;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    // Draw sentiment heatmap background with cyberpunk glow
    chartData.forEach((data, i) => {
      const x = (i / (chartData.length - 1)) * width;
      const hue = data.sentiment > 0 ? 180 : 340; // Cyan for bull, Pink for bear
      const saturation = Math.abs(data.sentiment) * 85;
      const lightness = 30 + Math.abs(data.sentiment) * 40;
      
      const gradient = ctx.createLinearGradient(x - 5, 0, x + 5, height);
      gradient.addColorStop(0, `hsla(${hue}, ${saturation}%, ${lightness}%, 0.4)`);
      gradient.addColorStop(1, `hsla(${hue}, ${saturation}%, ${lightness}%, 0.1)`);
      
      ctx.fillStyle = gradient;
      ctx.fillRect(x - 5, 0, 10, height);
    });

    // Draw holographic price line with glow effect
    ctx.shadowColor = '#00ffff';
    ctx.shadowBlur = 10;
    ctx.strokeStyle = '#00ffff';
    ctx.lineWidth = 3;
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
    
    // Reset shadow
    ctx.shadowBlur = 0;

    // Draw cyberpunk signal badges with neon glow
    chartData.forEach((data, i) => {
      data.signals.forEach(signal => {
        const x = (i / (chartData.length - 1)) * width;
        const y = height - ((signal.price - minPrice) / priceRange) * height;
        
        // Outer glow
        ctx.shadowColor = signal.type === 'buy' ? '#00ff00' : '#ff0080';
        ctx.shadowBlur = 15;
        
        // Signal circle with gradient
        const gradient = ctx.createRadialGradient(x, y, 0, x, y, 12);
        gradient.addColorStop(0, signal.type === 'buy' ? '#00ff88' : '#ff0080');
        gradient.addColorStop(1, signal.type === 'buy' ? '#004422' : '#440022');
        
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(x, y, 10, 0, 2 * Math.PI);
        ctx.fill();
        
        // Signal arrow with holographic effect
        ctx.shadowBlur = 5;
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 14px monospace';
        ctx.textAlign = 'center';
        ctx.fillText(signal.type === 'buy' ? '↗' : '↙', x, y + 5);
      });
    });
    
    // Reset shadow
    ctx.shadowBlur = 0;

  }, [chartData]);

  // Demo scenario updates with cyberpunk theme
  useEffect(() => {
    if (demoMode) {
      const scenarios = [
        { sentiment: 0.85, confidence: 0.92, boost: 0.35, alignment: 'QUANTUM_BULLISH' },
        { sentiment: -0.75, confidence: 0.88, boost: -0.20, alignment: 'MATRIX_BEARISH' },
        { sentiment: 0.45, confidence: 0.65, boost: 0.15, alignment: 'CYBER_NEUTRAL' }
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
    if (sentiment > 0.5) return 'text-cyber-blue';
    if (sentiment < -0.5) return 'text-cyber-pink';
    return 'text-cyber-green';
  };

  const getBoostBadgeColor = (boost: number) => {
    if (boost > 0.2) return 'bg-gradient-to-r from-cyber-blue to-cyber-green';
    if (boost < -0.1) return 'bg-gradient-to-r from-cyber-pink to-red-500';
    return 'bg-gradient-to-r from-purple-500 to-cyber-blue';
  };

  const currentPrice = tickers?.[symbol]?.price || (chartData.length ? chartData[chartData.length - 1].price : 65420);
  const change24h = tickers?.[symbol]?.change24h || 2.34;

  return (
    <Card className="cyber-card w-full h-96 relative overflow-hidden">
      <CardHeader className="pb-2">
        <CardTitle className={`flex items-center justify-between text-cyber-blue glow-text ${isGlitching ? 'cyber-glitch' : ''}`}>
          <div className="flex items-center gap-2">
            <Activity className="w-5 h-5 text-cyber-blue" />
            <span className="font-mono">{symbol}</span>
            <Badge variant="outline" className="cyber-button text-xs">
              {timeframe}
            </Badge>
          </div>
          <div className="flex items-center gap-2">
            <AnimatePresence>
              <motion.div
                key={currentSentiment.boost}
                initial={{ scale: 0, rotate: -180 }}
                animate={{ scale: 1, rotate: 0 }}
                exit={{ scale: 0, rotate: 180 }}
                transition={{ duration: 0.5, ease: "easeOut" }}
              >
                <Badge 
                  className={`${getBoostBadgeColor(currentSentiment.boost)} text-white font-mono text-xs px-2 py-1 glow-border`}
                >
                  <Brain className="w-3 h-3 mr-1" />
                  {currentSentiment.boost > 0 ? '+' : ''}{(currentSentiment.boost * 100).toFixed(0)}%
                </Badge>
              </motion.div>
            </AnimatePresence>
            <Badge className="cyber-button border-cyber-blue text-cyber-blue">
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
            className="w-full h-60 rounded-lg border border-cyber-blue/30 glow-border bg-gradient-to-br from-gray-900 to-black"
          />
          
          {/* Cyberpunk sentiment overlay */}
          <motion.div
            className="absolute top-3 left-3 holo-card p-3 backdrop-blur-md"
            animate={{ 
              boxShadow: currentSentiment.sentiment > 0 
                ? '0 0 25px rgba(0, 255, 255, 0.6)' 
                : '0 0 25px rgba(255, 0, 128, 0.6)' 
            }}
          >
            <div className="text-xs text-cyber-blue font-mono mb-1 glow-text">MARKET_TELEPATHY</div>
            <div className={`text-xl font-mono font-bold ${getSentimentColor(currentSentiment.sentiment)} glow-text`}>
              {currentSentiment.sentiment > 0 ? '+' : ''}{currentSentiment.sentiment.toFixed(3)}
            </div>
            <div className="text-xs text-cyber-green font-mono">{currentSentiment.alignment}</div>
          </motion.div>

          {/* Holographic price display */}
          <div className="absolute top-3 right-3 holo-card p-3 backdrop-blur-md">
            <div className="text-xs text-cyber-blue font-mono mb-1 glow-text">CURRENT_PRICE</div>
            <div className="text-xl font-mono font-bold text-cyber-blue glow-text">
              ${currentPrice.toFixed(2)}
            </div>
            <div className={`text-xs font-mono glow-text ${change24h >= 0 ? 'text-cyber-green' : 'text-cyber-pink'}`}>
              {change24h >= 0 ? '+' : ''}{change24h.toFixed(2)}%
            </div>
          </div>

          {/* Cyberpunk signal indicators */}
          <AnimatePresence>
            {activeSignals.map((signal, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, scale: 0, y: 20 }}
                animate={{ opacity: 1, scale: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0, y: -20 }}
                className="absolute bottom-3 left-3 holo-card border-cyber-green p-3"
              >
                <div className="flex items-center gap-2 text-cyber-green text-sm font-mono glow-text">
                  <Zap className="w-4 h-4" />
                  <span>SIGNAL: {signal.type.toUpperCase()}</span>
                  <span>BOOST: +{(signal.boost * 100).toFixed(0)}%</span>
                </div>
              </motion.div>
            ))}
          </AnimatePresence>

          {/* Holographic overlay effect */}
          <div className="absolute inset-0 pointer-events-none">
            <div className="w-full h-full bg-gradient-to-r from-transparent via-cyber-blue/5 to-transparent animate-pulse-glow" />
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default TradingChartWithSentiment;