// File: src/components/trading/SentimentTelepathyWidget.tsx
import React, { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useSentiment } from '@/hooks/useRealtimeData';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';

const SentimentTelepathyWidget: React.FC = () => {
  const sentiment = useSentiment();
  const [particles, setParticles] = useState<Array<{ id: number; x: number; y: number; emoji: string }>>([]);

  // Generate particles based on sentiment
  useEffect(() => {
    if (!sentiment?.sentiment) return;

    const sentimentScore = sentiment.sentiment;
    const confidence = sentiment.confidence || 0;
    
    if (Math.abs(sentimentScore) > 0.3 && confidence > 0.6) {
      const emoji = sentimentScore > 0 ? '🚀' : sentimentScore < -0.5 ? '💀' : '⚠️';
      const newParticles = Array.from({ length: Math.floor(confidence * 5) }, (_, i) => ({
        id: Date.now() + i,
        x: Math.random() * 300,
        y: Math.random() * 200,
        emoji
      }));
      
      setParticles(prev => [...prev, ...newParticles].slice(-20)); // Keep max 20 particles
      
      // Remove particles after animation
      setTimeout(() => {
        setParticles(prev => prev.filter(p => !newParticles.find(np => np.id === p.id)));
      }, 2000);
    }
  }, [sentiment?.sentiment, sentiment?.confidence]);

  const getSentimentColor = (score: number): string => {
    if (score > 0.5) return 'text-green-400';
    if (score > 0.2) return 'text-green-300';
    if (score > -0.2) return 'text-yellow-400';
    if (score > -0.5) return 'text-red-300';
    return 'text-red-400';
  };

  const getSentimentBgColor = (score: number): string => {
    if (score > 0.5) return 'bg-green-500/20 border-green-500/50';
    if (score > 0.2) return 'bg-green-500/10 border-green-500/30';
    if (score > -0.2) return 'bg-yellow-500/20 border-yellow-500/50';
    if (score > -0.5) return 'bg-red-500/10 border-red-500/30';
    return 'bg-red-500/20 border-red-500/50';
  };

  const getMarketRegimeInfo = (regime: string) => {
    const regimeMap: Record<string, { label: string; emoji: string; description: string }> = {
      'bull_euphoria': { label: 'Bull Euphoria', emoji: '🐂🚀', description: 'Extreme bullish sentiment' },
      'bull_optimism': { label: 'Bull Optimism', emoji: '🐂📈', description: 'Positive market outlook' },
      'neutral_mixed': { label: 'Neutral Mixed', emoji: '🦀⚖️', description: 'Sideways market sentiment' },
      'bear_pessimism': { label: 'Bear Pessimism', emoji: '🐻📉', description: 'Negative market outlook' },
      'bear_panic': { label: 'Bear Panic', emoji: '🐻💀', description: 'Extreme bearish sentiment' },
    };
    return regimeMap[regime] || { label: 'Unknown', emoji: '❓', description: 'Sentiment regime unknown' };
  };

  if (!sentiment) {
    return (
      <Card className="bg-gray-900 border-purple-500/30">
        <CardHeader>
          <CardTitle className="text-white">🧠 Sentiment Telepathy</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center text-gray-400">Loading sentiment data...</div>
        </CardContent>
      </Card>
    );
  }

  const regimeInfo = getMarketRegimeInfo(sentiment.market_regime || 'unknown');
  const sentimentScore = sentiment.sentiment || 0;
  const confidence = sentiment.confidence || 0;

  return (
    <Card className="bg-gray-900 border-purple-500/30 relative overflow-hidden">
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center justify-between text-white">
          <div className="flex items-center gap-2">
            🧠 Sentiment Telepathy
            {sentiment.signal_boost_active && (
              <Badge className="bg-purple-500/20 text-purple-400 animate-pulse">
                🚀 BOOST ACTIVE
              </Badge>
            )}
          </div>
          <div className="text-sm font-mono text-gray-400">
            {sentiment.sample_count || 0} samples
          </div>
        </CardTitle>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Main Sentiment Display */}
        <div className="flex items-center justify-center mb-6">
          <motion.div
            className={`text-6xl font-mono ${getSentimentColor(sentimentScore)}`}
            animate={{ scale: [1, 1.1, 1] }}
            transition={{ duration: 2, repeat: Infinity }}
          >
            {sentimentScore >= 0 ? '+' : ''}{sentimentScore.toFixed(3)}
          </motion.div>
        </div>

        {/* Market Regime */}
        <div className={`p-3 rounded border ${getSentimentBgColor(sentimentScore)}`}>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="text-xl">{regimeInfo.emoji}</span>
              <div>
                <div className="text-white font-semibold">{regimeInfo.label}</div>
                <div className="text-gray-400 text-sm">{regimeInfo.description}</div>
              </div>
            </div>
            <div className="text-right">
              <div className="text-white font-mono">{(confidence * 100).toFixed(1)}%</div>
              <div className="text-gray-400 text-xs">Confidence</div>
            </div>
          </div>
        </div>

        {/* Enhancement Metrics */}
        {sentiment.enhancement_multiplier && Math.abs(sentiment.enhancement_multiplier) > 0 && (
          <div className="bg-purple-900/30 p-3 rounded border border-purple-500/30">
            <div className="flex items-center justify-between mb-2">
              <span className="text-purple-400 font-semibold">Signal Enhancement</span>
              <span className="text-white font-mono">
                {sentiment.enhancement_multiplier >= 0 ? '+' : ''}{(sentiment.enhancement_multiplier * 100).toFixed(1)}%
              </span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-2">
              <motion.div
                className="bg-purple-500 h-2 rounded-full"
                initial={{ width: 0 }}
                animate={{ width: `${Math.abs(sentiment.enhancement_multiplier) * 100}%` }}
                transition={{ duration: 0.5 }}
              />
            </div>
          </div>
        )}

        {/* Source Breakdown */}
        <div className="grid grid-cols-2 gap-2">
          {sentiment.sources && Object.entries(sentiment.sources).map(([source, count]) => (
            <div key={source} className="bg-gray-800 p-2 rounded text-center">
              <div className="text-white font-mono">{count}</div>
              <div className="text-gray-400 text-xs capitalize">{source}</div>
            </div>
          ))}
        </div>

        {/* Crypto Ratio */}
        <div className="bg-gray-800 p-3 rounded">
          <div className="flex items-center justify-between mb-2">
            <span className="text-gray-400">Crypto Focus</span>
            <span className="text-blue-400 font-mono">
              {((sentiment.crypto_ratio || 0) * 100).toFixed(1)}%
            </span>
          </div>
          <div className="w-full bg-gray-700 rounded-full h-2">
            <motion.div
              className="bg-blue-500 h-2 rounded-full"
              initial={{ width: 0 }}
              animate={{ width: `${(sentiment.crypto_ratio || 0) * 100}%` }}
              transition={{ duration: 0.5 }}
            />
          </div>
        </div>

        {/* Floating Particles */}
        <div className="absolute inset-0 pointer-events-none overflow-hidden">
          <AnimatePresence>
            {particles.map((particle) => (
              <motion.div
                key={particle.id}
                className="absolute text-2xl"
                initial={{ 
                  x: particle.x, 
                  y: particle.y, 
                  opacity: 1, 
                  scale: 0 
                }}
                animate={{ 
                  y: particle.y - 100, 
                  opacity: 0, 
                  scale: 1,
                  rotate: 360 
                }}
                exit={{ opacity: 0 }}
                transition={{ duration: 2, ease: 'easeOut' }}
                style={{ left: particle.x, top: particle.y }}
              >
                {particle.emoji}
              </motion.div>
            ))}
          </AnimatePresence>
        </div>
      </CardContent>
    </Card>
  );
};

export default SentimentTelepathyWidget;