// File: src/components/trading/SentimentTelepathyWidget.tsx
import React, { useState, useEffect, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { useRealtimeData } from '@/hooks/useRealtimeData';
import { useIsMobile } from '@/hooks/useMobile';
import { 
  formatPercentage, 
  getSentimentColor, 
  cyberCard,
  fadeInScale,
  quantumGlow 
} from '@/lib/utils';
import type { SentimentData } from '@/types/trading';

interface SentimentTelepathyWidgetProps {
  sentiment?: SentimentData;
  demoMode?: boolean;
}

interface SlangTerm {
  term: string;
  count: number;
  trend: 'up' | 'down' | 'stable';
  impact: 'high' | 'medium' | 'low';
}

const SentimentTelepathyWidget: React.FC<SentimentTelepathyWidgetProps> = ({
  sentiment,
  demoMode = false
}) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [showDetails, setShowDetails] = useState(false);
  const [particles, setParticles] = useState<Array<{ id: number; x: number; y: number; emoji: string }>>([]);
  const isMobile = useIsMobile();
  
  const { sentiment: realtimeSentiment } = useRealtimeData('sentiment', '/api/sentiment/current', {
    updateInterval: 2000
  });

  // Use provided sentiment or fallback to realtime data
  const currentSentiment = sentiment || realtimeSentiment;

  // Generate sentiment particles based on current sentiment
  useEffect(() => {
    if (!currentSentiment) return;

    const sentimentScore = currentSentiment.sentiment;
    const shouldGenerateParticles = Math.abs(sentimentScore) > 0.3;

    if (shouldGenerateParticles) {
      const newParticles = Array.from({ length: 5 }, (_, i) => ({
        id: Date.now() + i,
        x: Math.random() * 100,
        y: Math.random() * 100,
        emoji: sentimentScore > 0 ? ['🚀', '💎', '🌙', '🔥', '💰'][i % 5] : ['💀', '📉', '😱', '🔴', '⚡'][i % 5]
      }));
      
      setParticles(newParticles);
      
      setTimeout(() => setParticles([]), 3000);
    }
  }, [currentSentiment?.sentiment]);

  // Generate mock slang terms
  const slangTerms: SlangTerm[] = useMemo(() => {
    if (!currentSentiment) return [];

    const bullishTerms = [
      { term: 'WAGMI', count: 45, trend: 'up' as const, impact: 'high' as const },
      { term: 'Diamond Hands', count: 32, trend: 'up' as const, impact: 'medium' as const },
      { term: 'To The Moon', count: 28, trend: 'stable' as const, impact: 'high' as const },
      { term: 'HODL', count: 22, trend: 'up' as const, impact: 'medium' as const },
      { term: 'Number Go Up', count: 18, trend: 'up' as const, impact: 'low' as const }
    ];

    const bearishTerms = [
      { term: 'NGMI', count: 38, trend: 'up' as const, impact: 'high' as const },
      { term: 'Paper Hands', count: 25, trend: 'down' as const, impact: 'medium' as const },
      { term: 'Rugpull', count: 20, trend: 'stable' as const, impact: 'high' as const },
      { term: 'Rekt', count: 15, trend: 'up' as const, impact: 'medium' as const },
      { term: 'Dump Incoming', count: 12, trend: 'up' as const, impact: 'low' as const }
    ];

    const neutralTerms = [
      { term: 'Probably Nothing', count: 30, trend: 'stable' as const, impact: 'medium' as const },
      { term: 'DYOR', count: 22, trend: 'up' as const, impact: 'low' as const },
      { term: 'Sideways Action', count: 18, trend: 'down' as const, impact: 'low' as const },
      { term: 'Crab Market', count: 15, trend: 'stable' as const, impact: 'medium' as const }
    ];

    const sentimentScore = currentSentiment.sentiment;
    if (sentimentScore > 0.3) return bullishTerms;
    if (sentimentScore < -0.3) return bearishTerms;
    return neutralTerms;
  }, [currentSentiment]);

  const getSentimentLabel = (score: number): string => {
    if (score > 0.7) return 'EUPHORIC';
    if (score > 0.4) return 'BULLISH';
    if (score > 0.1) return 'OPTIMISTIC';
    if (score > -0.1) return 'NEUTRAL';
    if (score > -0.4) return 'PESSIMISTIC';
    if (score > -0.7) return 'BEARISH';
    return 'PANIC';
  };

  const getSentimentEmoji = (score: number): string => {
    if (score > 0.5) return '🚀';
    if (score > 0.2) return '📈';
    if (score > -0.2) return '😐';
    if (score > -0.5) return '📉';
    return '💀';
  };

  const getRegimeColor = (regime: string): string => {
    switch (regime) {
      case 'bull_euphoria': return 'text-green-400 border-green-500';
      case 'bull_optimism': return 'text-green-300 border-green-400';
      case 'neutral_mixed': return 'text-yellow-400 border-yellow-500';
      case 'bear_pessimism': return 'text-red-300 border-red-400';
      case 'bear_panic': return 'text-red-400 border-red-500';
      default: return 'text-gray-400 border-gray-500';
    }
  };

  const getTrendIcon = (trend: string): string => {
    switch (trend) {
      case 'up': return '📈';
      case 'down': return '📉';
      default: return '➡️';
    }
  };

  const getImpactBadgeColor = (impact: string): string => {
    switch (impact) {
      case 'high': return 'bg-red-600';
      case 'medium': return 'bg-yellow-600';
      case 'low': return 'bg-blue-600';
      default: return 'bg-gray-600';
    }
  };

  if (!currentSentiment) {
    return (
      <Card className={cyberCard}>
        <CardContent className="p-6">
          <motion.div
            className="text-center text-gray-400"
            animate={{ opacity: [0.5, 1, 0.5] }}
            transition={{ duration: 1.5, repeat: Infinity }}
          >
            <div className="text-3xl mb-2">🧠</div>
            <div>Connecting to sentiment network...</div>
          </motion.div>
        </CardContent>
      </Card>
    );
  }

  const sentimentScore = currentSentiment.sentiment;
  const confidence = currentSentiment.confidence;
  const sentimentLabel = getSentimentLabel(sentimentScore);
  const sentimentEmoji = getSentimentEmoji(sentimentScore);

  return (
    <motion.div
      className={`${cyberCard} ${Math.abs(sentimentScore) > 0.5 ? quantumGlow : ''} relative overflow-hidden`}
      {...fadeInScale}
    >
      {/* Particle Effects */}
      <AnimatePresence>
        {particles.map((particle) => (
          <motion.div
            key={particle.id}
            className="absolute pointer-events-none text-2xl z-10"
            initial={{ x: `${particle.x}%`, y: `${particle.y}%`, opacity: 1, scale: 0 }}
            animate={{ 
              y: `${particle.y - 50}%`, 
              opacity: 0, 
              scale: 1,
              rotate: [0, 360]
            }}
            exit={{ opacity: 0, scale: 0 }}
            transition={{ duration: 3, ease: 'easeOut' }}
          >
            {particle.emoji}
          </motion.div>
        ))}
      </AnimatePresence>

      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-white flex items-center gap-2">
            🧠 Market Telepathy
            {currentSentiment.signal_boost_active && (
              <Badge className="bg-purple-600 text-white animate-pulse">
                BOOST
              </Badge>
            )}
          </CardTitle>
          {!isMobile && (
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setIsExpanded(!isExpanded)}
              className="text-gray-400 hover:text-white"
            >
              {isExpanded ? '📉' : '📊'}
            </Button>
          )}
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Main Sentiment Display */}
        <div className="text-center space-y-2">
          <motion.div
            className="text-6xl"
            animate={{ 
              scale: [1, 1.1, 1],
              rotate: Math.abs(sentimentScore) > 0.7 ? [0, 5, -5, 0] : 0
            }}
            transition={{ duration: 2, repeat: Infinity }}
          >
            {sentimentEmoji}
          </motion.div>
          
          <div className={`text-xl font-bold ${getSentimentColor(sentimentScore)}`}>
            {sentimentLabel}
          </div>
          
          <div className="text-3xl font-mono text-white">
            {formatPercentage(sentimentScore * 100, 1)}
          </div>
          
          <div className="text-sm text-gray-400">
            Confidence: {formatPercentage(confidence * 100, 0)}
          </div>
        </div>

        {/* Sentiment Bar */}
        <div className="space-y-2">
          <div className="flex justify-between text-xs text-gray-400">
            <span>😱 PANIC</span>
            <span>😐 NEUTRAL</span>
            <span>🚀 EUPHORIA</span>
          </div>
          
          <div className="relative h-3 bg-gray-800 rounded-full overflow-hidden">
            <motion.div
              className={`h-full rounded-full ${
                sentimentScore > 0 
                  ? 'bg-gradient-to-r from-green-600 to-green-400' 
                  : 'bg-gradient-to-r from-red-600 to-red-400'
              }`}
              initial={{ width: 0 }}
              animate={{ width: `${Math.abs(sentimentScore) * 50}%` }}
              transition={{ duration: 1, ease: 'easeOut' }}
              style={{
                marginLeft: sentimentScore > 0 ? '50%' : `${50 - Math.abs(sentimentScore) * 50}%`
              }}
            />
            
            {/* Center indicator */}
            <div className="absolute top-0 left-1/2 w-0.5 h-full bg-white/50 transform -translate-x-0.5" />
          </div>
        </div>

        {/* Market Regime Badge */}
        {currentSentiment.market_regime && (
          <div className="flex justify-center">
            <Badge 
              className={`${getRegimeColor(currentSentiment.market_regime)} bg-transparent`}
              variant="outline"
            >
              {currentSentiment.market_regime.replace('_', ' ').toUpperCase()}
            </Badge>
          </div>
        )}

        {/* Stats Grid */}
        <div className="grid grid-cols-2 gap-4 text-center">
          <div className="bg-gray-800 rounded p-3">
            <div className="text-sm text-gray-400">Sample Size</div>
            <div className="text-lg font-mono text-white">
              {currentSentiment.sample_count}
            </div>
          </div>
          
          <div className="bg-gray-800 rounded p-3">
            <div className="text-sm text-gray-400">Crypto Ratio</div>
            <div className="text-lg font-mono text-white">
              {formatPercentage(currentSentiment.crypto_ratio * 100, 0)}
            </div>
          </div>
        </div>

        {/* Sources Breakdown - Mobile Collapsible */}
        <div className="space-y-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setShowDetails(!showDetails)}
            className="w-full text-gray-400 hover:text-white justify-between"
          >
            <span>📊 Data Sources</span>
            <span>{showDetails ? '▲' : '▼'}</span>
          </Button>
          
          <AnimatePresence>
            {showDetails && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: 'auto', opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ duration: 0.3 }}
                className="space-y-3"
              >
                {/* Source Distribution */}
                <div className="space-y-2">
                  {Object.entries(currentSentiment.sources).map(([source, count]) => (
                    <div key={source} className="flex items-center justify-between">
                      <span className="text-sm text-gray-400 capitalize">
                        {source === 'twitter' ? '🐦' : source === 'reddit' ? '🤖' : 
                         source === 'discord' ? '💬' : '📰'} {source}
                      </span>
                      <div className="flex items-center gap-2">
                        <div className="w-16 h-2 bg-gray-700 rounded overflow-hidden">
                          <motion.div
                            className="h-full bg-blue-500"
                            initial={{ width: 0 }}
                            animate={{ width: `${(count / Math.max(...Object.values(currentSentiment.sources))) * 100}%` }}
                            transition={{ duration: 1, delay: 0.2 }}
                          />
                        </div>
                        <span className="text-xs text-white font-mono w-8 text-right">
                          {count}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>

                {/* Trending Slang Terms */}
                <div className="space-y-2">
                  <div className="text-sm text-gray-400 font-medium">🔥 Trending Terms</div>
                  <div className="space-y-1 max-h-32 overflow-y-auto">
                    {slangTerms.slice(0, isMobile ? 3 : 5).map((term, index) => (
                      <motion.div
                        key={term.term}
                        className="flex items-center justify-between p-2 bg-gray-800 rounded text-xs"
                        initial={{ opacity: 0, x: -10 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: index * 0.1 }}
                      >
                        <div className="flex items-center gap-2">
                          <span className="text-white font-medium">{term.term}</span>
                          <span className="text-xs">{getTrendIcon(term.trend)}</span>
                          <Badge 
                            className={`${getImpactBadgeColor(term.impact)} text-white text-xs px-1 py-0`}
                          >
                            {term.impact}
                          </Badge>
                        </div>
                        <span className="text-gray-400 font-mono">{term.count}</span>
                      </motion.div>
                    ))}
                  </div>
                </div>

                {/* Enhancement Multiplier */}
                {currentSentiment.enhancement_multiplier !== 0 && (
                  <div className="bg-purple-900/30 border border-purple-500/30 rounded p-3">
                    <div className="text-sm text-purple-300 mb-1">Signal Enhancement</div>
                    <div className="flex items-center justify-between">
                      <span className="text-purple-400 font-medium">
                        {currentSentiment.enhancement_multiplier > 0 ? '+' : ''}
                        {formatPercentage(currentSentiment.enhancement_multiplier * 100, 1)}
                      </span>
                      <Badge className={`${currentSentiment.enhancement_multiplier > 0 ? 'bg-green-600' : 'bg-red-600'} text-white`}>
                        {currentSentiment.enhancement_multiplier > 0 ? 'BOOST' : 'DAMPEN'}
                      </Badge>
                    </div>
                  </div>
                )}
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Demo Mode Indicator */}
        {demoMode && (
          <div className="text-center">
            <Badge className="bg-purple-600 text-white">
              🎭 Demo Data
            </Badge>
          </div>
        )}
      </CardContent>
    </motion.div>
  );
};

export default SentimentTelepathyWidget;
