// File: src/components/trading/SentimentTelepathyWidget.tsx
import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useRealtimeData } from '@/hooks/useRealtimeData';
import { SentimentData } from '@/types/trading';

interface SentimentWidgetProps {
  sentiment?: SentimentData;
}

interface EmojiRainProps {
  emojis: string[];
  density: number;
}

interface SlangTickerProps {
  stats: { [term: string]: number };
}

const SentimentTelepathyWidget: React.FC<SentimentWidgetProps> = ({ sentiment: propSentiment }) => {
  const { data } = useRealtimeData({ 
    symbol: 'BTC/USD',
    enableWebSocket: true,
    updateInterval: 1000 
  });
  
  // Use prop data or fallback to realtime data
  const sentiment = propSentiment || data?.sentiment;
  const sentimentScore = sentiment?.sentiment || 0;  // Using 'sentiment' field now
  const confidence = sentiment?.confidence || 0;
  const slangStats = sentiment?.slangStats || {};
  
  const gaugeColor = sentimentScore > 0 
    ? `hsl(142, 76%, ${36 + sentimentScore * 14}%)`
    : `hsl(0, 76%, ${36 + Math.abs(sentimentScore) * 14}%)`;

  const pulseSpeed = Math.abs(sentimentScore) > 0.5 ? 0.5 : 1;

  return (
    <motion.div 
      className="cyber-card p-4 relative overflow-hidden"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <div className="text-xl font-bold text-cyber-blue mb-2 glow-text font-mono">🧠 MARKET_TELEPATHY</div>
      
      {/* Animated Sentiment Gauge */}
      <motion.div
        className="h-4 rounded-full relative overflow-hidden mb-3 glow-border"
        style={{ backgroundColor: gaugeColor }}
        animate={{ scale: [1, 1.02, 1] }}
        transition={{ duration: pulseSpeed, repeat: Infinity }}
      >
        <motion.div
          className="h-full bg-gradient-to-r from-transparent to-white/30"
          animate={{ width: `${Math.abs(sentimentScore) * 100}%` }}
          transition={{ duration: 0.5 }}
        />
      </motion.div>
      
      {/* Emoji Rain for Extreme Sentiment */}
      <AnimatePresence>
        {Math.abs(sentimentScore) > 0.7 && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute inset-0 pointer-events-none overflow-hidden"
          >
            <EmojiRain 
              emojis={sentimentScore > 0 ? ['🚀', '💎', '🌙'] : ['📉', '💀', '🔥']} 
              density={Math.abs(sentimentScore)} 
            />
          </motion.div>
        )}
      </AnimatePresence>
      
      {/* Stats Display */}
      <div className="mt-2 text-sm text-gray-400 grid grid-cols-2 gap-2 font-mono">
        <div>SENTIMENT: <span className="text-cyber-blue glow-text">{sentimentScore.toFixed(3)}</span></div>
        <div>CONFIDENCE: <span className="text-cyber-green glow-text">{(confidence * 100).toFixed(0)}%</span></div>
        <div>TREND: <span className="text-cyber-pink glow-text">{sentiment?.trend > 0 ? 'UP' : sentiment?.trend < 0 ? 'DOWN' : 'SIDEWAYS'}</span></div>
        <div>LATENCY: <span className="text-purple-400 glow-text">{data?.latency || 0}ms</span></div>
      </div>
      
      {/* Slang Ticker */}
      <SlangTicker stats={slangStats} />
      
      {/* Holographic overlay */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="w-full h-full bg-gradient-to-r from-transparent via-cyber-blue/5 to-transparent animate-pulse-glow" />
      </div>
    </motion.div>
  );
};

const EmojiRain: React.FC<EmojiRainProps> = ({ emojis, density }) => {
  const particles = Array.from({ length: Math.min(Math.floor(density * 15), 15) }, (_, i) => ({
    id: i,
    emoji: emojis[Math.floor(Math.random() * emojis.length)],
    x: Math.random() * 100,
    duration: 2 + Math.random() * 2,
    delay: Math.random() * 2,
  }));

  return (
    <div className="absolute inset-0">
      {particles.map((particle) => (
        <motion.div
          key={particle.id}
          className="absolute text-2xl"
          style={{ left: `${particle.x}%` }}
          initial={{ y: '-10%', opacity: 1 }}
          animate={{ y: '110%', opacity: 0 }}
          transition={{ 
            duration: particle.duration, 
            repeat: Infinity, 
            delay: particle.delay,
            ease: 'linear'
          }}
        >
          {particle.emoji}
        </motion.div>
      ))}
    </div>
  );
};

const SlangTicker: React.FC<SlangTickerProps> = ({ stats }) => {
  const entries = Object.entries(stats);
  
  if (entries.length === 0) {
    return (
      <div className="mt-2 text-xs text-gray-500 font-mono">
        ANALYZING_CRYPTO_SENTIMENT... 🔍
      </div>
    );
  }

  return (
    <div className="mt-2 text-xs text-gray-500 overflow-hidden font-mono">
      <motion.div
        animate={{ x: ['0%', '-100%'] }}
        transition={{ duration: 15, repeat: Infinity, ease: 'linear' }}
        className="whitespace-nowrap"
      >
        {entries.map(([term, count], index) => (
          <span key={`${term}-${index}`} className="mx-4 text-cyber-blue glow-text">
            {term.toUpperCase()}: <span className="text-cyber-green font-mono">{typeof count === 'number' ? count.toFixed(0) : '0'}</span>
          </span>
        ))}
      </motion.div>
    </div>
  );
};

export default SentimentTelepathyWidget;