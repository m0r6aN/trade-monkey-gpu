// File: src/components/trading/MarketRegimeRadar.tsx
import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Brain, Zap, TrendingUp, Activity } from 'lucide-react';

interface MarketRegime {
  name: string;
  confidence: number;
  color: string;
  icon: string;
  description: string;
}

interface RadarDataPoint {
  regime: string;
  strength: number;
  angle: number;
}

interface MarketRegimeRadarProps {
  demoMode?: boolean;
}

const MarketRegimeRadar: React.FC<MarketRegimeRadarProps> = ({ demoMode = false }) => {
  const [currentRegime, setCurrentRegime] = useState<MarketRegime>({
    name: 'QUANTUM_BULLISH',
    confidence: 0.87,
    color: '#00ffff',
    icon: '🐂',
    description: 'Strong upward momentum detected'
  });
  
  const [radarData, setRadarData] = useState<RadarDataPoint[]>([]);
  const [isScanning, setIsScanning] = useState(false);

  // Cyberpunk regime definitions
  const regimes = [
    { name: 'ACCUMULATION', color: '#00ff88', icon: '🟢', description: 'Smart money loading up' },
    { name: 'DISTRIBUTION', color: '#ff0080', icon: '🔴', description: 'Whales taking profits' },
    { name: 'MANIPULATION', color: '#ffff00', icon: '🟡', description: 'Market makers at work' },
    { name: 'VOLATILITY', color: '#ff8000', icon: '🟠', description: 'Chaos in the matrix' },
    { name: 'TREND', color: '#00ffff', icon: '🔵', description: 'Clear directional movement' },
    { name: 'SIDEWAYS', color: '#8000ff', icon: '🟣', description: 'Crab market engaged' }
  ];

  // Generate radar data
  useEffect(() => {
    if (demoMode) {
      const generateRadarData = () => {
        const data = regimes.map((regime, index) => ({
          regime: regime.name,
          strength: Math.random() * 100,
          angle: (index / regimes.length) * 360
        }));
        
        setRadarData(data);
        
        // Update current regime to strongest one
        const strongest = data.reduce((prev, current) => 
          current.strength > prev.strength ? current : prev
        );
        
        const regimeInfo = regimes.find(r => r.name === strongest.regime);
        if (regimeInfo) {
          setCurrentRegime({
            name: strongest.regime,
            confidence: strongest.strength / 100,
            color: regimeInfo.color,
            icon: regimeInfo.icon,
            description: regimeInfo.description
          });
        }
      };

      generateRadarData();
      const interval = setInterval(generateRadarData, 8000);
      return () => clearInterval(interval);
    }
  }, [demoMode]);

  // Scanning animation
  useEffect(() => {
    const scanInterval = setInterval(() => {
      setIsScanning(true);
      setTimeout(() => setIsScanning(false), 2000);
    }, 10000);

    return () => clearInterval(scanInterval);
  }, []);

  return (
    <Card className="cyber-card">
      <CardHeader>
        <CardTitle className="flex items-center justify-between text-cyber-blue glow-text">
          <div className="flex items-center gap-2">
            <Brain className="w-5 h-5" />
            <span className="font-mono cyber-glitch">MARKET_REGIME_RADAR</span>
          </div>
          <Badge className="cyber-button border-cyber-green text-cyber-green">
            <Activity className="w-3 h-3 mr-1" />
            SCANNING
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="p-6">
        {/* Radar Display */}
        <div className="relative w-80 h-80 mx-auto mb-6">
          {/* Radar Background */}
          <div className="absolute inset-0 rounded-full border-2 border-cyber-blue/30 bg-gradient-to-br from-gray-900/50 to-black/50 backdrop-blur-sm">
            {/* Radar Rings */}
            {[1, 2, 3, 4].map(ring => (
              <div
                key={ring}
                className="absolute border border-cyber-blue/20 rounded-full"
                style={{
                  top: `${12.5 * ring}%`,
                  left: `${12.5 * ring}%`,
                  right: `${12.5 * ring}%`,
                  bottom: `${12.5 * ring}%`
                }}
              />
            ))}
            
            {/* Radar Grid Lines */}
            <div className="absolute inset-0">
              <div className="absolute w-full h-0.5 bg-cyber-blue/20 top-1/2 transform -translate-y-0.5" />
              <div className="absolute h-full w-0.5 bg-cyber-blue/20 left-1/2 transform -translate-x-0.5" />
              <div className="absolute w-full h-0.5 bg-cyber-blue/20 top-1/2 transform -translate-y-0.5 rotate-45 origin-center" />
              <div className="absolute w-full h-0.5 bg-cyber-blue/20 top-1/2 transform -translate-y-0.5 -rotate-45 origin-center" />
            </div>
          </div>

          {/* Scanning Beam */}
          <AnimatePresence>
            {isScanning && (
              <motion.div
                className="absolute inset-0 rounded-full"
                initial={{ rotate: 0 }}
                animate={{ rotate: 360 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 2, ease: "linear" }}
                style={{
                  background: `conic-gradient(from 0deg, transparent 0deg, rgba(0, 255, 255, 0.3) 30deg, transparent 60deg)`
                }}
              />
            )}
          </AnimatePresence>

          {/* Radar Points */}
          {radarData.map((point, index) => {
            const radius = (point.strength / 100) * 120; // Max radius 120px
            const angleRad = (point.angle * Math.PI) / 180;
            const x = Math.cos(angleRad) * radius;
            const y = Math.sin(angleRad) * radius;
            
            return (
              <motion.div
                key={point.regime}
                className="absolute w-4 h-4 rounded-full glow-border"
                style={{
                  backgroundColor: regimes.find(r => r.name === point.regime)?.color || '#00ffff',
                  left: `calc(50% + ${x}px - 8px)`,
                  top: `calc(50% + ${y}px - 8px)`,
                  boxShadow: `0 0 10px ${regimes.find(r => r.name === point.regime)?.color || '#00ffff'}`
                }}
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ delay: index * 0.1 }}
              />
            );
          })}

          {/* Center Hub */}
          <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-6 h-6 rounded-full bg-cyber-blue glow-border" />
        </div>

        {/* Current Regime Display */}
        <motion.div
          className="holo-card p-4 text-center"
          animate={{ 
            boxShadow: `0 0 20px ${currentRegime.color}40` 
          }}
        >
          <div className="text-lg font-mono text-cyber-blue glow-text mb-2">
            DOMINANT_REGIME
          </div>
          <div 
            className="text-2xl font-mono font-bold glow-text mb-2"
            style={{ color: currentRegime.color }}
          >
            {currentRegime.icon} {currentRegime.name}
          </div>
          <div className="text-sm text-gray-400 font-mono mb-2">
            {currentRegime.description}
          </div>
          <Badge 
            className="cyber-button font-mono"
            style={{ 
              borderColor: currentRegime.color,
              color: currentRegime.color 
            }}
          >
            CONFIDENCE: {(currentRegime.confidence * 100).toFixed(0)}%
          </Badge>
        </motion.div>

        {/* Regime Legend */}
        <div className="grid grid-cols-2 gap-2 mt-4">
          {regimes.map((regime, index) => (
            <motion.div
              key={regime.name}
              className="holo-card p-2 text-xs font-mono text-center cursor-pointer hover:glow-border transition-all"
              whileHover={{ scale: 1.02 }}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.05 }}
            >
              <div 
                className="glow-text"
                style={{ color: regime.color }}
              >
                {regime.icon} {regime.name}
              </div>
            </motion.div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
};

export default MarketRegimeRadar;