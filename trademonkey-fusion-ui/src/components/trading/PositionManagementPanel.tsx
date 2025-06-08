// File: src/components/trading/PositionManagementPanel.tsx
import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  TrendingUp, 
  TrendingDown, 
  DollarSign, 
  Shield, 
  Target,
  AlertTriangle,
  Zap,
  Activity
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';

interface Position {
  id: string;
  symbol: string;
  side: 'long' | 'short';
  size: number;
  entryPrice: number;
  currentPrice: number;
  pnl: number;
  pnlPercent: number;
  stopLoss: number;
  takeProfit: number;
  confidence: number;
  sentimentBoost: number;
}

interface PositionManagementProps {
  demoMode?: boolean;
}

const PositionManagementPanel: React.FC<PositionManagementProps> = ({ demoMode = false }) => {
  const [positions, setPositions] = useState<Position[]>([]);
  const [totalPnL, setTotalPnL] = useState(0);
  const [riskLevel, setRiskLevel] = useState(35);
  const [isGlitching, setIsGlitching] = useState(false);

  // Generate demo positions with cyberpunk flair
  useEffect(() => {
    if (demoMode) {
      const generatePositions = () => {
        const demoPositions: Position[] = [
          {
            id: 'pos_001',
            symbol: 'BTC/USD',
            side: 'long',
            size: 0.25,
            entryPrice: 64500,
            currentPrice: 65420,
            pnl: 230,
            pnlPercent: 1.43,
            stopLoss: 63200,
            takeProfit: 67800,
            confidence: 0.87,
            sentimentBoost: 0.35
          },
          {
            id: 'pos_002',
            symbol: 'ETH/USD',
            side: 'short',
            size: 2.1,
            entryPrice: 3420,
            currentPrice: 3380,
            pnl: 84,
            pnlPercent: 1.17,
            stopLoss: 3520,
            takeProfit: 3280,
            confidence: 0.72,
            sentimentBoost: -0.15
          }
        ];

        // Add some randomness for demo
        demoPositions.forEach(pos => {
          const randomChange = (Math.random() - 0.5) * 0.02;
          pos.currentPrice = pos.entryPrice * (1 + randomChange);
          pos.pnl = (pos.currentPrice - pos.entryPrice) * pos.size * (pos.side === 'long' ? 1 : -1);
          pos.pnlPercent = (pos.pnl / (pos.entryPrice * pos.size)) * 100;
        });

        setPositions(demoPositions);
        setTotalPnL(demoPositions.reduce((sum, pos) => sum + pos.pnl, 0));
        setRiskLevel(Math.max(0, Math.min(100, 35 + (Math.random() - 0.5) * 20)));
      };

      generatePositions();
      const interval = setInterval(generatePositions, 3000);
      return () => clearInterval(interval);
    }
  }, [demoMode]);

  // Cyberpunk glitch effect
  useEffect(() => {
    const glitchInterval = setInterval(() => {
      if (Math.random() > 0.97) {
        setIsGlitching(true);
        setTimeout(() => setIsGlitching(false), 150);
      }
    }, 2000);

    return () => clearInterval(glitchInterval);
  }, []);

  const getRiskColor = (risk: number) => {
    if (risk > 70) return 'text-cyber-pink';
    if (risk > 40) return 'text-yellow-400';
    return 'text-cyber-blue';
  };

  const getPnLColor = (pnl: number) => {
    return pnl >= 0 ? 'text-cyber-green' : 'text-cyber-pink';
  };

  const getProgressColor = (risk: number) => {
    if (risk > 70) return 'bg-gradient-to-r from-cyber-pink to-red-500';
    if (risk > 40) return 'bg-gradient-to-r from-yellow-400 to-orange-500';
    return 'bg-gradient-to-r from-cyber-blue to-cyber-green';
  };

  return (
    <div className="space-y-4">
      {/* Portfolio Overview Card */}
      <Card className="cyber-card">
        <CardHeader>
          <CardTitle className={`flex items-center justify-between text-cyber-blue glow-text ${isGlitching ? 'cyber-glitch' : ''}`}>
            <div className="flex items-center gap-2">
              <DollarSign className="w-5 h-5" />
              <span className="font-mono">PORTFOLIO_STATUS</span>
            </div>
            <Badge className="cyber-button border-cyber-blue text-cyber-blue">
              {positions.length} ACTIVE
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Total P&L Display */}
          <div className="holo-card p-4">
            <div className="text-sm text-cyber-blue font-mono mb-2 glow-text">TOTAL_P&L</div>
            <div className={`text-3xl font-mono font-bold ${getPnLColor(totalPnL)} glow-text`}>
              {totalPnL >= 0 ? '+' : ''}${totalPnL.toFixed(2)}
            </div>
            <div className="text-sm text-gray-400 font-mono">
              +{((totalPnL / 10000) * 100).toFixed(2)}% PORTFOLIO
            </div>
          </div>

          {/* Risk Thermometer */}
          <div className="holo-card p-4">
            <div className="flex items-center justify-between mb-3">
              <div className="text-sm text-cyber-blue font-mono glow-text">RISK_LEVEL</div>
              <div className={`text-lg font-mono font-bold ${getRiskColor(riskLevel)} glow-text`}>
                {riskLevel.toFixed(0)}%
              </div>
            </div>
            <div className="relative">
              <Progress 
                value={riskLevel} 
                className="h-3 bg-gray-800 border border-cyber-blue/30"
              />
              <div 
                className={`absolute top-0 left-0 h-3 rounded ${getProgressColor(riskLevel)} glow-border`}
                style={{ width: `${riskLevel}%` }}
              />
            </div>
            <div className="flex justify-between text-xs text-gray-400 font-mono mt-1">
              <span>SAFE</span>
              <span>MODERATE</span>
              <span>EXTREME</span>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Individual Positions */}
      <div className="space-y-3">
        <h3 className="text-lg font-mono text-cyber-blue glow-text">ACTIVE_POSITIONS</h3>
        <AnimatePresence>
          {positions.map((position, index) => (
            <motion.div
              key={position.id}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
              transition={{ delay: index * 0.1 }}
            >
              <Card className="cyber-card">
                <CardContent className="p-4">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <div className={`w-3 h-3 rounded-full ${position.side === 'long' ? 'bg-cyber-green' : 'bg-cyber-pink'} glow-border`} />
                      <span className="font-mono text-cyber-blue text-lg glow-text">{position.symbol}</span>
                      <Badge className={`${position.side === 'long' ? 'bg-cyber-green/20 text-cyber-green' : 'bg-cyber-pink/20 text-cyber-pink'} border-0 font-mono`}>
                        {position.side.toUpperCase()}
                      </Badge>
                    </div>
                    <div className="text-right">
                      <div className={`text-xl font-mono font-bold ${getPnLColor(position.pnl)} glow-text`}>
                        {position.pnl >= 0 ? '+' : ''}${position.pnl.toFixed(2)}
                      </div>
                      <div className={`text-sm font-mono ${getPnLColor(position.pnlPercent)}`}>
                        {position.pnlPercent >= 0 ? '+' : ''}{position.pnlPercent.toFixed(2)}%
                      </div>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-4 text-sm font-mono">
                    <div className="holo-card p-3">
                      <div className="text-gray-400 mb-1">ENTRY</div>
                      <div className="text-cyber-blue glow-text">${position.entryPrice.toFixed(2)}</div>
                    </div>
                    <div className="holo-card p-3">
                      <div className="text-gray-400 mb-1">CURRENT</div>
                      <div className="text-cyber-blue glow-text">${position.currentPrice.toFixed(2)}</div>
                    </div>
                    <div className="holo-card p-3">
                      <div className="text-gray-400 mb-1">STOP_LOSS</div>
                      <div className="text-cyber-pink glow-text">${position.stopLoss.toFixed(2)}</div>
                    </div>
                    <div className="holo-card p-3">
                      <div className="text-gray-400 mb-1">TAKE_PROFIT</div>
                      <div className="text-cyber-green glow-text">${position.takeProfit.toFixed(2)}</div>
                    </div>
                  </div>

                  {/* Sentiment Enhancement Indicator */}
                  {Math.abs(position.sentimentBoost) > 0.1 && (
                    <motion.div
                      className="mt-3 holo-card p-2 border-cyber-green"
                      animate={{ 
                        boxShadow: position.sentimentBoost > 0 
                          ? '0 0 15px rgba(0, 255, 0, 0.4)' 
                          : '0 0 15px rgba(255, 0, 128, 0.4)' 
                      }}
                    >
                      <div className="flex items-center gap-2 text-sm">
                        <Zap className="w-4 h-4 text-cyber-green" />
                        <span className="text-cyber-green font-mono glow-text">
                          SENTIMENT_BOOST: {position.sentimentBoost > 0 ? '+' : ''}{(position.sentimentBoost * 100).toFixed(0)}%
                        </span>
                      </div>
                    </motion.div>
                  )}
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>

      {/* Quick Actions */}
      <Card className="cyber-card">
        <CardContent className="p-4">
          <div className="grid grid-cols-2 gap-3">
            <button className="cyber-button p-3 text-left hover:glow-border transition-all">
              <div className="flex items-center gap-2 text-cyber-blue">
                <Shield className="w-4 h-4" />
                <span className="font-mono">RISK_MANAGEMENT</span>
              </div>
            </button>
            <button className="cyber-button p-3 text-left hover:glow-border transition-all">
              <div className="flex items-center gap-2 text-cyber-green">
                <Target className="w-4 h-4" />
                <span className="font-mono">AUTO_REBALANCE</span>
              </div>
            </button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default PositionManagementPanel;