import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  TrendingUp, 
  TrendingDown, 
  DollarSign, 
  Shield, 
  Target,
  Zap
} from 'lucide-react';
import { useRealtimeData } from '@/hooks/useRealtimeData';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';

interface Position {
  id: string;
  symbol: string;
  side: string;
  size: number;
  entry_price: number;
  current_price: number;
  pnl: number;
  pnl_percent: number;
  sentiment_enhanced: boolean;
  enhancement_multiplier: number;
  open_time: string;
}

const PositionManagementPanel: React.FC = () => {
  const { positions, isConnected } = useRealtimeData<Position[]>('positions', '/api/positions/live');
  const [totalPnL, setTotalPnL] = useState(0);
  const [riskLevel, setRiskLevel] = useState(35);
  const [isGlitching, setIsGlitching] = useState(false);

  // Calculate total P&L and risk level
  useEffect(() => {
    if (positions) {
      const total = positions.reduce((sum, pos) => sum + pos.pnl, 0);
      setTotalPnL(total);
      // Placeholder risk calculation (to be enhanced in Phase 3)
      const calculatedRisk = Math.max(0, Math.min(100, 35 + (positions.length * 10)));
      setRiskLevel(calculatedRisk);
    }
  }, [positions]);

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

  const handleClosePosition = async (positionId: string) => {
    try {
      await fetch(`/api/positions/${positionId}`, { method: 'DELETE' });
    } catch (err) {
      console.error('Failed to close position:', err);
    }
  };

  const getRiskColor = (risk: number) => {
    if (risk > 70) return 'text-laser-red';
    if (risk > 40) return 'text-gold-rush';
    return 'text-quantum-blue';
  };

  const getPnLColor = (pnl: number) => {
    return pnl >= 0 ? 'text-matrix-green' : 'text-laser-red';
  };

  const getProgressColor = (risk: number) => {
    if (risk > 70) return 'bg-gradient-to-r from-laser-red to-destructive';
    if (risk > 40) return 'bg-gradient-to-r from-gold-rush to-orange-500';
    return 'bg-gradient-to-r from-quantum-blue to-matrix-green';
  };

  if (!isConnected && !positions) {
    return (
      <Card className="p-4 bg-background quantum-glow">
        <div className="text-center text-muted-foreground">Loading positions...</div>
      </Card>
    );
  }

  return (
    <div className="space-y-4">
      {/* Portfolio Overview Card */}
      <Card className="bg-background border-border quantum-glow">
        <CardHeader>
          <CardTitle className={`flex items-center justify-between text-quantum-blue transition-colors ${isGlitching ? 'animate-pulse' : ''}`}>
            <div className="flex items-center gap-2">
              <DollarSign className="w-5 h-5" />
              <span className="font-mono">PORTFOLIO_STATUS</span>
            </div>
            <Badge className="bg-secondary text-secondary-foreground border-border">
              {positions?.length || 0} ACTIVE
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Total P&L Display */}
          <div className="p-4 border border-border rounded-lg">
            <div className="text-sm text-quantum-blue font-mono mb-2 transition-colors">TOTAL_P&L</div>
            <div className={`text-3xl font-mono font-bold ${getPnLColor(totalPnL)} transition-colors`}>
              {totalPnL >= 0 ? '+' : ''}${totalPnL.toFixed(2)}
            </div>
            <div className="text-sm text-muted-foreground font-mono">
              +{((totalPnL / 10000) * 100).toFixed(2)}% PORTFOLIO
            </div>
          </div>

          {/* Risk Thermometer */}
          <div className="p-4 border border-border rounded-lg">
            <div className="flex items-center justify-between mb-3">
              <div className="text-sm text-quantum-blue font-mono transition-colors">RISK_LEVEL</div>
              <div className={`text-lg font-mono font-bold ${getRiskColor(riskLevel)} transition-colors`}>
                {riskLevel.toFixed(0)}%
              </div>
            </div>
            <div className="relative">
              <Progress 
                value={riskLevel} 
                className="h-3 bg-muted border border-border"
              />
              <div 
                className={`absolute top-0 left-0 h-3 rounded ${getProgressColor(riskLevel)} matrix-glow transition-colors`}
                style={{ width: `${riskLevel}%` }}
              />
            </div>
            <div className="flex justify-between text-xs text-muted-foreground font-mono mt-1">
              <span>SAFE</span>
              <span>MODERATE</span>
              <span>EXTREME</span>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Individual Positions */}
      <div className="space-y-3">
        <h3 className="text-lg font-mono text-quantum-blue transition-colors">ACTIVE_POSITIONS</h3>
        <AnimatePresence>
          {positions?.map((position, index) => (
            <motion.div
              key={position.id}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
              transition={{ delay: index * 0.1 }}
            >
              <Card className="bg-background border-border quantum-glow">
                <CardContent className="p-4">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <div className={`w-3 h-3 rounded-full ${position.side === 'long' ? 'bg-matrix-green' : 'bg-laser-red'} matrix-glow`} />
                      <span className="font-mono text-quantum-blue text-lg transition-colors">{position.symbol}</span>
                      <Badge className={`${position.side === 'long' ? 'bg-matrix-green/20 text-matrix-green' : 'bg-laser-red/20 text-laser-red'} border-0 font-mono`}>
                        {position.side.toUpperCase()}
                      </Badge>
                    </div>
                    <div className="text-right">
                      <div className={`text-xl font-mono font-bold ${getPnLColor(position.pnl)} transition-colors`}>
                        {position.pnl >= 0 ? '+' : ''}${position.pnl.toFixed(2)}
                      </div>
                      <div className={`text-sm font-mono ${getPnLColor(position.pnl)} transition-colors`}>
                        {position.pnl >= 0 ? '+' : ''}{(position.pnl / (position.entry_price * position.size) * 100).toFixed(2)}%
                      </div>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-4 text-sm font-mono">
                    <div className="p-3 border border-border rounded-lg">
                      <div className="text-muted-foreground mb-1">ENTRY</div>
                      <div className="text-quantum-blue transition-colors">${position.entry_price.toFixed(2)}</div>
                    </div>
                    <div className="p-3 border border-border rounded-lg">
                      <div className="text-muted-foreground mb-1">CURRENT</div>
                      <div className="text-quantum-blue transition-colors">${position.current_price.toFixed(2)}</div>
                    </div>
                    <div className="p-3 border border-border rounded-lg">
                      <div className="text-muted-foreground mb-1">SIZE</div>
                      <div className="text-quantum-blue transition-colors">{position.size.toFixed(4)}</div>
                    </div>
                    <div className="p-3 border border-border rounded-lg">
                      <div className="text-muted-foreground mb-1">P&L %</div>
                      <div className={`${getPnLColor(position.pnl)} transition-colors`}>
                        {position.pnl_percent >= 0 ? '+' : ''}{position.pnl_percent.toFixed(2)}%
                      </div>
                    </div>
                  </div>

                  {/* Sentiment Enhancement Indicator */}
                  {position.sentiment_enhanced && (
                    <motion.div
                      className="mt-3 p-2 border border-matrix-green rounded-lg"
                      animate={{ 
                        boxShadow: '0 0 15px hsla(var(--matrix-green) / 0.4)'
                      }}
                    >
                      <div className="flex items-center gap-2 text-sm">
                        <Zap className="w-4 h-4 text-matrix-green" />
                        <span className="text-matrix-green font-mono transition-colors">
                          SENTIMENT_ENHANCED: {position.enhancement_multiplier.toFixed(2)}x
                        </span>
                      </div>
                    </motion.div>
                  )}

                  <Button
                    onClick={() => handleClosePosition(position.id)}
                    className="mt-3 bg-destructive text-destructive-foreground hover:bg-destructive/90"
                  >
                    Close Position
                  </Button>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>

      {/* Quick Actions */}
      <Card className="bg-background border-border quantum-glow">
        <CardContent className="p-4">
          <div className="grid grid-cols-2 gap-3">
            <Button className="p-3 text-left bg-secondary hover:bg-secondary/90 transition-colors">
              <div className="flex items-center gap-2 text-quantum-blue">
                <Shield className="w-4 h-4" />
                <span className="font-mono">RISK_MANAGEMENT</span>
              </div>
            </Button>
            <Button className="p-3 text-left bg-secondary hover:bg-secondary/90 transition-colors">
              <div className="flex items-center gap-2 text-matrix-green">
                <Target className="w-4 h-4" />
                <span className="font-mono">AUTO_REBALANCE</span>
              </div>
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default PositionManagementPanel;
