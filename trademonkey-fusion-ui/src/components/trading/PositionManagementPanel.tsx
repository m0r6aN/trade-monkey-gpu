// File: src/components/trading/PositionManagementPanel.tsx
import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  TrendingUp, 
  TrendingDown, 
  DollarSign, 
  Shield, 
  Target,
  Zap,
  AlertTriangle,
  CheckCircle,
  Settings
} from 'lucide-react';
import { usePositions, usePortfolio } from '@/hooks/useRealtimeData';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { GlitchText, PulseGlow, SuccessParticles, LoadingSpinner } from '@/components/ui/cyberpunk-animations';

interface Position {
  id: string;
  symbol: string;
  side: 'long' | 'short';
  size: number;
  entry_price: number;
  current_price: number;
  pnl: number;
  pnl_percent: number;
  stop_loss: number;
  take_profit: number;
  confidence: number;
  sentiment_boost: number;
  time_open_minutes: number;
}

interface QuickAction {
  id: string;
  label: string;
  action: () => Promise<void>;
  loading: boolean;
  icon: React.ReactNode;
  color: string;
}

const PositionManagementPanel: React.FC = () => {
  const positions = usePositions();
  const portfolio = usePortfolio();
  const [selectedPosition, setSelectedPosition] = useState<string | null>(null);
  const [showSuccessParticles, setShowSuccessParticles] = useState(false);
  const [quickActions, setQuickActions] = useState<QuickAction[]>([]);
  const [riskLevel, setRiskLevel] = useState(0);
  const [isGlitching, setIsGlitching] = useState(false);

  // Initialize quick actions
  useEffect(() => {
    setQuickActions([
      {
        id: 'risk_management',
        label: 'RISK_MGMT',
        action: handleRiskManagement,
        loading: false,
        icon: <Shield className="w-4 h-4" />,
        color: 'bg-yellow-600 hover:bg-yellow-700'
      },
      {
        id: 'auto_rebalance',
        label: 'AUTO_REBALANCE',
        action: handleAutoRebalance,
        loading: false,
        icon: <Target className="w-4 h-4" />,
        color: 'bg-blue-600 hover:bg-blue-700'
      },
      {
        id: 'emergency_close',
        label: 'EMERGENCY_CLOSE',
        action: handleEmergencyClose,
        loading: false,
        icon: <AlertTriangle className="w-4 h-4" />,
        color: 'bg-red-600 hover:bg-red-700'
      }
    ]);
  }, []);

  // Calculate risk level from portfolio
  useEffect(() => {
    if (portfolio && positions) {
      const positionCount = Array.isArray(positions) ? positions.length : 0;
      const totalPnL = portfolio.total_pnl || 0;
      const calculatedRisk = Math.min(100, Math.max(0, 
        (positionCount * 15) + 
        (totalPnL < 0 ? Math.abs(totalPnL) * 0.1 : 0)
      ));
      setRiskLevel(calculatedRisk);
    }
  }, [portfolio, positions]);

  // Cyberpunk glitch effect
  useEffect(() => {
    const glitchInterval = setInterval(() => {
      if (Math.random() > 0.98) {
        setIsGlitching(true);
        setTimeout(() => setIsGlitching(false), 150);
      }
    }, 3000);

    return () => clearInterval(glitchInterval);
  }, []);

  const handleClosePosition = async (positionId: string) => {
    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/positions/${positionId}`, {
        method: 'DELETE',