// File: D:\Repos\trade-monkey-lite\trade-monkey-gpu\trademonkey-fusion-ui\src\components\system\SystemHealthDashboard.tsx
import React from 'react';
import { motion } from 'framer-motion';
import { useRealtimeData } from '@/hooks/useRealtimeData';

interface HealthData {
  healthScore: number;
  uptime: number;
  apiConnections: Array<{
    name: string;
    status: 'healthy' | 'warning' | 'critical';
    latency: number;
  }>;
  errorRates: number[];
  queueThroughput: number;
}

interface SystemHealthDashboardProps {
  demoMode?: boolean;
}

interface SparklineProps {
  label: string;
  data: number[];
  color?: string;
}

const SystemHealthDashboard: React.FC<SystemHealthDashboardProps> = ({ demoMode = false }) => {
  const { data } = useRealtimeData();
  
  // Mock health data with proper typing
  const healthData: HealthData = {
    healthScore: 99,
    uptime: 99.8,
    apiConnections: [
      { name: 'Kraken', status: 'healthy', latency: 45 },
      { name: 'Redis', status: 'healthy', latency: 12 },
      { name: 'GPU', status: 'healthy', latency: 8 },
      { name: 'Sentiment', status: 'warning', latency: 120 },
      { name: 'WebSocket', status: 'healthy', latency: 25 },
      { name: 'Database', status: 'healthy', latency: 15 }
    ],
    errorRates: [0.1, 0.2, 0.05, 0.3, 0.1, 0.0, 0.1, 0.15],
    queueThroughput: 2450
  };

  const getStatusColor = (status: string): string => {
    switch (status) {
      case 'healthy': return 'bg-green-900 text-green-200';
      case 'warning': return 'bg-yellow-900 text-yellow-200';
      case 'critical': return 'bg-red-900 text-red-200';
      default: return 'bg-gray-900 text-gray-200';
    }
  };

  const getHealthColor = (score: number): string => {
    if (score >= 95) return 'text-green-400';
    if (score >= 85) return 'text-yellow-400';
    return 'text-red-400';
  };

  return (
    <motion.div 
      className="p-4 rounded-lg bg-gray-900 border border-gray-700"
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <div className="text-xl font-bold text-white mb-4 flex items-center gap-2">
        🏥 System Health
        <span className="text-sm text-gray-400 font-normal">
          ({healthData.queueThroughput}/min)
        </span>
      </div>
      
      {/* Health Score Gauge */}
      <div className="flex items-center gap-4 mb-6">
        <motion.div
          className="w-20 h-20 rounded-full bg-gray-800 border-2 border-gray-600 flex items-center justify-center relative overflow-hidden"
          animate={{ scale: [1, 1.05, 1] }}
          transition={{ duration: 2, repeat: Infinity }}
        >
          <motion.div
            className="absolute inset-0 rounded-full"
            style={{
              background: `conic-gradient(from 0deg, ${healthData.healthScore >= 95 ? '#10b981' : '#ef4444'} ${healthData.healthScore * 3.6}deg, transparent ${healthData.healthScore * 3.6}deg)`
            }}
          />
          <span className={`text-2xl font-mono ${getHealthColor(healthData.healthScore)} relative z-10`}>
            {healthData.healthScore}
          </span>
        </motion.div>
        <div className="flex-1">
          <div className="text-sm text-gray-400 mb-1">
            System Uptime: <span className="text-green-400 font-mono">{healthData.uptime.toFixed(1)}%</span>
          </div>
          <div className="text-sm text-gray-400">
            Queue Throughput: <span className="text-blue-400 font-mono">{healthData.queueThroughput}</span>/min
          </div>
        </div>
      </div>
      
      {/* Connection Matrix */}
      <div className="grid grid-cols-2 md:grid-cols-3 gap-2 mb-4">
        {healthData.apiConnections.map((conn, index) => (
          <motion.div
            key={conn.name}
            className={`p-2 rounded text-center text-sm transition-all ${getStatusColor(conn.status)}`}
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: index * 0.1 }}
            whileHover={{ scale: 1.05 }}
          >
            <div className="font-semibold">{conn.name}</div>
            <div className="text-xs opacity-75">{conn.latency}ms</div>
          </motion.div>
        ))}
      </div>
      
      {/* Queue Particle Flow */}
      <div className="h-8 bg-gray-800 rounded overflow-hidden relative mb-4 border border-gray-600">
        <motion.div
          className="absolute inset-0 flex items-center gap-2 px-2"
          animate={{ x: ['0%', '-100%'] }}
          transition={{ duration: 8, repeat: Infinity, ease: 'linear' }}
        >
          {Array.from({ length: 15 }).map((_, i) => (
            <motion.div 
              key={i} 
              className="w-2 h-2 bg-blue-400 rounded-full flex-shrink-0"
              animate={{ 
                scale: [1, 1.5, 1],
                opacity: [0.5, 1, 0.5]
              }}
              transition={{ 
                duration: 1, 
                repeat: Infinity, 
                delay: i * 0.2 
              }}
            />
          ))}
        </motion.div>
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-xs text-gray-500 font-mono">Processing Queue</span>
        </div>
      </div>
      
      {/* Error Rate Sparkline */}
      <Sparkline 
        label="Error Rate (%)" 
        data={healthData.errorRates} 
        color="#ef4444"
      />
    </motion.div>
  );
};

const Sparkline: React.FC<SparklineProps> = ({ label, data, color = '#3b82f6' }) => {
  const maxValue = Math.max(...data, 1);
  const points = data.map((value, index) => {
    const x = (index / (data.length - 1)) * 100;
    const y = 100 - (value / maxValue) * 100;
    return `${x},${y}`;
  }).join(' ');

  return (
    <div className="flex flex-col">
      <span className="text-gray-400 text-sm mb-1">{label}</span>
      <div className="h-12 w-full bg-gray-800 rounded p-1">
        <svg viewBox="0 0 100 100" className="w-full h-full">
          <polyline
            fill="none"
            stroke={color}
            strokeWidth="2"
            points={points}
          />
          <defs>
            <linearGradient id={`gradient-${label}`} x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" style={{ stopColor: color, stopOpacity: 0.3 }} />
              <stop offset="100%" style={{ stopColor: color, stopOpacity: 0.1 }} />
            </linearGradient>
          </defs>
          <polygon
            fill={`url(#gradient-${label})`}
            points={`0,100 ${points} 100,100`}
          />
        </svg>
      </div>
      <span className="text-xs text-gray-500 mt-1">
        Current: {data[data.length - 1]?.toFixed(2) || '0.00'}
      </span>
    </div>
  );
};

export default SystemHealthDashboard;
