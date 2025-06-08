// File: D:\Repos\trade-monkey-lite\trade-monkey-gpu\trademonkey-fusion-ui\src\components\system\GPUPerformanceMonitor.tsx
import React from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { useRealtimeData } from '@/hooks/useRealtimeData';

interface GPUMetrics {
  memoryUsage: number;
  memoryTotal: number;
  memoryFree: number;
  temperature: number;
  processingSpeed: number;
  throughput: number;
  memoryHistory: number[];
  speedHistory: number[];
  throughputHistory: number[];
}

interface GPUPerformanceMonitorProps {
  demoMode?: boolean;
}

interface SparklineProps {
  label: string;
  data: number[];
  color?: string;
  unit?: string;
}

const GPUPerformanceMonitor: React.FC<GPUPerformanceMonitorProps> = ({ demoMode = false }) => {
  const { data } = useRealtimeData();
  
  // Mock GPU data with proper typing
  const gpuMetrics: GPUMetrics = {
    memoryUsage: 76,
    memoryTotal: 11264,
    memoryFree: 2702,
    temperature: 72,
    processingSpeed: 340,
    throughput: 2450,
    memoryHistory: [65, 70, 72, 75, 76, 78, 76, 74],
    speedHistory: [320, 340, 330, 350, 340, 345, 340, 335],
    throughputHistory: [2300, 2400, 2450, 2500, 2450, 2480, 2450, 2420]
  };

  const cubeGlow = gpuMetrics.memoryUsage > 80 ? '#ef4444' : '#10b981';
  const memoryLevel = gpuMetrics.memoryUsage;
  const isHighUsage = memoryLevel > 80;
  const isCriticalUsage = memoryLevel > 90;

  const getMemoryColor = (): string => {
    if (isCriticalUsage) return 'from-red-600 to-red-400';
    if (isHighUsage) return 'from-yellow-600 to-yellow-400';
    return 'from-blue-600 to-blue-400';
  };

  const getTemperatureColor = (temp: number): string => {
    if (temp > 80) return 'text-red-400';
    if (temp > 70) return 'text-yellow-400';
    return 'text-green-400';
  };

  return (
    <motion.div 
      className="p-4 rounded-lg bg-gray-900 border border-gray-700"
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.5 }}
    >
      <div className="text-xl font-bold text-white mb-4 flex items-center gap-2">
        ⚡ GPU Core
        <span className="text-sm text-gray-400 font-normal">
          ({gpuMetrics.memoryTotal}MB)
        </span>
      </div>
      
      {/* Quantum Cube */}
      <div className="flex items-center gap-4 mb-6">
        <motion.div
          className="w-20 h-20 border-2 rounded-lg flex items-center justify-center relative"
          style={{ 
            borderColor: cubeGlow,
            boxShadow: `0 0 20px ${cubeGlow}40`
          }}
          animate={{ 
            rotateY: 360,
            scale: [1, 1.05, 1]
          }}
          transition={{ 
            rotateY: { duration: 4, repeat: Infinity, ease: 'linear' },
            scale: { duration: 2, repeat: Infinity }
          }}
        >
          <motion.div
            className="text-2xl"
            animate={{ rotateY: -360 }}
            transition={{ duration: 4, repeat: Infinity, ease: 'linear' }}
          >
            🔮
          </motion.div>
        </motion.div>
        
        <div className="flex-1">
          <div className="text-sm text-gray-400 mb-1">
            Usage: <span className={isHighUsage ? 'text-red-400' : 'text-green-400'}>{memoryLevel}%</span>
          </div>
          <div className="text-sm text-gray-400 mb-1">
            Free: <span className="text-blue-400 font-mono">{gpuMetrics.memoryFree}MB</span>
          </div>
          <div className="text-sm text-gray-400">
            Temp: <span className={`font-mono ${getTemperatureColor(gpuMetrics.temperature)}`}>{gpuMetrics.temperature}°C</span>
          </div>
        </div>
      </div>
      
      {/* Memory Pool Animation */}
      <div className="mb-4 h-8 bg-gray-800 rounded overflow-hidden relative border border-gray-600">
        <motion.div
          className={`h-full bg-gradient-to-r ${getMemoryColor()}`}
          animate={{ width: `${memoryLevel}%` }}
          transition={{ duration: 0.5 }}
        >
          <motion.div 
            className="absolute inset-0 bg-gradient-to-r from-transparent to-white/20"
            animate={{ opacity: [0.3, 0.7, 0.3] }}
            transition={{ duration: 1.5, repeat: Infinity }}
          />
        </motion.div>
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-xs text-white font-mono drop-shadow-lg">
            {memoryLevel}% VRAM
          </span>
        </div>
      </div>
      
      {/* Performance Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-4">
        <Sparkline 
          label="Memory Usage" 
          data={gpuMetrics.memoryHistory} 
          color={isHighUsage ? '#ef4444' : '#3b82f6'}
          unit="%" 
        />
        <Sparkline 
          label="Processing Speed" 
          data={gpuMetrics.speedHistory} 
          color="#10b981"
          unit="ms" 
        />
      </div>
      
      <Sparkline 
        label="Throughput" 
        data={gpuMetrics.throughputHistory} 
        color="#8b5cf6"
        unit="/min" 
      />
      
      {/* Alerts */}
      <AnimatePresence>
        {isHighUsage && (
          <motion.div
            className={`mt-4 p-3 rounded text-sm ${
              isCriticalUsage 
                ? 'bg-red-900/50 text-red-200 border border-red-500' 
                : 'bg-yellow-900/50 text-yellow-200 border border-yellow-500'
            }`}
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
          >
            <div className="flex items-center gap-2">
              <span>{isCriticalUsage ? '🚨' : '⚠️'}</span>
              <span className="font-semibold">
                {isCriticalUsage ? 'Critical' : 'High'} Memory Usage: {gpuMetrics.memoryUsage}%
              </span>
            </div>
            <div className="text-xs mt-1 opacity-75">
              Consider reducing batch size or clearing cache
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};

const Sparkline: React.FC<SparklineProps> = ({ label, data, color = '#3b82f6', unit = '' }) => {
  const maxValue = Math.max(...data, 1);
  const minValue = Math.min(...data, 0);
  const range = maxValue - minValue || 1;
  
  const points = data.map((value, index) => {
    const x = (index / (data.length - 1)) * 100;
    const y = 100 - ((value - minValue) / range) * 100;
    return `${x},${y}`;
  }).join(' ');

  const currentValue = data[data.length - 1] || 0;

  return (
    <div className="flex flex-col">
      <div className="flex justify-between items-center mb-1">
        <span className="text-gray-400 text-sm">{label}</span>
        <span className="text-xs text-gray-500">
          {currentValue.toFixed(unit === 'ms' ? 0 : 1)}{unit}
        </span>
      </div>
      <div className="h-12 w-full bg-gray-800 rounded p-1 border border-gray-600">
        <svg viewBox="0 0 100 100" className="w-full h-full">
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
          <polyline
            fill="none"
            stroke={color}
            strokeWidth="2"
            points={points}
          />
        </svg>
      </div>
    </div>
  );
};

export default GPUPerformanceMonitor;
