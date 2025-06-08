// File: src/hooks/useRealtimeData.ts
import { useState, useEffect, useCallback } from 'react';
import { RealtimeData, SentimentData, TickerData, GPUData, HealthData, ActivityData } from '@/types/trading';

interface UseRealtimeDataProps {
  symbol?: string;
  enableWebSocket?: boolean;
  updateInterval?: number;
}

interface UseRealtimeDataReturn {
  data: RealtimeData;
  isConnected: boolean;
  latency: number;
  refresh: () => void;
}

// Mock data generators for demo mode
const generateMockSentiment = (): SentimentData => {
  const time = Date.now() / 10000;
  const baseScore = Math.sin(time) * 0.6;
  const noise = (Math.random() - 0.5) * 0.2;
  const sentimentValue = baseScore + noise;
  
  return {
    sentiment: sentimentValue,  // Main sentiment score (-1 to 1)
    confidence: 0.7 + Math.random() * 0.3,
    crypto_related: true,
    trend: sentimentValue > 0.1 ? 0.5 : sentimentValue < -0.1 ? -0.5 : 0,
    sample_count: Math.floor(Math.random() * 150) + 50,
    crypto_ratio: 0.6 + Math.random() * 0.4,
    sources: {
      twitter: Math.floor(Math.random() * 50) + 20,
      reddit: Math.floor(Math.random() * 30) + 10,
      discord: Math.floor(Math.random() * 20) + 5,
      news: Math.floor(Math.random() * 15) + 2
    },
    // Optional display fields
    slangStats: {
      'WAGMI': Math.floor(Math.random() * 100),
      'DIAMOND_HANDS': Math.floor(Math.random() * 80),
      'TO_THE_MOON': Math.floor(Math.random() * 60),
      'HODL': Math.floor(Math.random() * 90),
      'LAMBO': Math.floor(Math.random() * 40)
    },
    alignment: sentimentValue > 0.3 ? 'QUANTUM_BULLISH' : sentimentValue < -0.3 ? 'MATRIX_BEARISH' : 'CYBER_NEUTRAL',
    boost: sentimentValue * 0.5
  };
};

const generateMockTickers = (): TickerData => {
  const symbols = ['BTC/USD', 'ETH/USD', 'SOL/USD', 'ADA/USD'];
  const tickers: TickerData = {};
  
  symbols.forEach(symbol => {
    const basePrice = symbol === 'BTC/USD' ? 65000 : symbol === 'ETH/USD' ? 3200 : symbol === 'SOL/USD' ? 150 : 0.5;
    const change = (Math.random() - 0.5) * 10;
    
    tickers[symbol] = {
      symbol,
      price: basePrice + (basePrice * change / 100),
      change24h: change,
      volume24h: Math.random() * 1000000 + 500000,
      timestamp: Date.now()
    };
  });
  
  return tickers;
};

const generateMockGPU = (): GPUData => {
  const baseUsage = 40 + Math.sin(Date.now() / 60000) * 20 + Math.random() * 15;
  
  return {
    memory_used_gb: (baseUsage / 100) * 11,
    memory_total_gb: 11.0,
    memory_usage_pct: Math.min(95, Math.max(20, baseUsage)),
    processing_speed_ms: 200 + Math.random() * 300,
    queue_throughput: 2000 + Math.random() * 1000,
    temperature_c: 55 + (baseUsage / 100) * 20 + Math.random() * 10
  };
};

const generateMockHealth = (): HealthData => {
  const connections = ['healthy', 'healthy', 'healthy', 'warning'][Math.floor(Math.random() * 4)] as 'healthy' | 'warning' | 'critical';
  
  return {
    overall_score: connections === 'healthy' ? 95 + Math.random() * 5 : 75 + Math.random() * 15,
    api_connections: {
      kraken: connections,
      redis: 'healthy',
      sentiment_engine: 'healthy',
      gpu: connections
    },
    uptime_hours: 24 + Math.random() * 100,
    total_trades: 156 + Math.floor(Math.random() * 50),
    current_positions: Math.floor(Math.random() * 4)
  };
};

const generateMockActivities = (): ActivityData[] => {
  const activities: ActivityData[] = [];
  const types: ActivityData['type'][] = ['sentiment', 'gpu', 'trade', 'signal', 'system'];
  const severities: ActivityData['severity'][] = ['info', 'warning', 'error', 'success'];
  
  for (let i = 0; i < 10; i++) {
    const type = types[Math.floor(Math.random() * types.length)];
    const severity = severities[Math.floor(Math.random() * severities.length)];
    
    activities.push({
      id: `activity_${i}_${Date.now()}`,
      type,
      severity,
      message: `${type.toUpperCase()}: Mock activity message ${i + 1}`,
      timestamp: new Date(Date.now() - i * 60000).toISOString()
    });
  }
  
  return activities;
};

export const useRealtimeData = (props: UseRealtimeDataProps = {}): UseRealtimeDataReturn => {
  const {
    symbol = 'BTC/USD',
    enableWebSocket = true,
    updateInterval = 2000
  } = props;

  const [data, setData] = useState<RealtimeData>({
    sentiment: generateMockSentiment(),
    tickers: generateMockTickers(),
    gpu: generateMockGPU(),
    health: generateMockHealth(),
    activities: generateMockActivities(),
    latency: 0
  });
  
  const [isConnected, setIsConnected] = useState(false);
  const [latency, setLatency] = useState(0);

  const refresh = useCallback(() => {
    const startTime = Date.now();
    
    // Simulate API call latency
    setTimeout(() => {
      setData({
        sentiment: generateMockSentiment(),
        tickers: generateMockTickers(),
        gpu: generateMockGPU(),
        health: generateMockHealth(),
        activities: generateMockActivities(),
        latency: Date.now() - startTime
      });
      
      setLatency(Date.now() - startTime);
    }, 50 + Math.random() * 100); // 50-150ms latency
  }, []);

  useEffect(() => {
    if (!enableWebSocket) return;

    // Simulate WebSocket connection
    setIsConnected(true);
    
    const interval = setInterval(() => {
      refresh();
    }, updateInterval);

    // Simulate occasional disconnections
    const disconnectInterval = setInterval(() => {
      if (Math.random() > 0.95) { // 5% chance
        setIsConnected(false);
        setTimeout(() => {
          setIsConnected(true);
          refresh();
        }, 2000);
      }
    }, 10000);

    return () => {
      clearInterval(interval);
      clearInterval(disconnectInterval);
    };
  }, [enableWebSocket, updateInterval, refresh]);

  return {
    data,
    isConnected,
    latency,
    refresh
  };
};