// File: src/hooks/useRealtimeData.ts
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useEffect, useRef, useState, useCallback } from 'react';

export interface TickerData {
  [symbol: string]: {
    symbol: string;
    price: number;
    change24h: number;
    volume24h: number;
    high24h: number;
    low24h: number;
    bid: number;
    ask: number;
    spread: number;
    timestamp: number;
    market_cap_rank: number;
  };
}

export interface SentimentData {
  sentiment: number;
  confidence: number;
  crypto_related: boolean;
  trend: number;
  sample_count: number;
  crypto_ratio: number;
  sources: {
    twitter: number;
    reddit: number;
    discord: number;
    news: number;
  };
  enhancement_multiplier: number;
  market_regime: string;
  signal_boost_active: boolean;
  volatility_adjusted: boolean;
}

export interface GPUData {
  memory_used_gb: number;
  memory_total_gb: number;
  memory_usage_pct: number;
  processing_speed_ms: number;
  queue_throughput: number;
  temperature_c: number;
}

export interface HealthData {
  overall_score: number;
  api_connections: {
    kraken: string;
    redis: string;
    sentiment_engine: string;
    gpu: string;
  };
  uptime_hours: number;
  total_trades: number;
  current_positions: number;
  portfolio_value: number;
}

export interface PortfolioData {
  total_pnl: number;
  total_pnl_percent: number;
  open_positions: number;
  winning_positions: number;
  largest_winner: number;
  largest_loser: number;
  sentiment_enhanced_trades: number;
  last_updated: string;
}

export interface ActivityEvent {
  id: string;
  timestamp: string;
  type: 'sentiment' | 'gpu' | 'trade' | 'signal' | 'system';
  message: string;
  severity: 'info' | 'success' | 'warning' | 'error';
}

export interface AgentStatus {
  id: string;
  name: string;
  port: number;
  status: 'active' | 'idle' | 'error' | 'busy';
  currentTask?: string;
  tasksCompleted: number;
  uptime: string;
  responseTime: number;
  accuracy: number;
  load: number;
}

export interface RealtimeData {
  tickers: TickerData;
  sentiment: SentimentData;
  gpu: GPUData;
  health: HealthData;
  portfolio: PortfolioData;
  activities: ActivityEvent[];
  agents: AgentStatus[];
  positions: any[];
}

type WebSocketMessage = 
  | { type: 'ticker_update'; data: TickerData; timestamp: string }
  | { type: 'sentiment_update'; data: SentimentData; timestamp: string }
  | { type: 'gpu_update'; data: GPUData; timestamp: string }
  | { type: 'health_update'; data: HealthData; timestamp: string }
  | { type: 'portfolio_update'; data: PortfolioData; timestamp: string }
  | { type: 'activity_update'; data: ActivityEvent[]; timestamp: string }
  | { type: 'agent_update'; data: AgentStatus[]; timestamp: string }
  | { type: 'initial_data'; data: RealtimeData }
  | { type: 'ping' | 'pong' };

interface UseRealtimeDataOptions {
  symbol?: string;
  enableWebSocket?: boolean;
  updateInterval?: number;
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080';
const WS_URL = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8080/ws';

export const useRealtimeData = (options: UseRealtimeDataOptions = {}) => {
  const { enableWebSocket = true, updateInterval = 2000 } = options;
  const queryClient = useQueryClient();
  const wsRef = useRef<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  // Individual data queries
  const tickers = useQuery({
    queryKey: ['tickers'],
    queryFn: async (): Promise<TickerData> => {
      const response = await fetch(`${API_BASE_URL}/api/kraken/tickers`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbols: ['BTC/USD', 'ETH/USD', 'SOL/USD', 'ADA/USD'] })
      });
      if (!response.ok) throw new Error('Failed to fetch tickers');
      return response.json();
    },
    staleTime: 1000 * 5, // 5 seconds
    enabled: !enableWebSocket
  });

  const sentiment = useQuery({
    queryKey: ['sentiment'],
    queryFn: async (): Promise<SentimentData> => {
      const response = await fetch(`${API_BASE_URL}/api/sentiment/current`);
      if (!response.ok) throw new Error('Failed to fetch sentiment');
      return response.json();
    },
    staleTime: 1000 * 10, // 10 seconds
    enabled: !enableWebSocket
  });

  const gpu = useQuery({
    queryKey: ['gpu'],
    queryFn: async (): Promise<GPUData> => {
      const response = await fetch(`${API_BASE_URL}/api/gpu/metrics`);
      if (!response.ok) throw new Error('Failed to fetch GPU metrics');
      return response.json();
    },
    staleTime: 1000 * 5, // 5 seconds
    enabled: !enableWebSocket
  });

  const health = useQuery({
    queryKey: ['health'],
    queryFn: async (): Promise<HealthData> => {
      const response = await fetch(`${API_BASE_URL}/api/system/health`);
      if (!response.ok) throw new Error('Failed to fetch system health');
      return response.json();
    },
    staleTime: 1000 * 30, // 30 seconds
    enabled: !enableWebSocket
  });

  const portfolio = useQuery({
    queryKey: ['portfolio'],
    queryFn: async (): Promise<PortfolioData> => {
      const response = await fetch(`${API_BASE_URL}/api/portfolio/metrics`);
      if (!response.ok) throw new Error('Failed to fetch portfolio metrics');
      return response.json();
    },
    staleTime: 1000 * 5, // 5 seconds
    enabled: !enableWebSocket
  });

  const activities = useQuery({
    queryKey: ['activities'],
    queryFn: async (): Promise<ActivityEvent[]> => {
      const response = await fetch(`${API_BASE_URL}/api/activities/feed`);
      if (!response.ok) throw new Error('Failed to fetch activities');
      return response.json();
    },
    staleTime: 1000 * 10, // 10 seconds
    enabled: !enableWebSocket
  });

  const agents = useQuery({
    queryKey: ['agents'],
    queryFn: async (): Promise<AgentStatus[]> => {
      const response = await fetch(`${API_BASE_URL}/api/agents/status`);
      if (!response.ok) throw new Error('Failed to fetch agent status');
      return response.json();
    },
    staleTime: 1000 * 15, // 15 seconds
    enabled: !enableWebSocket
  });

  const positions = useQuery({
    queryKey: ['positions'],
    queryFn: async () => {
      const response = await fetch(`${API_BASE_URL}/api/positions/live`);
      if (!response.ok) throw new Error('Failed to fetch positions');
      return response.json();
    },
    staleTime: 1000 * 5, // 5 seconds
    enabled: !enableWebSocket
  });

  // WebSocket connection management
  const connectWebSocket = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    try {
      wsRef.current = new WebSocket(WS_URL);
      
      wsRef.current.onopen = () => {
        console.log('🔌 WebSocket connected');
        setIsConnected(true);
        setConnectionError(null);
        
        // Subscribe to all channels
        wsRef.current?.send(JSON.stringify({
          type: 'subscribe',
          channels: ['tickers', 'sentiment', 'gpu', 'health', 'portfolio', 'activities', 'agents']
        }));
      };

      wsRef.current.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data);
          setLastUpdate(new Date());

          switch (message.type) {
            case 'initial_data':
              // Set all initial data
              queryClient.setQueryData(['tickers'], message.data.tickers);
              queryClient.setQueryData(['sentiment'], message.data.sentiment);
              queryClient.setQueryData(['gpu'], message.data.gpu);
              queryClient.setQueryData(['health'], message.data.health);
              queryClient.setQueryData(['portfolio'], message.data.portfolio);
              queryClient.setQueryData(['activities'], message.data.activities);
              queryClient.setQueryData(['agents'], message.data.agents);
              queryClient.setQueryData(['positions'], message.data.positions);
              break;
            
            case 'ticker_update':
              queryClient.setQueryData(['tickers'], message.data);
              break;
            
            case 'sentiment_update':
              queryClient.setQueryData(['sentiment'], message.data);
              break;
            
            case 'gpu_update':
              queryClient.setQueryData(['gpu'], message.data);
              break;
            
            case 'health_update':
              queryClient.setQueryData(['health'], message.data);
              break;
            
            case 'portfolio_update':
              queryClient.setQueryData(['portfolio'], message.data);
              break;
            
            case 'activity_update':
              queryClient.setQueryData(['activities'], message.data);
              break;
            
            case 'agent_update':
              queryClient.setQueryData(['agents'], message.data);
              break;

            case 'ping':
              wsRef.current?.send(JSON.stringify({ type: 'pong' }));
              break;
          }
        } catch (error) {
          console.error('WebSocket message parsing error:', error);
        }
      };

      wsRef.current.onerror = (error) => {
        console.error('WebSocket error:', error);
        setConnectionError('WebSocket connection error');
        setIsConnected(false);
      };

      wsRef.current.onclose = () => {
        console.log('🔌 WebSocket disconnected');
        setIsConnected(false);
        
        // Attempt to reconnect after 5 seconds
        setTimeout(() => {
          if (enableWebSocket) {
            connectWebSocket();
          }
        }, 5000);
      };

    } catch (error) {
      console.error('WebSocket connection failed:', error);
      setConnectionError('Failed to connect to WebSocket');
      setIsConnected(false);
    }
  }, [enableWebSocket, queryClient]);

  // Initialize WebSocket connection
  useEffect(() => {
    if (enableWebSocket) {
      connectWebSocket();
    }

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [enableWebSocket, connectWebSocket]);

  // Manual refresh function
  const refresh = useCallback(async () => {
    if (enableWebSocket && wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'refresh_all' }));
    } else {
      // Manually refetch all queries
      await Promise.all([
        queryClient.refetchQueries({ queryKey: ['tickers'] }),
        queryClient.refetchQueries({ queryKey: ['sentiment'] }),
        queryClient.refetchQueries({ queryKey: ['gpu'] }),
        queryClient.refetchQueries({ queryKey: ['health'] }),
        queryClient.refetchQueries({ queryKey: ['portfolio'] }),
        queryClient.refetchQueries({ queryKey: ['activities'] }),
        queryClient.refetchQueries({ queryKey: ['agents'] }),
        queryClient.refetchQueries({ queryKey: ['positions'] }),
      ]);
    }
  }, [enableWebSocket, queryClient]);

  // Combined data object
  const data: RealtimeData = {
    tickers: tickers.data || {},
    sentiment: sentiment.data || {} as SentimentData,
    gpu: gpu.data || {} as GPUData,
    health: health.data || {} as HealthData,
    portfolio: portfolio.data || {} as PortfolioData,
    activities: activities.data || [],
    agents: agents.data || [],
    positions: positions.data?.positions || []
  };

  const isLoading = enableWebSocket ? 
    !isConnected :
    tickers.isLoading || sentiment.isLoading || gpu.isLoading || health.isLoading;

  const error = connectionError || 
    tickers.error || sentiment.error || gpu.error || health.error || 
    portfolio.error || activities.error || agents.error || positions.error;

  return {
    data,
    isLoading,
    error,
    isConnected,
    connectionError,
    lastUpdate,
    refresh,
    // Individual query states for granular control
    queries: {
      tickers,
      sentiment,
      gpu,
      health,
      portfolio,
      activities,
      agents,
      positions
    }
  };
};

// Utility hooks for specific data types
export const useTickers = () => {
  const { data } = useRealtimeData();
  return data.tickers;
};

export const useSentiment = () => {
  const { data } = useRealtimeData();
  return data.sentiment;
};

export const useGPUMetrics = () => {
  const { data } = useRealtimeData();
  return data.gpu;
};

export const useSystemHealth = () => {
  const { data } = useRealtimeData();
  return data.health;
};

export const usePortfolio = () => {
  const { data } = useRealtimeData();
  return data.portfolio;
};

export const useActivities = () => {
  const { data } = useRealtimeData();
  return data.activities;
};

export const useAgents = () => {
  const { data } = useRealtimeData();
  return data.agents;
};

export const usePositions = () => {
  const { data } = useRealtimeData();
  return data.positions;
};