// File: src/hooks/useRealtimeData.ts
import { useState, useEffect, useRef, useCallback } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';

interface RealtimeDataOptions {
  symbol?: string;
  updateInterval?: number;
  enableWebSocket?: boolean;
  autoReconnect?: boolean;
}

interface ActivityEvent {
  id: string;
  timestamp: string;
  type: 'sentiment' | 'gpu' | 'trade' | 'signal' | 'system';
  message: string;
  severity: 'info' | 'success' | 'warning' | 'error';
}

interface RealtimeData {
  tickers?: Record<string, any>;
  sentiment?: {
    sentiment: number;
    confidence: number;
    crypto_related: boolean;
    trend: number;
    sample_count: number;
    crypto_ratio: number;
    sources: Record<string, number>;
    enhancement_multiplier: number;
    market_regime: string;
    signal_boost_active: boolean;
    volatility_adjusted: boolean;
  };
  gpu?: {
    memory_used_gb: number;
    memory_total_gb: number;
    memory_usage_pct: number;
    processing_speed_ms: number;
    queue_throughput: number;
    temperature_c: number;
  };
  health?: {
    overall_score: number;
    api_connections: Record<string, string>;
    uptime_hours: number;
    total_trades: number;
    current_positions: number;
    portfolio_value: number;
  };
  portfolio?: {
    total_pnl: number;
    total_pnl_percent: number;
    open_positions: number;
    winning_positions: number;
    largest_winner: number;
    largest_loser: number;
    sentiment_enhanced_trades: number;
    risk_level: number;
    last_updated: string;
  };
  positions?: Array<{
    id: string;
    symbol: string;
    side: string;
    entry_price: number;
    current_price: number;
    size: number;
    pnl: number;
    pnl_percent: number;
    sentiment_enhanced: boolean;
    enhancement_multiplier: number;
    open_time: string;
  }>;
  agents?: Array<{
    id: string;
    name: string;
    port: number;
    status: 'active' | 'idle' | 'busy' | 'error';
    currentTask?: string;
    tasksCompleted: number;
    uptime: string;
    responseTime: number;
    accuracy: number;
    load: number;
  }>;
  activities?: ActivityEvent[];
}

export const useRealtimeData = <TData>(dataKey: keyof RealtimeData, _endpoint: string, options: RealtimeDataOptions = {}) => {
  const {
    symbol = 'BTC/USD',
    updateInterval = 2000,
    enableWebSocket = true,
    autoReconnect = true
  } = options;

  const [data, setData] = useState<RealtimeData>({});
  const [isConnected, setIsConnected] = useState(false);
  const [latency, setLatency] = useState<number>(0);
  const [reconnectAttempts, setReconnectAttempts] = useState(0);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const pingIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const queryClient = useQueryClient();

  // REST API fallback queries
  const { data: tickersData, refetch: refetchTickers } = useQuery({
    queryKey: ['tickers', symbol],
    queryFn: async () => {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/kraken/tickers`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbols: [symbol] })
      });
      if (!response.ok) throw new Error('Failed to fetch tickers');
      return response.json();
    },
    staleTime: updateInterval,
    enabled: !enableWebSocket || !isConnected
  });

  const { data: sentimentData, refetch: refetchSentiment } = useQuery({
    queryKey: ['sentiment'],
    queryFn: async () => {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/sentiment/current`);
      if (!response.ok) throw new Error('Failed to fetch sentiment');
      return response.json();
    },
    staleTime: updateInterval,
    enabled: !enableWebSocket || !isConnected
  });

  const { data: portfolioData, refetch: refetchPortfolio } = useQuery({
    queryKey: ['portfolio'],
    queryFn: async () => {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/portfolio/metrics`);
      if (!response.ok) throw new Error('Failed to fetch portfolio');
      return response.json();
    },
    staleTime: updateInterval,
    enabled: !enableWebSocket || !isConnected
  });

  const { data: positionsData, refetch: refetchPositions } = useQuery({
    queryKey: ['positions'],
    queryFn: async () => {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/positions/live`);
      if (!response.ok) throw new Error('Failed to fetch positions');
      const result = await response.json();
      return result.positions || [];
    },
    staleTime: updateInterval,
    enabled: !enableWebSocket || !isConnected
  });

  const { data: gpuData, refetch: refetchGpu } = useQuery({
    queryKey: ['gpu'],
    queryFn: async () => {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/gpu/metrics`);
      if (!response.ok) throw new Error('Failed to fetch GPU metrics');
      return response.json();
    },
    staleTime: updateInterval,
    enabled: !enableWebSocket || !isConnected
  });

  const { data: healthData, refetch: refetchHealth } = useQuery({
    queryKey: ['health'],
    queryFn: async () => {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/system/health`);
      if (!response.ok) throw new Error('Failed to fetch health');
      return response.json();
    },
    staleTime: updateInterval,
    enabled: !enableWebSocket || !isConnected
  });

  const { data: activitiesData, refetch: refetchActivities } = useQuery({
    queryKey: ['activities'],
    queryFn: async () => {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/activities/feed?limit=50`);
      if (!response.ok) throw new Error('Failed to fetch activities');
      return response.json();
    },
    staleTime: updateInterval,
    enabled: !enableWebSocket || !isConnected
  });

  const { data: agentsData, refetch: refetchAgents } = useQuery({
    queryKey: ['agents'],
    queryFn: async () => {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/agents/status`);
      if (!response.ok) throw new Error('Failed to fetch agents');
      return response.json();
    },
    staleTime: updateInterval,
    enabled: !enableWebSocket || !isConnected
  });

  // WebSocket connection setup
  const connectWebSocket = useCallback(() => {
    if (!enableWebSocket) return;

    try {
      const wsUrl = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8080/ws';
      const ws = new WebSocket(wsUrl);
      
      ws.onopen = () => {
        console.log('🔌 WebSocket connected');
        setIsConnected(true);
        setReconnectAttempts(0);
        
        // Subscribe to all channels
        ws.send(JSON.stringify({
          type: 'subscribe',
          channels: ['tickers', 'sentiment', 'gpu', 'health', 'portfolio', 'positions', 'agents', 'activities']
        }));

        // Start ping interval for latency monitoring
        pingIntervalRef.current = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) {
            const pingTime = Date.now();
            ws.send(JSON.stringify({ type: 'ping', timestamp: pingTime }));
          }
        }, 10000); // Ping every 10 seconds
      };

      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          const now = new Date();
          setLastUpdate(now);

          // Handle pong for latency calculation
          if (message.type === 'pong') {
            const pingTime = message.timestamp;
            const currentTime = Date.now();
            setLatency(currentTime - pingTime);
            return;
          }

          // Update data based on message type
          if (message.type === 'initial_data') {
            setData(message.data);
          } else if (message.type === 'ticker_update') {
            setData(prev => ({ ...prev, tickers: message.data }));
          } else if (message.type === 'sentiment_update') {
            setData(prev => ({ ...prev, sentiment: message.data }));
          } else if (message.type === 'gpu_update') {
            setData(prev => ({ ...prev, gpu: message.data }));
          } else if (message.type === 'health_update') {
            setData(prev => ({ ...prev, health: message.data }));
          } else if (message.type === 'portfolio_update') {
            setData(prev => ({ ...prev, portfolio: message.data }));
          } else if (message.type === 'position_update') {
            setData(prev => ({ ...prev, positions: message.data }));
          } else if (message.type === 'agent_update') {
            setData(prev => ({ ...prev, agents: message.data }));
          } else if (message.type === 'activity_update') {
            setData(prev => ({ ...prev, activities: message.data }));
          } else if (message.type === 'refresh_response') {
            setData(message.data);
          }

          // Invalidate React Query cache when receiving fresh data
          queryClient.invalidateQueries({ queryKey: ['tickers'] });
          queryClient.invalidateQueries({ queryKey: ['sentiment'] });
          queryClient.invalidateQueries({ queryKey: ['portfolio'] });
          queryClient.invalidateQueries({ queryKey: ['positions'] });
          queryClient.invalidateQueries({ queryKey: ['gpu'] });
          queryClient.invalidateQueries({ queryKey: ['health'] });
          queryClient.invalidateQueries({ queryKey: ['activities'] });
          queryClient.invalidateQueries({ queryKey: ['agents'] });

        } catch (error) {
          console.error('WebSocket message parsing error:', error);
        }
      };

      ws.onclose = () => {
        console.log('🔌 WebSocket disconnected');
        setIsConnected(false);
        
        if (pingIntervalRef.current) {
          clearInterval(pingIntervalRef.current);
          pingIntervalRef.current = null;
        }

        // Auto-reconnect logic
        if (autoReconnect && reconnectAttempts < 5) {
          const delay = Math.min(1000 * Math.pow(2, reconnectAttempts), 30000); // Exponential backoff, max 30s
          console.log(`🔄 Reconnecting in ${delay}ms... (attempt ${reconnectAttempts + 1})`);
          
          reconnectTimeoutRef.current = setTimeout(() => {
            setReconnectAttempts(prev => prev + 1);
            connectWebSocket();
          }, delay);
        }
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setIsConnected(false);
      };

      wsRef.current = ws;

    } catch (error) {
      console.error('WebSocket connection error:', error);
      setIsConnected(false);
    }
  }, [enableWebSocket, autoReconnect, reconnectAttempts, queryClient]);

  // Initialize WebSocket connection
  useEffect(() => {
    if (enableWebSocket) {
      connectWebSocket();
    } else {
      // Use REST API polling as fallback
      const interval = setInterval(() => {
        refetchTickers();
        refetchSentiment();
        refetchPortfolio();
        refetchPositions();
        refetchGpu();
        refetchHealth();
        refetchActivities();
        refetchAgents();
      }, updateInterval);

      return () => clearInterval(interval);
    }

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (pingIntervalRef.current) {
        clearInterval(pingIntervalRef.current);
      }
    };
  }, [enableWebSocket, connectWebSocket]);

  // Merge WebSocket data with REST API fallback data
  const mergedData: RealtimeData = {
    tickers: data.tickers || tickersData,
    sentiment: data.sentiment || sentimentData,
    gpu: data.gpu || gpuData,
    health: data.health || healthData,
    portfolio: data.portfolio || portfolioData,
    positions: data.positions || positionsData,
    agents: data.agents || agentsData,
    activities: data.activities || activitiesData
  };

  // Manual refresh function
  const refresh = useCallback(async () => {
    if (isConnected && wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'refresh_all' }));
    } else {
      // Use REST API for refresh
      await Promise.all([
        refetchTickers(),
        refetchSentiment(),
        refetchPortfolio(),
        refetchPositions(),
        refetchGpu(),
        refetchHealth(),
        refetchActivities(),
        refetchAgents()
      ]);
    }
  }, [isConnected, refetchTickers, refetchSentiment, refetchPortfolio, refetchPositions, refetchGpu, refetchHealth, refetchActivities, refetchAgents]);

  // Force reconnect function
  const reconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
    }
    setReconnectAttempts(0);
    setTimeout(() => connectWebSocket(), 1000);
  }, [connectWebSocket]);

  return {
    data: mergedData,
    isConnected,
    latency,
    reconnectAttempts,
    lastUpdate,
    refresh,
    reconnect,
    // Individual data accessors for convenience
    tickers: mergedData.tickers,
    sentiment: mergedData.sentiment,
    gpu: mergedData.gpu,
    health: mergedData.health,
    portfolio: mergedData.portfolio,
    positions: mergedData.positions,
    agents: mergedData.agents,
    activities: mergedData.activities
  };
};