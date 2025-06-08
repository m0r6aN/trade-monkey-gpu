// File: src/types/trading.ts
export interface SentimentData {
  sentiment: number;  // The actual sentiment score (-1 to 1)
  confidence: number; // Confidence in the prediction (0 to 1)
  crypto_related: boolean; // Whether the sentiment is crypto-related
  trend: number; // Trend direction as number (-1 to 1)
  sample_count: number; // Number of samples analyzed
  crypto_ratio: number; // Ratio of crypto-related content (0 to 1)
  sources: {
    twitter: number;
    reddit: number;
    discord: number;
    news: number;
  };
  // Optional fields for enhanced display
  slangStats?: { [term: string]: number };
  alignment?: string; // Human-readable alignment like "QUANTUM_BULLISH"
  boost?: number; // Signal boost percentage
}

export interface TickerData {
  [symbol: string]: {
    symbol: string;
    price: number;
    change24h: number;
    volume24h: number;
    timestamp: number;
  };
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
    kraken: 'healthy' | 'warning' | 'critical';
    redis: 'healthy' | 'warning' | 'critical';
    sentiment_engine: 'healthy' | 'warning' | 'critical';
    gpu: 'healthy' | 'warning' | 'critical';
  };
  uptime_hours: number;
  total_trades: number;
  current_positions: number;
}

export interface ActivityData {
  id: string;
  type: 'sentiment' | 'gpu' | 'trade' | 'signal' | 'system';
  message: string;
  timestamp: string;
  severity: 'info' | 'warning' | 'error' | 'success';
}

export interface RealtimeData {
  sentiment: SentimentData;
  tickers: TickerData;
  gpu: GPUData;
  health: HealthData;
  activities: ActivityData[];
  latency?: number;
}

export interface SentimentSignal {
  confidence: number;
  sentiment: number;
  boost: number;
  alignment: string;
}

export interface ChartData {
  timestamp: number;
  price: number;
  volume: number;
  sentiment: number;
  signals: Array<{
    type: 'buy' | 'sell';
    price: number;
    confidence: number;
    boost: number;
  }>;
}