// File: src/types/trading.ts

export interface Ticker {
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

export interface GPUMetrics {
  memory_used_gb: number;
  memory_total_gb: number;
  memory_usage_pct: number;
  processing_speed_ms: number;
  queue_throughput: number;
  temperature_c: number;
}

export interface SystemHealth {
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

export interface Position {
  id: string;
  symbol: string;
  side: 'long' | 'short';
  entry_price: number;
  current_price: number;
  size: number;
  pnl: number;
  pnl_percent: number;
  unrealized_pnl: number;
  sentiment_enhanced: boolean;
  enhancement_multiplier: number;
  open_time: string;
  time_open_minutes: number;
  stop_loss?: number;
  take_profit?: number;
  confidence?: number;
  sentiment_boost?: number;
}

export interface PortfolioMetrics {
  total_pnl: number;
  total_pnl_percent: number;
  open_positions: number;
  winning_positions: number;
  largest_winner: number;
  largest_loser: number;
  sentiment_enhanced_trades: number;
  risk_level: number;
  last_updated: string;
}

export interface ActivityEvent {
  id: string;
  timestamp: string;
  type: 'sentiment' | 'gpu' | 'trade' | 'signal' | 'system';
  message: string;
  severity: 'info' | 'success' | 'warning' | 'error';
  metadata?: Record<string, any>;
}

export interface Agent {
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
}

export interface MLTrainingStatus {
  training_id: string;
  status: 'started' | 'training' | 'completed' | 'failed';
  progress: number;
  model_type: string;
  metrics?: {
    loss?: number;
    accuracy?: number;
    validation_loss?: number;
    validation_accuracy?: number;
  };
  start_time?: string;
  end_time?: string;
  error_message?: string;
}

export interface SignalData {
  symbol: string;
  signal_strength: number;
  confidence: number;
  direction: 'buy' | 'sell' | 'hold';
  timeframe: string;
  sentiment_enhanced: boolean;
  enhancement_multiplier: number;
  market_regime: string;
  indicators: {
    rsi?: number;
    macd?: number;
    bollinger_bands?: {
      upper: number;
      middle: number;
      lower: number;
    };
    moving_averages?: {
      sma_20: number;
      sma_50: number;
      ema_12: number;
      ema_26: number;
    };
  };
}

export interface TradeAction {
  type: 'open_position' | 'close_position' | 'adjust_risk' | 'rebalance' | 'emergency_close';
  symbol?: string;
  side?: 'long' | 'short';
  size?: number;
  entry_price?: number;
  position_id?: string;
  risk_level?: number;
  strategy?: string;
}

export interface ApiResponse<T = any> {
  status: 'success' | 'error';
  data?: T;
  message?: string;
  error?: string;
  timestamp: string;
}

export interface WebSocketMessage {
  type: 'initial_data' | 'ticker_update' | 'sentiment_update' | 'gpu_update' | 
        'health_update' | 'portfolio_update' | 'position_update' | 'agent_update' | 
        'activity_update' | 'ping' | 'pong' | 'subscribe' | 'refresh_all' | 'refresh_response';
  data?: any;
  timestamp?: string;
  channels?: string[];
}

export interface DemoScenario {
  name: string;
  sentiment: number;
  confidence: number;
  duration: number;
  effects: string[];
  positions: Partial<Position>[];
}

export interface ChartData {
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  sentiment?: number;
}

export interface QuickAction {
  id: string;
  label: string;
  icon: string;
  action: TradeAction;
  confirmRequired: boolean;
  description: string;
  color: 'green' | 'red' | 'blue' | 'yellow' | 'purple';
}

export interface MarketRegime {
  current: 'bull_euphoria' | 'bull_optimism' | 'neutral_mixed' | 'bear_pessimism' | 'bear_panic';
  confidence: number;
  duration_minutes: number;
  volatility: 'low' | 'medium' | 'high';
  trend_strength: number;
  support_levels: number[];
  resistance_levels: number[];
}

export interface RiskMetrics {
  portfolio_var: number; // Value at Risk
  max_drawdown: number;
  sharpe_ratio: number;
  win_rate: number;
  profit_factor: number;
  current_risk_level: number;
  position_sizing: {
    max_position_size: number;
    current_exposure: number;
    available_margin: number;
  };
  alerts: Array<{
    level: 'low' | 'medium' | 'high' | 'critical';
    message: string;
    recommended_action: string;
  }>;
}

export interface TradingPair {
  symbol: string;
  base: string;
  quote: string;
  min_size: number;
  max_size: number;
  size_increment: number;
  price_increment: number;
  margin_enabled: boolean;
}

// Error types
export interface TradingError {
  code: string;
  message: string;
  details?: any;
  timestamp: string;
  endpoint?: string;
}

// Event emitter types
export interface EventCallbacks {
  onPositionOpened?: (position: Position) => void;
  onPositionClosed?: (position: Position, pnl: number) => void;
  onSentimentUpdate?: (sentiment: SentimentData) => void;
  onPriceAlert?: (symbol: string, price: number, alertLevel: number) => void;
  onRiskAlert?: (risk: RiskMetrics) => void;
  onConnectionLost?: () => void;
  onConnectionRestored?: () => void;
}

export type ConnectionStatus = 'connected' | 'disconnected' | 'reconnecting' | 'error';
export type TimeFrame = '1m' | '5m' | '15m' | '1h' | '4h' | '1d' | '1w';
export type OrderType = 'market' | 'limit' | 'stop' | 'stop_limit';
export type PositionSide = 'long' | 'short';
export type MarketCondition = 'bullish' | 'bearish' | 'sideways' | 'volatile';

// Utility types
export type Partial<T> = {
  [P in keyof T]?: T[P];
};

export type Required<T> = {
  [P in keyof T]-?: T[P];
};

export type Omit<T, K extends keyof T> = Pick<T, Exclude<keyof T, K>>;

export type Pick<T, K extends keyof T> = {
  [P in K]: T[P];
};