# TradeMonkey Fusion UI

The front-end for the world's first AI with market telepathy! Built with Next.js, TypeScript, Tailwind, and shadcn/ui, this dashboard delivers real-time trading insights with GPU-accelerated sentiment analysis, now integrated with the OMEGA framework for multi-agent collaboration.

## Setup

1. **Install dependencies**:
   ```bash
   npm install
   ```

2. **Run development server**:
   ```bash
   npm run dev
   ```

3. **Build for production**:
   ```bash
   npm run build
   npm run start
   ```

## Key Components

- **DashboardLayout.tsx**: Responsive grid with all trading and system widgets.
- **SentimentTelepathyWidget.tsx**: Animated sentiment gauge with emoji rain.
- **TradingChartWithSentiment.tsx**: TradingView chart with sentiment heatmap.
- **PositionManagementPanel.tsx**: Position cards with risk thermometer and profit fountain.
- **MarketRegimeRadar.tsx**: Radar chart with regime icons and sweep animation.
- **GPUPerformanceMonitor.tsx**: Quantum cube and memory pool visuals.
- **SystemHealthDashboard.tsx**: Health score gauge and connection matrix.
- **ActivityFeed.tsx**: Virtualized event list with filters.
- **Sidebar.tsx**: Navigation with animated links.
- **HeaderBar.tsx**: Theme toggle and user profile.
- **MLTrainingDashboard.tsx**: Tabbed interface for data sources, ML training, and backtesting with OMEGA integration (Research Agent, TradeMonkey).
- **AIChatAssistant.tsx**: Chat interface with MCP tool integration (Web Search, Calculator, TradeMonkey, Research, Math Solver).
- **OMEGAAgentConsole.tsx**: Monitor OMEGA agent activity (Research, Math Solver, TradeMonkey, Workflow Planner).

## Testing

- **Unit Tests**:
  ```bash
  npm run test
  ```

- **E2E Tests** (Playwright):
  ```bash
  npm run test:e2e
  ```

## Performance Targets

- Initial Load: <1.4s (achieved with `React.lazy`)
- WebSocket Latency: <80ms
- Mobile Lighthouse Score: >90
- Animation: 60fps

## Demo Screenshots

- **AI Chat Assistant**: Real-time conversation with OMEGA agents, confidence scoring, and tool badges.
- **ML Training Dashboard**: Tabbed interface with training progress, backtest results, and profit particle effects.
- **OMEGA Agent Console**: Monitoring Research Agent, Math Solver, and TradeMonkey in action.

**LFG TO THE QUANTUM MOON!** 🚀