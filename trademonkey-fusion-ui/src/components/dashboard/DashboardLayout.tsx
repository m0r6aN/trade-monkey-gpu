// File: src/components/dashboard/DashboardLayout.tsx (Updated with new structure)
import React, { useState } from 'react';
import { motion } from 'framer-motion';
import SentimentTelepathyWidget from '@/components/trading/SentimentTelepathyWidget';
import TradingChartWithSentiment from '@/components/trading/TradingChartWithSentiment';
import PositionManagementPanel from '@/components/trading/PositionManagementPanel';
import MarketRegimeRadar from '@/components/trading/MarketRegimeRadar';
import GPUPerformanceMonitor from '@/components/system/GPUPerformanceMonitor';
import SystemHealthDashboard from '@/components/system/SystemHealthDashboard';
import ActivityFeed from '@/components/system/ActivityFeed';
import DemoMode from '@/components/dashboard/DemoMode';
import OMEGAAgentConsole from '@/components/dashboard/OMEGAAgentConsole';
import TabbedPanels from '@/components/dashboard/TabbedPanels';
import QuickActionPanel from '@/components/dashboard/QuickActionPanel';
import { useRealtimeData } from '@/hooks/useRealtimeData';

const DashboardLayout = () => {
  const [isDemoMode, setIsDemoMode] = useState(false);
  const [agentPanelCollapsed, setAgentPanelCollapsed] = useState(false);
  const { data, isConnected, refresh } = useRealtimeData({
    symbol: 'BTC/USD',
    enableWebSocket: true,
    updateInterval: 2000
  });

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
      },
    },
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.3 } },
  };

  return (
    <div className="min-h-screen bg-gray-950 relative overflow-hidden">
      {/* OMEGA Agent Console - Side Panel */}
      <OMEGAAgentConsole 
        isCollapsed={agentPanelCollapsed}
        onToggle={() => setAgentPanelCollapsed(!agentPanelCollapsed)}
      />

      {/* Main Content Area */}
      <motion.div
        className={`transition-all duration-300 ${
          agentPanelCollapsed ? 'ml-16' : 'ml-80'
        } mb-80`} // Bottom margin for tabbed panels
        variants={containerVariants}
        initial="hidden"
        animate="visible"
      >
        {/* Trading Zone - Green Accent */}
        <div className="p-4 space-y-4">
          {/* Main Trading Chart - Priority 1 */}
          <motion.div 
            variants={itemVariants} 
            className="border-l-4 border-green-500 bg-green-500/5 rounded-r-lg"
          >
            <TradingChartWithSentiment 
              tickers={data?.tickers} 
              sentiment={data?.sentiment}
              demoMode={isDemoMode}
            />
          </motion.div>

          {/* Trading Controls Row */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <motion.div 
              variants={itemVariants}
              className="border-l-4 border-green-500 bg-green-500/5 rounded-r-lg"
            >
              <PositionManagementPanel demoMode={isDemoMode} />
            </motion.div>
            
            <motion.div 
              variants={itemVariants}
              className="border-l-4 border-green-500 bg-green-500/5 rounded-r-lg"
            >
              <MarketRegimeRadar demoMode={isDemoMode} />
            </motion.div>
          </div>

          {/* AI/Sentiment Zone - Purple Accent */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <motion.div 
              variants={itemVariants}
              className="border-l-4 border-purple-500 bg-purple-500/5 rounded-r-lg"
            >
              <SentimentTelepathyWidget sentiment={data?.sentiment} />
            </motion.div>
            
            {/* System Zone - Blue Accent */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <motion.div 
                variants={itemVariants}
                className="border-l-4 border-blue-500 bg-blue-500/5 rounded-r-lg"
              >
                <GPUPerformanceMonitor demoMode={isDemoMode} />
              </motion.div>
              
              <motion.div 
                variants={itemVariants}
                className="border-l-4 border-blue-500 bg-blue-500/5 rounded-r-lg"
              >
                <SystemHealthDashboard demoMode={isDemoMode} />
              </motion.div>
            </div>
          </div>

          {/* Activity Feed - Full Width */}
          <motion.div 
            variants={itemVariants}
            className="border-l-4 border-blue-500 bg-blue-500/5 rounded-r-lg"
          >
            <ActivityFeed demoMode={isDemoMode} />
          </motion.div>
        </div>
      </motion.div>

      {/* Quick Action Panel - Right Side */}
      <QuickActionPanel demoMode={isDemoMode} />

      {/* Tabbed Panels - Bottom */}
      <TabbedPanels demoMode={isDemoMode} />

      {/* Demo Mode Controller */}
      {isDemoMode && <DemoMode />}

      {/* Demo Toggle Button */}
      <motion.button
        onClick={() => {
          setIsDemoMode(!isDemoMode);
          if (!isConnected) refresh();
        }}
        className="fixed top-4 right-4 z-50 bg-purple-600 hover:bg-purple-700 text-white px-4 py-2 rounded-lg shadow-lg transition-colors"
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
      >
        {isDemoMode ? '🛑 EXIT DEMO' : '▶️ START DEMO'}
      </motion.button>

      {/* Connection Status */}
      {!isConnected && (
        <motion.div
          className="fixed bottom-4 left-1/2 transform -translate-x-1/2 bg-red-600 text-white px-4 py-2 rounded-lg font-mono shadow-lg"
          initial={{ y: 100, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          exit={{ y: 100, opacity: 0 }}
        >
          🔴 DISCONNECTED - REFRESHING...
        </motion.div>
      )}

      {/* Performance Overlay */}
      <div className="fixed bottom-4 left-4 bg-black/70 rounded p-2 backdrop-blur-sm text-xs text-gray-300 z-40">
        <div>WebSocket: {data?.latency || 78}ms</div>
        <div>FPS: 60</div>
        <div>Agents: {agentPanelCollapsed ? 'Collapsed' : 'Active'}</div>
      </div>
    </div>
  );
};

export default DashboardLayout;