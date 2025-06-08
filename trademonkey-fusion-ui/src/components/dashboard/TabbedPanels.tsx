// File: src/components/dashboard/TabbedPanels.tsx
import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { ChevronUp, ChevronDown, Settings, TrendingUp, Brain, BarChart3 } from 'lucide-react';
import MLTrainingDashboard from './MLTrainingDashboard';

interface TabbedPanelsProps {
  demoMode?: boolean;
}

const TabbedPanels: React.FC<TabbedPanelsProps> = ({ demoMode = false }) => {
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [activeTab, setActiveTab] = useState('ml-training');

  const tabs = [
    {
      id: 'ml-training',
      label: 'ML Training',
      icon: <Brain className="w-4 h-4" />,
      component: <MLTrainingDashboard />
    },
    {
      id: 'backtesting',
      label: 'Backtesting',
      icon: <BarChart3 className="w-4 h-4" />,
      component: <BacktestingPanel demoMode={demoMode} />
    },
    {
      id: 'analytics',
      label: 'Analytics',
      icon: <TrendingUp className="w-4 h-4" />,
      component: <AdvancedAnalytics demoMode={demoMode} />
    },
    {
      id: 'settings',
      label: 'Settings',
      icon: <Settings className="w-4 h-4" />,
      component: <SystemSettings />
    }
  ];

  return (
    <motion.div
      className={`fixed bottom-0 left-0 right-0 z-30 bg-gray-900/95 backdrop-blur-sm border-t border-purple-500/30 transition-all duration-300 ${
        isCollapsed ? 'h-12' : 'h-80'
      }`}
      initial={{ y: 100, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      {/* Collapse/Expand Header */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-gray-700">
        <div className="flex items-center gap-2">
          <span className="text-lg">🛠️</span>
          <span className="text-white font-semibold">Advanced Tools</span>
          <div className="flex gap-1">
            {tabs.map((tab) => (
              <motion.div
                key={tab.id}
                className={`w-2 h-2 rounded-full transition-colors ${
                  activeTab === tab.id ? 'bg-purple-400' : 'bg-gray-600'
                }`}
                whileHover={{ scale: 1.2 }}
              />
            ))}
          </div>
        </div>
        
        <motion.button
          onClick={() => setIsCollapsed(!isCollapsed)}
          className="flex items-center gap-2 px-3 py-1 rounded bg-gray-800 hover:bg-gray-700 text-gray-300 hover:text-white transition-colors"
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          {isCollapsed ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          <span className="text-sm">{isCollapsed ? 'Expand' : 'Collapse'}</span>
        </motion.button>
      </div>

      <AnimatePresence>
        {!isCollapsed && (
          <motion.div
            className="h-full overflow-hidden"
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: '100%' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.3 }}
          >
            <Tabs value={activeTab} onValueChange={setActiveTab} className="h-full flex flex-col">
              <TabsList className="grid grid-cols-4 w-full max-w-md mx-auto bg-gray-800 rounded-none border-b border-gray-700">
                {tabs.map((tab) => (
                  <TabsTrigger
                    key={tab.id}
                    value={tab.id}
                    className="flex items-center gap-2 data-[state=active]:bg-purple-600 data-[state=active]:text-white"
                  >
                    {tab.icon}
                    <span className="hidden sm:inline">{tab.label}</span>
                  </TabsTrigger>
                ))}
              </TabsList>
              
              <div className="flex-1 overflow-auto p-4">
                {tabs.map((tab) => (
                  <TabsContent key={tab.id} value={tab.id} className="h-full m-0">
                    <motion.div
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.2 }}
                      className="h-full"
                    >
                      {tab.component}
                    </motion.div>
                  </TabsContent>
                ))}
              </div>
            </Tabs>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};

// Quick components for the tabs
const BacktestingPanel: React.FC<{ demoMode?: boolean }> = ({ demoMode }) => (
  <Card className="bg-gray-800 border-yellow-500/30 h-full">
    <CardHeader>
      <CardTitle className="text-yellow-400 flex items-center gap-2">
        <BarChart3 className="w-5 h-5" />
        GPU-Accelerated Backtesting
      </CardTitle>
    </CardHeader>
    <CardContent>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="space-y-2">
          <label className="text-gray-300 text-sm">Time Range</label>
          <select className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white">
            <option>Last 30 Days</option>
            <option>Last 90 Days</option>
            <option>Last Year</option>
          </select>
        </div>
        <div className="space-y-2">
          <label className="text-gray-300 text-sm">Strategy</label>
          <select className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white">
            <option>Momentum + Sentiment</option>
            <option>Mean Reversion</option>
            <option>Breakout</option>
          </select>
        </div>
        <div className="space-y-2">
          <label className="text-gray-300 text-sm">Capital</label>
          <input
            type="number"
            defaultValue={10000}
            className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white"
          />
        </div>
      </div>
      <motion.button
        className="mt-4 bg-yellow-600 hover:bg-yellow-700 text-white px-6 py-2 rounded transition-colors"
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
      >
        🚀 Run Backtest
      </motion.button>
    </CardContent>
  </Card>
);

const AdvancedAnalytics: React.FC<{ demoMode?: boolean }> = ({ demoMode }) => (
  <Card className="bg-gray-800 border-blue-500/30 h-full">
    <CardHeader>
      <CardTitle className="text-blue-400 flex items-center gap-2">
        <TrendingUp className="w-5 h-5" />
        Advanced Analytics
      </CardTitle>
    </CardHeader>
    <CardContent>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-gray-700 rounded p-4 text-center">
          <div className="text-2xl text-green-400 font-mono">+24.5%</div>
          <div className="text-gray-400 text-sm">30D Return</div>
        </div>
        <div className="bg-gray-700 rounded p-4 text-center">
          <div className="text-2xl text-blue-400 font-mono">2.1</div>
          <div className="text-gray-400 text-sm">Sharpe Ratio</div>
        </div>
        <div className="bg-gray-700 rounded p-4 text-center">
          <div className="text-2xl text-yellow-400 font-mono">68%</div>
          <div className="text-gray-400 text-sm">Win Rate</div>
        </div>
        <div className="bg-gray-700 rounded p-4 text-center">
          <div className="text-2xl text-red-400 font-mono">-8.2%</div>
          <div className="text-gray-400 text-sm">Max DD</div>
        </div>
      </div>
    </CardContent>
  </Card>
);

const SystemSettings: React.FC = () => (
  <Card className="bg-gray-800 border-gray-500/30 h-full">
    <CardHeader>
      <CardTitle className="text-gray-400 flex items-center gap-2">
        <Settings className="w-5 h-5" />
        System Settings
      </CardTitle>
    </CardHeader>
    <CardContent>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="space-y-4">
          <h3 className="text-white font-semibold">Trading Settings</h3>
          <div className="space-y-2">
            <label className="text-gray-300 text-sm">Risk Level</label>
            <input
              type="range"
              min="1"
              max="10"
              defaultValue="5"
              className="w-full"
            />
          </div>
          <div className="space-y-2">
            <label className="text-gray-300 text-sm">Position Size (%)</label>
            <input
              type="number"
              defaultValue={25}
              className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white"
            />
          </div>
        </div>
        <div className="space-y-4">
          <h3 className="text-white font-semibold">System Settings</h3>
          <div className="space-y-2">
            <label className="text-gray-300 text-sm">Update Interval (ms)</label>
            <input
              type="number"
              defaultValue={2000}
              className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white"
            />
          </div>
          <div className="space-y-2">
            <label className="text-gray-300 text-sm">GPU Memory Limit (MB)</label>
            <input
              type="number"
              defaultValue={8192}
              className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white"
            />
          </div>
        </div>
      </div>
    </CardContent>
  </Card>
);

export default TabbedPanels;