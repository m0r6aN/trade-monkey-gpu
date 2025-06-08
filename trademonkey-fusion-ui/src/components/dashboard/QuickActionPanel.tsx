// File: src/components/dashboard/QuickActionPanel.tsx
import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { TrendingUp, TrendingDown, X, DollarSign, Target, Shield } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';

interface QuickActionPanelProps {
  demoMode?: boolean;
}

const QuickActionPanel: React.FC<QuickActionPanelProps> = ({ demoMode = false }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [selectedAction, setSelectedAction] = useState<'buy' | 'sell' | 'close' | null>(null);
  const [amount, setAmount] = useState(1000);

  const quickActions = [
    {
      id: 'buy',
      label: 'Quick Buy',
      icon: <TrendingUp className="w-5 h-5" />,
      color: 'green',
      bgColor: 'bg-green-600 hover:bg-green-700',
      description: 'Market buy order'
    },
    {
      id: 'sell',
      label: 'Quick Sell',
      icon: <TrendingDown className="w-5 h-5" />,
      color: 'red',
      bgColor: 'bg-red-600 hover:bg-red-700',
      description: 'Market sell order'
    },
    {
      id: 'close',
      label: 'Close All',
      icon: <X className="w-5 h-5" />,
      color: 'orange',
      bgColor: 'bg-orange-600 hover:bg-orange-700',
      description: 'Emergency close all positions'
    }
  ];

  const positions = [
    { symbol: 'BTC/USD', side: 'LONG', size: 0.5, pnl: +1250.50, current: 65432.76 },
    { symbol: 'ETH/USD', side: 'SHORT', size: 2.1, pnl: -145.23, current: 3456.89 }
  ];

  return (
    <>
      {/* Floating Action Button */}
      <motion.div
        className="fixed right-6 top-1/2 transform -translate-y-1/2 z-40"
        initial={{ x: 100, opacity: 0 }}
        animate={{ x: 0, opacity: 1 }}
        transition={{ duration: 0.5, delay: 0.2 }}
      >
        <motion.button
          onClick={() => setIsOpen(!isOpen)}
          className={`w-14 h-14 rounded-full shadow-lg flex items-center justify-center text-white transition-all ${
            isOpen ? 'bg-purple-600' : 'bg-blue-600 hover:bg-blue-700'
          }`}
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.95 }}
          animate={{
            boxShadow: isOpen 
              ? '0 0 25px rgba(147, 51, 234, 0.5)' 
              : '0 0 20px rgba(59, 130, 246, 0.3)'
          }}
        >
          <motion.div
            animate={{ rotate: isOpen ? 45 : 0 }}
            transition={{ duration: 0.2 }}
          >
            {isOpen ? <X className="w-6 h-6" /> : <DollarSign className="w-6 h-6" />}
          </motion.div>
        </motion.button>
      </motion.div>

      {/* Side Panel */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            className="fixed right-6 top-1/2 transform -translate-y-1/2 z-30 mr-16"
            initial={{ x: 100, opacity: 0, scale: 0.8 }}
            animate={{ x: 0, opacity: 1, scale: 1 }}
            exit={{ x: 100, opacity: 0, scale: 0.8 }}
            transition={{ duration: 0.3 }}
          >
            <Card className="w-80 bg-gray-900/95 backdrop-blur-sm border border-purple-500/30">
              <CardContent className="p-4">
                <div className="space-y-4">
                  {/* Header */}
                  <div className="flex items-center justify-between">
                    <h3 className="text-white font-semibold flex items-center gap-2">
                      ⚡ Quick Actions
                    </h3>
                    <span className="text-xs text-gray-400">
                      {demoMode ? 'DEMO MODE' : 'LIVE TRADING'}
                    </span>
                  </div>

                  {/* Quick Action Buttons */}
                  <div className="grid grid-cols-3 gap-2">
                    {quickActions.map((action) => (
                      <motion.button
                        key={action.id}
                        onClick={() => setSelectedAction(selectedAction === action.id ? null : action.id as any)}
                        className={`p-3 rounded text-center transition-all ${action.bgColor} ${
                          selectedAction === action.id ? 'ring-2 ring-white' : ''
                        }`}
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                      >
                        <div className="flex flex-col items-center gap-1">
                          {action.icon}
                          <span className="text-xs font-medium">{action.label}</span>
                        </div>
                      </motion.button>
                    ))}
                  </div>

                  {/* Action Form */}
                  <AnimatePresence>
                    {selectedAction && (
                      <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        exit={{ opacity: 0, height: 0 }}
                        className="space-y-3 border-t border-gray-700 pt-3"
                      >
                        {selectedAction !== 'close' && (
                          <>
                            <div>
                              <label className="text-gray-400 text-sm">Amount ($)</label>
                              <input
                                type="number"
                                value={amount}
                                onChange={(e) => setAmount(Number(e.target.value))}
                                className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-white font-mono"
                              />
                            </div>
                            <div>
                              <label className="text-gray-400 text-sm">Symbol</label>
                              <select className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-white">
                                <option>BTC/USD</option>
                                <option>ETH/USD</option>
                                <option>ADA/USD</option>
                              </select>
                            </div>
                          </>
                        )}
                        
                        <Button
                          className={`w-full ${
                            selectedAction === 'buy' ? 'bg-green-600 hover:bg-green-700' :
                            selectedAction === 'sell' ? 'bg-red-600 hover:bg-red-700' :
                            'bg-orange-600 hover:bg-orange-700'
                          }`}
                          onClick={() => {
                            // Execute action
                            setSelectedAction(null);
                            setIsOpen(false);
                          }}
                        >
                          {selectedAction === 'close' ? '🛑 Close All Positions' : 
                           `${selectedAction === 'buy' ? '📈' : '📉'} Execute ${selectedAction.toUpperCase()}`}
                        </Button>
                      </motion.div>
                    )}
                  </AnimatePresence>

                  {/* Current Positions */}
                  <div className="space-y-2">
                    <h4 className="text-gray-400 text-sm flex items-center gap-2">
                      <Target className="w-4 h-4" />
                      Active Positions
                    </h4>
                    {positions.map((pos, index) => (
                      <motion.div
                        key={pos.symbol}
                        className="bg-gray-800 rounded p-2 text-xs"
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: index * 0.1 }}
                      >
                        <div className="flex justify-between items-center">
                          <div>
                            <span className="text-white font-medium">{pos.symbol}</span>
                            <span className={`ml-2 px-1 rounded text-xs ${
                              pos.side === 'LONG' ? 'bg-green-600' : 'bg-red-600'
                            }`}>
                              {pos.side}
                            </span>
                          </div>
                          <div className="text-right">
                            <div className={`font-mono ${pos.pnl > 0 ? 'text-green-400' : 'text-red-400'}`}>
                              ${pos.pnl > 0 ? '+' : ''}{pos.pnl.toFixed(2)}
                            </div>
                            <div className="text-gray-400">{pos.size} @ ${pos.current}</div>
                          </div>
                        </div>
                      </motion.div>
                    ))}
                  </div>

                  {/* Risk Indicator */}
                  <div className="bg-gray-800 rounded p-2">
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-gray-400 flex items-center gap-1">
                        <Shield className="w-3 h-3" />
                        Portfolio Risk
                      </span>
                      <span className="text-yellow-400 font-mono">34%</span>
                    </div>
                    <div className="w-full bg-gray-700 rounded-full h-1 mt-1">
                      <motion.div
                        className="bg-yellow-400 h-1 rounded-full"
                        initial={{ width: 0 }}
                        animate={{ width: '34%' }}
                        transition={{ duration: 0.5 }}
                      />
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
};

export default QuickActionPanel;