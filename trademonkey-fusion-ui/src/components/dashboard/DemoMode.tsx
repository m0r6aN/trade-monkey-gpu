// File: D:\Repos\trade-monkey-lite\trade-monkey-gpu\trademonkey-fusion-ui\src\components\dashboard\DemoMode.tsx
import React, { useEffect, useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface ScenarioData {
  sentiment: {
    score: number;
    confidence: number;
    trend: 'up' | 'down' | 'sideways';
    cryptoRatio: number;
    slangStats: { [key: string]: number };
  };
  gpuMetrics: {
    memoryUsage: number;
    processingSpeed: number;
    queueThroughput: number;
  };
  systemHealth: {
    healthScore: number;
    uptime: number;
    errorRates: number[];
  };
}

type ScenarioType = 'bull' | 'bear' | 'mixed';

interface DemoModeProps {
  onScenarioChange?: (scenario: ScenarioType, data: ScenarioData) => void;
}

const DemoMode: React.FC<DemoModeProps> = ({ onScenarioChange }) => {
  const [currentScenario, setCurrentScenario] = useState<ScenarioType>('bull');
  const [isActive, setIsActive] = useState(false);
  const [cycleProgress, setCycleProgress] = useState(0);

  const sentimentMocks: Record<ScenarioType, ScenarioData> = {
    bull: {
      sentiment: { 
        score: 0.85, 
        confidence: 0.92, 
        trend: 'up', 
        cryptoRatio: 0.89,
        slangStats: { 'WAGMI': 50, 'Diamond Hands': 30, 'To The Moon': 25, 'NGMI': 5 }
      },
      gpuMetrics: { 
        memoryUsage: 76, 
        processingSpeed: 340, 
        queueThroughput: 2450 
      },
      systemHealth: { 
        healthScore: 99, 
        uptime: 99.9, 
        errorRates: [0.01, 0.02, 0.01, 0.0, 0.01] 
      }
    },
    bear: {
      sentiment: { 
        score: -0.75, 
        confidence: 0.88, 
        trend: 'down', 
        cryptoRatio: 0.85,
        slangStats: { 'NGMI': 40, 'Paper Hands': 25, 'Rugpull': 20, 'WAGMI': 10 }
      },
      gpuMetrics: { 
        memoryUsage: 82, 
        processingSpeed: 400, 
        queueThroughput: 2000 
      },
      systemHealth: { 
        healthScore: 95, 
        uptime: 99.5, 
        errorRates: [0.03, 0.04, 0.02, 0.05, 0.03] 
      }
    },
    mixed: {
      sentiment: { 
        score: 0.45, 
        confidence: 0.84, 
        trend: 'sideways', 
        cryptoRatio: 0.87,
        slangStats: { 'WAGMI': 25, 'NGMI': 20, 'Probably Nothing': 15, 'HODL': 18 }
      },
      gpuMetrics: { 
        memoryUsage: 75, 
        processingSpeed: 350, 
        queueThroughput: 2300 
      },
      systemHealth: { 
        healthScore: 97, 
        uptime: 99.7, 
        errorRates: [0.02, 0.01, 0.01, 0.02, 0.01] 
      }
    }
  };

  const getNextScenario = useCallback((current: ScenarioType): ScenarioType => {
    switch (current) {
      case 'bull': return 'bear';
      case 'bear': return 'mixed';
      case 'mixed': return 'bull';
      default: return 'bull';
    }
  }, []);

  const getScenarioEmoji = (scenario: ScenarioType): string => {
    switch (scenario) {
      case 'bull': return '🐂';
      case 'bear': return '🐻';
      case 'mixed': return '🦀';
      default: return '📊';
    }
  };

  const getScenarioColor = (scenario: ScenarioType): string => {
    switch (scenario) {
      case 'bull': return 'text-green-400';
      case 'bear': return 'text-red-400';
      case 'mixed': return 'text-yellow-400';
      default: return 'text-gray-400';
    }
  };

  // Auto-cycle scenarios every 45 seconds with progress indicator
  useEffect(() => {
    if (!isActive) {
      setCycleProgress(0);
      return;
    }
    
    const cycleInterval = 45000; // 45 seconds
    const progressInterval = 100; // Update progress every 100ms
    
    let progressCounter = 0;
    const maxProgress = cycleInterval / progressInterval;
    
    const progressTimer = setInterval(() => {
      progressCounter++;
      setCycleProgress((progressCounter / maxProgress) * 100);
      
      if (progressCounter >= maxProgress) {
        setCurrentScenario(prev => getNextScenario(prev));
        progressCounter = 0;
      }
    }, progressInterval);
    
    return () => clearInterval(progressTimer);
  }, [isActive, getNextScenario]);

  // Notify parent component of scenario changes
  useEffect(() => {
    if (isActive && onScenarioChange) {
      const mockData = sentimentMocks[currentScenario];
      onScenarioChange(currentScenario, mockData);
    }
  }, [currentScenario, isActive, onScenarioChange, sentimentMocks]);

  const handleManualScenarioChange = (scenario: ScenarioType) => {
    setCurrentScenario(scenario);
    setCycleProgress(0); // Reset progress when manually changing
  };

  const toggleDemo = () => {
    setIsActive(!isActive);
    if (!isActive) {
      setCycleProgress(0);
    }
  };

  return (
    <>
      {/* Demo Control Panel */}
      <motion.div 
        className="fixed top-4 right-4 z-50"
        initial={{ opacity: 0, scale: 0.8 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5 }}
      >
        <motion.div 
          className="p-4 rounded-lg bg-purple-900/90 border border-purple-500 backdrop-blur-sm"
          whileHover={{ scale: 1.02 }}
        >
          <div className="flex items-center gap-3 mb-3">
            <motion.button
              onClick={toggleDemo}
              className={`px-4 py-2 rounded font-semibold transition-all ${
                isActive 
                  ? 'bg-red-600 hover:bg-red-700 text-white' 
                  : 'bg-green-600 hover:bg-green-700 text-white'
              }`}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              {isActive ? '⏹️ Stop Demo' : '▶️ Start Demo'}
            </motion.button>
            
            <span className="text-purple-200 text-sm font-semibold">
              🎭 Demo Mode
            </span>
          </div>
          
          <AnimatePresence>
            {isActive && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                className="space-y-3"
              >
                {/* Scenario Buttons */}
                <div className="flex gap-2">
                  {(['bull', 'bear', 'mixed'] as ScenarioType[]).map((scenario) => (
                    <motion.button
                      key={scenario}
                      onClick={() => handleManualScenarioChange(scenario)}
                      className={`px-3 py-1 rounded text-sm transition-all ${
                        currentScenario === scenario
                          ? 'bg-purple-600 text-white'
                          : 'bg-purple-800 text-purple-200 hover:bg-purple-700'
                      }`}
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                    >
                      {getScenarioEmoji(scenario)} {scenario}
                    </motion.button>
                  ))}
                </div>
                
                {/* Current Status */}
                <div className="text-sm text-purple-200">
                  <div className="flex items-center gap-2 mb-2">
                    <motion.div 
                      className="w-2 h-2 bg-purple-400 rounded-full"
                      animate={{ opacity: [1, 0.3, 1] }}
                      transition={{ duration: 1, repeat: Infinity }}
                    />
                    <span className={`font-semibold ${getScenarioColor(currentScenario)}`}>
                      {getScenarioEmoji(currentScenario)} {currentScenario.toUpperCase()} Market
                    </span>
                  </div>
                  
                  {/* Progress Bar */}
                  <div className="w-full bg-purple-800 rounded-full h-2 mb-2">
                    <motion.div
                      className="bg-purple-400 h-2 rounded-full"
                      style={{ width: `${cycleProgress}%` }}
                      transition={{ duration: 0.1 }}
                    />
                  </div>
                  
                  <div className="text-xs opacity-75">
                    Next cycle in {Math.ceil((100 - cycleProgress) * 0.45)}s
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>
      </motion.div>
      
      {/* Scenario Overlay */}
      <AnimatePresence>
        {isActive && (
          <motion.div
            className="fixed bottom-4 left-4 z-40 p-4 rounded-lg bg-black/80 border border-blue-500 backdrop-blur-sm"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
          >
            <div className={`font-semibold ${getScenarioColor(currentScenario)}`}>
              🎭 Simulating {getScenarioEmoji(currentScenario)} {currentScenario.toUpperCase()} Market Scenario
            </div>
            <div className="text-sm text-gray-300 mt-1">
              Sentiment: {sentimentMocks[currentScenario].sentiment.score > 0 ? '+' : ''}{sentimentMocks[currentScenario].sentiment.score.toFixed(3)} | 
              GPU: {sentimentMocks[currentScenario].gpuMetrics.memoryUsage}% | 
              Health: {sentimentMocks[currentScenario].systemHealth.healthScore}%
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
};

export default DemoMode;