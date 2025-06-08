// File: src/components/dashboard/OMEGAAgentConsole.tsx (Updated for Side Panel)
import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Activity, Brain, Calculator, Search, TrendingUp, Zap, Settings, CheckCircle, AlertCircle, Clock, ChevronRight, ChevronLeft } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Button } from '@/components/ui/button';

interface OMEGAAgent {
  id: string;
  name: string;
  port: number;
  status: 'active' | 'idle' | 'error' | 'busy';
  icon: React.ReactNode;
  currentTask?: string;
  tasksCompleted: number;
  uptime: string;
  responseTime: number;
  accuracy: number;
  load: number;
}

interface TaskLog {
  id: string;
  agent: string;
  task: string;
  status: 'completed' | 'failed' | 'in_progress';
  timestamp: string;
  duration: number;
  confidence?: number;
}

interface OMEGAAgentConsoleProps {
  isCollapsed?: boolean;
  onToggle?: () => void;
}

const OMEGAAgentConsole: React.FC<OMEGAAgentConsoleProps> = ({ 
  isCollapsed = false, 
  onToggle 
}) => {
  const [agents, setAgents] = useState<OMEGAAgent[]>([
    {
      id: 'trademonkey',
      name: 'TradeMonkey',
      port: 9026,
      status: 'active',
      icon: <TrendingUp className="w-4 h-4" />,
      currentTask: 'Analyzing BTC sentiment',
      tasksCompleted: 147,
      uptime: '99.8%',
      responseTime: 245,
      accuracy: 92.5,
      load: 67
    },
    {
      id: 'research',
      name: 'Research Agent',
      port: 9010,
      status: 'busy',
      icon: <Search className="w-4 h-4" />,
      currentTask: 'Fetching market news',
      tasksCompleted: 89,
      uptime: '98.2%',
      responseTime: 1250,
      accuracy: 88.7,
      load: 84
    },
    {
      id: 'math_solver',
      name: 'Math Solver',
      port: 9002,
      status: 'idle',
      icon: <Calculator className="w-4 h-4" />,
      currentTask: undefined,
      tasksCompleted: 203,
      uptime: '99.9%',
      responseTime: 89,
      accuracy: 99.1,
      load: 23
    },
    {
      id: 'code_generator',
      name: 'Code Generator',
      port: 9014,
      status: 'active',
      icon: <Brain className="w-4 h-4" />,
      currentTask: 'Generating trading strategy',
      tasksCompleted: 56,
      uptime: '97.5%',
      responseTime: 3420,
      accuracy: 85.3,
      load: 91
    },
    {
      id: 'orchestrator',
      name: 'Orchestrator',
      port: 9000,
      status: 'active',
      icon: <Zap className="w-4 h-4" />,
      currentTask: 'Coordinating workflow',
      tasksCompleted: 312,
      uptime: '99.9%',
      responseTime: 156,
      accuracy: 94.8,
      load: 45
    }
  ]);

  const [taskLogs, setTaskLogs] = useState<TaskLog[]>([
    {
      id: '1',
      agent: 'TradeMonkey',
      task: 'BTC Sentiment Analysis',
      status: 'completed',
      timestamp: '14:32:15',
      duration: 245,
      confidence: 0.92
    },
    {
      id: '2',
      agent: 'Research Agent',
      task: 'Market News Fetch',
      status: 'in_progress',
      timestamp: '14:31:58',
      duration: 1250,
      confidence: 0.88
    },
    {
      id: '3',
      agent: 'Math Solver',
      task: 'Portfolio Optimization',
      status: 'completed',
      timestamp: '14:31:45',
      duration: 89,
      confidence: 0.99
    }
  ]);

  const [selectedAgent, setSelectedAgent] = useState<string | null>(null);
  const [networkLoad, setNetworkLoad] = useState(65);

  // Simulate real-time updates
  useEffect(() => {
    const interval = setInterval(() => {
      setAgents(prev => prev.map(agent => ({
        ...agent,
        responseTime: agent.responseTime + (Math.random() - 0.5) * 20,
        load: Math.max(0, Math.min(100, agent.load + (Math.random() - 0.5) * 10)),
        tasksCompleted: agent.status === 'active' && Math.random() > 0.95 ? agent.tasksCompleted + 1 : agent.tasksCompleted
      })));

      setNetworkLoad(prev => Math.max(0, Math.min(100, prev + (Math.random() - 0.5) * 5)));
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'bg-green-500/20 text-green-400 border-green-500/50';
      case 'busy': return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/50';
      case 'idle': return 'bg-gray-500/20 text-gray-400 border-gray-500/50';
      case 'error': return 'bg-red-500/20 text-red-400 border-red-500/50';
      default: return 'bg-gray-500/20 text-gray-400 border-gray-500/50';
    }
  };

  const getTaskStatusIcon = (status: string) => {
    switch (status) {
      case 'completed': return <CheckCircle className="w-3 h-3 text-green-400" />;
      case 'failed': return <AlertCircle className="w-3 h-3 text-red-400" />;
      case 'in_progress': return <Clock className="w-3 h-3 text-yellow-400 animate-spin" />;
      default: return <Clock className="w-3 h-3 text-gray-400" />;
    }
  };

  const getLoadColor = (load: number) => {
    if (load < 30) return 'bg-green-500';
    if (load < 70) return 'bg-yellow-500';
    return 'bg-red-500';
  };

  return (
    <motion.div
      className={`fixed left-0 top-0 bottom-0 z-30 bg-gray-900/95 backdrop-blur-sm border-r border-purple-500/30 transition-all duration-300 ${
        isCollapsed ? 'w-16' : 'w-80'
      }`}
      initial={{ x: -320, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      {/* Toggle Button */}
      <motion.button
        onClick={onToggle}
        className="absolute -right-10 top-1/2 transform -translate-y-1/2 w-8 h-16 bg-purple-600 hover:bg-purple-700 rounded-r-lg flex items-center justify-center text-white transition-colors"
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
      >
        {isCollapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
      </motion.button>

      <div className="h-full overflow-hidden flex flex-col">
        {/* Header */}
        <div className="p-4 border-b border-gray-700">
          <div className={`flex items-center gap-2 ${isCollapsed ? 'justify-center' : 'justify-between'}`}>
            {!isCollapsed && (
              <div className="flex items-center gap-2">
                <Activity className="w-5 h-5 text-purple-400" />
                <span className="text-white font-semibold">OMEGA Network</span>
                <Badge className="bg-purple-500/20 text-purple-400">
                  Active
                </Badge>
              </div>
            )}
            
            {isCollapsed ? (
              <Activity className="w-6 h-6 text-purple-400" />
            ) : (
              <div className="flex items-center gap-2">
                <span className="text-xs text-gray-400">Load:</span>
                <div className="w-12 h-2 bg-gray-700 rounded-full overflow-hidden">
                  <motion.div
                    className={`h-full ${getLoadColor(networkLoad)}`}
                    initial={{ width: 0 }}
                    animate={{ width: `${networkLoad}%` }}
                    transition={{ duration: 0.5 }}
                  />
                </div>
                <span className="text-xs text-white font-mono">{networkLoad.toFixed(0)}%</span>
              </div>
            )}
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-auto p-4 space-y-4">
          <AnimatePresence>
            {isCollapsed ? (
              // Collapsed View - Just status indicators
              <div className="space-y-3">
                {agents.map((agent, index) => (
                  <motion.div
                    key={agent.id}
                    className="flex flex-col items-center gap-1"
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.1 }}
                  >
                    <div className={`w-8 h-8 rounded-full border flex items-center justify-center ${getStatusColor(agent.status)}`}>
                      {agent.icon}
                    </div>
                    <div className="w-2 h-2 bg-gray-600 rounded-full">
                      <motion.div
                        className={`w-full h-full rounded-full ${getLoadColor(agent.load)}`}
                        animate={{ opacity: [0.5, 1, 0.5] }}
                        transition={{ repeat: Infinity, duration: 2 }}
                      />
                    </div>
                  </motion.div>
                ))}
              </div>
            ) : (
              // Expanded View - Full console
              <>
                {/* Agent Grid */}
                <div className="space-y-3">
                  <h4 className="text-white font-medium text-sm">Active Agents</h4>
                  {agents.map((agent, index) => (
                    <motion.div
                      key={agent.id}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: index * 0.05 }}
                      className={`p-3 rounded border cursor-pointer transition-all hover:border-purple-500/50 ${
                        selectedAgent === agent.id ? 'border-purple-500/50 bg-purple-500/5' : 'border-gray-700 bg-gray-800'
                      }`}
                      onClick={() => setSelectedAgent(selectedAgent === agent.id ? null : agent.id)}
                    >
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center gap-2">
                          {agent.icon}
                          <span className="text-white text-sm font-medium">{agent.name}</span>
                        </div>
                        <Badge className={`text-xs ${getStatusColor(agent.status)}`}>
                          {agent.status.toUpperCase()}
                        </Badge>
                      </div>

                      <div className="space-y-2">
                        <div className="text-xs text-gray-400">Port: {agent.port}</div>
                        
                        {agent.currentTask && (
                          <div className="text-xs text-gray-300">
                            <div className="flex items-center gap-1">
                              <motion.div
                                className="w-2 h-2 bg-blue-400 rounded-full"
                                animate={{ opacity: [0.5, 1, 0.5] }}
                                transition={{ repeat: Infinity, duration: 1.5 }}
                              />
                              <span className="truncate">{agent.currentTask}</span>
                            </div>
                          </div>
                        )}

                        <div className="flex justify-between items-center text-xs">
                          <span className="text-gray-400">Load:</span>
                          <div className="flex items-center gap-1">
                            <div className="w-8 h-1 bg-gray-700 rounded-full overflow-hidden">
                              <motion.div
                                className={`h-full ${getLoadColor(agent.load)}`}
                                animate={{ width: `${agent.load}%` }}
                                transition={{ duration: 0.3 }}
                              />
                            </div>
                            <span className="text-white font-mono w-8">{agent.load}%</span>
                          </div>
                        </div>

                        <AnimatePresence>
                          {selectedAgent === agent.id && (
                            <motion.div
                              initial={{ opacity: 0, height: 0 }}
                              animate={{ opacity: 1, height: 'auto' }}
                              exit={{ opacity: 0, height: 0 }}
                              className="space-y-1 text-xs border-t border-gray-700 pt-2 mt-2"
                            >
                              <div className="flex justify-between">
                                <span className="text-gray-400">Tasks:</span>
                                <span className="text-white">{agent.tasksCompleted}</span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-gray-400">Uptime:</span>
                                <span className="text-green-400">{agent.uptime}</span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-gray-400">Response:</span>
                                <span className="text-blue-400">{agent.responseTime.toFixed(0)}ms</span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-gray-400">Accuracy:</span>
                                <span className="text-green-400">{agent.accuracy.toFixed(1)}%</span>
                              </div>
                            </motion.div>
                          )}
                        </AnimatePresence>
                      </div>
                    </motion.div>
                  ))}
                </div>

                {/* Task Log */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <h4 className="text-white font-medium text-sm">Recent Tasks</h4>
                    <Button 
                      size="sm" 
                      variant="outline" 
                      className="text-xs border-gray-600 text-gray-300 hover:bg-gray-800"
                    >
                      <Settings className="w-3 h-3 mr-1" />
                      Config
                    </Button>
                  </div>
                  
                  <div className="bg-gray-800 rounded border border-gray-700 max-h-32 overflow-y-auto">
                    <AnimatePresence>
                      {taskLogs.slice(0, 5).map((task, index) => (
                        <motion.div
                          key={task.id}
                          initial={{ opacity: 0, x: -10 }}
                          animate={{ opacity: 1, x: 0 }}
                          exit={{ opacity: 0, x: 10 }}
                          transition={{ delay: index * 0.05 }}
                          className="flex items-center justify-between p-2 border-b border-gray-700 last:border-b-0 hover:bg-gray-750"
                        >
                          <div className="flex items-center gap-2 flex-1 min-w-0">
                            {getTaskStatusIcon(task.status)}
                            <div className="flex-1 min-w-0">
                              <div className="text-white text-xs font-medium truncate">{task.task}</div>
                              <div className="text-gray-400 text-xs truncate">{task.agent}</div>
                            </div>
                          </div>
                          
                          <div className="flex flex-col items-end text-xs">
                            {task.confidence && (
                              <Badge className="bg-blue-500/20 text-blue-400 text-xs mb-1">
                                {(task.confidence * 100).toFixed(0)}%
                              </Badge>
                            )}
                            <span className="text-gray-400 font-mono">{task.timestamp}</span>
                          </div>
                        </motion.div>
                      ))}
                    </AnimatePresence>
                  </div>
                </div>

                {/* Network Stats */}
                <div className="grid grid-cols-2 gap-2 text-center">
                  <div className="bg-gray-800 rounded p-2">
                    <div className="text-green-400 text-lg font-mono">
                      {agents.filter(a => a.status === 'active').length}
                    </div>
                    <div className="text-gray-400 text-xs">Active</div>
                  </div>
                  <div className="bg-gray-800 rounded p-2">
                    <div className="text-blue-400 text-lg font-mono">
                      {taskLogs.filter(t => t.status === 'completed').length}
                    </div>
                    <div className="text-gray-400 text-xs">Completed</div>
                  </div>
                </div>
              </>
            )}
          </AnimatePresence>
        </div>
      </div>
    </motion.div>
  );
};

export default OMEGAAgentConsole;