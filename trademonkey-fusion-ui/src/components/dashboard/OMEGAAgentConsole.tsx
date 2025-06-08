// File: src/components/dashboard/OMEGAAgentConsole.tsx
import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Activity, Brain, Calculator, Search, TrendingUp, Zap, Settings, CheckCircle, AlertCircle, Clock } from 'lucide-react';
import { useAgents } from '@/hooks/useRealtimeData';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Button } from '@/components/ui/button';

interface OMEGAAgent {
  id: string;
  name: string;
  port: number;
  status: 'active' | 'idle' | 'error' | 'busy';
  icon?: React.ReactNode;
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

const OMEGAAgentConsole: React.FC = () => {
  const agents = useAgents();
  const [selectedAgent, setSelectedAgent] = useState<string | null>(null);
  const [networkLoad, setNetworkLoad] = useState(0);
  const [taskLogs, setTaskLogs] = useState<TaskLog[]>([]);
  const [isGlitching, setIsGlitching] = useState(false);

  // Calculate network load from agent data
  useEffect(() => {
    if (agents && Array.isArray(agents)) {
      const avgLoad = agents.reduce((sum, agent) => sum + (agent.load || 0), 0) / agents.length;
      setNetworkLoad(avgLoad || 0);
    }
  }, [agents]);

  // Mock task logs (to be replaced with real data in Phase 3)
  useEffect(() => {
    if (agents && Array.isArray(agents)) {
      const logs = agents.slice(0, 5).map((agent, index) => {
        // Map agent status to TaskLog status
        const getTaskStatus = (agentStatus: string): 'completed' | 'failed' | 'in_progress' => {
          switch (agentStatus) {
            case 'active': return 'completed';
            case 'busy': return 'in_progress';
            case 'error': return 'failed';
            case 'idle': return 'completed';
            default: return 'completed';
          }
        };

        return {
          id: `${agent.id}-${Date.now()}-${index}`,
          agent: agent.name,
          task: agent.currentTask || `Processing ${agent.name.toLowerCase()} data`,
          status: getTaskStatus(agent.status),
          timestamp: new Date(Date.now() - Math.random() * 600000).toLocaleTimeString(),
          duration: agent.responseTime || Math.floor(Math.random() * 2000) + 100,
          confidence: agent.accuracy ? agent.accuracy / 100 : 0.8 + Math.random() * 0.2
        };
      });
      setTaskLogs(logs);
    }
  }, [agents]);

  // Cyberpunk glitch effect
  useEffect(() => {
    const glitchInterval = setInterval(() => {
      if (Math.random() > 0.98) {
        setIsGlitching(true);
        setTimeout(() => setIsGlitching(false), 150);
      }
    }, 2000);

    return () => clearInterval(glitchInterval);
  }, []);

  const getAgentIcon = (agentName: string): React.ReactNode => {
    const iconMap: Record<string, React.ReactNode> = {
      'TradeMonkey': <TrendingUp className="w-4 h-4" />,
      'Research Agent': <Search className="w-4 h-4" />,
      'Math Solver': <Calculator className="w-4 h-4" />,
      'Code Generator': <Brain className="w-4 h-4" />,
      'Orchestrator': <Zap className="w-4 h-4" />
    };
    return iconMap[agentName] || <Activity className="w-4 h-4" />;
  };

  const getStatusColor = (status: string): string => {
    switch (status) {
      case 'active': return 'bg-green-500/20 text-green-400 border-green-500/50';
      case 'busy': return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/50';
      case 'idle': return 'bg-gray-500/20 text-gray-400 border-gray-500/50';
      case 'error': return 'bg-red-500/20 text-red-400 border-red-500/50';
      default: return 'bg-gray-500/20 text-gray-400 border-gray-500/50';
    }
  };

  const getTaskStatusIcon = (status: string): React.ReactNode => {
    switch (status) {
      case 'completed': return <CheckCircle className="w-3 h-3 text-green-400" />;
      case 'failed': return <AlertCircle className="w-3 h-3 text-red-400" />;
      case 'in_progress': return <Clock className="w-3 h-3 text-yellow-400 animate-spin" />;
      default: return <Clock className="w-3 h-3 text-gray-400" />;
    }
  };

  const getLoadColor = (load: number): string => {
    if (load < 30) return 'bg-green-500';
    if (load < 70) return 'bg-yellow-500';
    return 'bg-red-500';
  };

  if (!agents) {
    return (
      <Card className="bg-gray-900 border-purple-500/30">
        <CardHeader>
          <CardTitle className="text-white">🤖 OMEGA Agent Console</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center text-gray-400">Loading agent network...</div>
        </CardContent>
      </Card>
    );
  }

  const agentList = Array.isArray(agents) ? agents : [];

  return (
    <Card className="bg-gray-900 border-purple-500/30">
      <CardHeader className="pb-2">
        <CardTitle className={`flex items-center justify-between text-white transition-colors ${isGlitching ? 'animate-pulse' : ''}`}>
          <div className="flex items-center gap-2">
            <Activity className="w-5 h-5 text-purple-400" />
            <span className="font-mono">OMEGA_AGENT_CONSOLE</span>
            <Badge className="bg-purple-500/20 text-purple-400">
              Network Active
            </Badge>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-400 font-mono">NET_LOAD:</span>
            <div className="w-16 h-2 bg-gray-700 rounded-full overflow-hidden">
              <motion.div
                className={`h-full ${getLoadColor(networkLoad)}`}
                initial={{ width: 0 }}
                animate={{ width: `${networkLoad}%` }}
                transition={{ duration: 0.5 }}
              />
            </div>
            <span className="text-xs text-white font-mono">{networkLoad.toFixed(0)}%</span>
          </div>
        </CardTitle>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Agent Status Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
          {agentList.map((agent, index) => (
            <motion.div
              key={agent.id}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              className={`p-3 rounded border cursor-pointer transition-all quantum-glow hover:border-purple-500/50 ${
                selectedAgent === agent.id ? 'border-purple-500/50 bg-purple-500/5' : 'border-gray-700 bg-gray-800'
              }`}
              onClick={() => setSelectedAgent(selectedAgent === agent.id ? null : agent.id)}
            >
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  {getAgentIcon(agent.name)}
                  <span className="text-white text-sm font-medium font-mono">{agent.name}</span>
                </div>
                <Badge className={`text-xs font-mono ${getStatusColor(agent.status)}`}>
                  {agent.status.toUpperCase()}
                </Badge>
              </div>

              <div className="space-y-2">
                <div className="text-xs text-gray-400 font-mono">PORT: {agent.port}</div>
                
                {agent.currentTask && (
                  <div className="text-xs text-gray-300">
                    <div className="flex items-center gap-1">
                      <motion.div
                        className="w-2 h-2 bg-blue-400 rounded-full"
                        animate={{ opacity: [0.5, 1, 0.5] }}
                        transition={{ repeat: Infinity, duration: 1.5 }}
                      />
                      <span className="font-mono">{agent.currentTask}</span>
                    </div>
                  </div>
                )}

                <div className="flex justify-between items-center text-xs">
                  <span className="text-gray-400 font-mono">LOAD:</span>
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
                        <span className="text-gray-400 font-mono">TASKS:</span>
                        <span className="text-white font-mono">{agent.tasksCompleted}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400 font-mono">UPTIME:</span>
                        <span className="text-green-400 font-mono">{agent.uptime}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400 font-mono">RESPONSE:</span>
                        <span className="text-blue-400 font-mono">{agent.responseTime?.toFixed(0) || 0}ms</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400 font-mono">ACCURACY:</span>
                        <span className="text-green-400 font-mono">{agent.accuracy?.toFixed(1) || 0}%</span>
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
            <h4 className="text-white font-medium font-mono">TASK_ACTIVITY_LOG</h4>
            <Button 
              size="sm" 
              variant="outline" 
              className="text-xs border-gray-600 text-gray-300 hover:bg-gray-800 font-mono"
            >
              <Settings className="w-3 h-3 mr-1" />
              CONFIG
            </Button>
          </div>
          
          <div className="bg-gray-800 rounded border border-gray-700 max-h-48 overflow-y-auto">
            <AnimatePresence>
              {taskLogs.map((task, index) => (
                <motion.div
                  key={task.id}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: 10 }}
                  transition={{ delay: index * 0.05 }}
                  className="flex items-center justify-between p-2 border-b border-gray-700 last:border-b-0 hover:bg-gray-750"
                >
                  <div className="flex items-center gap-2 flex-1">
                    {getTaskStatusIcon(task.status)}
                    <div className="flex-1 min-w-0">
                      <div className="text-white text-xs font-medium truncate font-mono">{task.task}</div>
                      <div className="text-gray-400 text-xs font-mono">{task.agent}</div>
                    </div>
                  </div>
                  
                  <div className="flex items-center gap-2 text-xs">
                    {task.confidence && (
                      <Badge className="bg-blue-500/20 text-blue-400 text-xs font-mono">
                        {(task.confidence * 100).toFixed(0)}%
                      </Badge>
                    )}
                    <span className="text-gray-400 font-mono">{task.timestamp}</span>
                    <span className="text-gray-500 font-mono w-12 text-right">{task.duration}ms</span>
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        </div>

        {/* Network Status */}
        <div className="grid grid-cols-3 gap-4 text-center">
          <div className="bg-gray-800 rounded p-2 border border-gray-700">
            <div className="text-green-400 text-lg font-mono">
              {agentList.filter(a => a.status === 'active').length}
            </div>
            <div className="text-gray-400 text-xs font-mono">ACTIVE_AGENTS</div>
          </div>
          <div className="bg-gray-800 rounded p-2 border border-gray-700">
            <div className="text-blue-400 text-lg font-mono">
              {taskLogs.filter(t => t.status === 'completed').length}
            </div>
            <div className="text-gray-400 text-xs font-mono">TASKS_COMPLETE</div>
          </div>
          <div className="bg-gray-800 rounded p-2 border border-gray-700">
            <div className="text-yellow-400 text-lg font-mono">
              {agentList.length > 0 ? Math.floor(agentList.reduce((sum, a) => sum + (a.responseTime || 0), 0) / agentList.length) : 0}ms
            </div>
            <div className="text-gray-400 text-xs font-mono">AVG_RESPONSE</div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default OMEGAAgentConsole;
