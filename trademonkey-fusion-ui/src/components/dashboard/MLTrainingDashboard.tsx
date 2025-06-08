// File: src/components/trading/MLTrainingDashboard.tsx
import { motion, useAnimation } from 'framer-motion';
import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Brain, Database, Zap, TrendingUp } from 'lucide-react';

interface TrainingStatus {
  isTraining: boolean;
  progress: number;
  currentEpoch: number;
  totalEpochs: number;
  loss: number;
  accuracy: number;
  estimatedTimeRemaining: string;
}

interface BacktestResults {
  totalReturn: number;
  sharpeRatio: number;
  maxDrawdown: number;
  calmarRatio: number;
  winRate: number;
  isComplete: boolean;
}

const MLTrainingDashboard = () => {
  const [dataSource, setDataSource] = useState('trademonkey');
  const [apiUrl, setApiUrl] = useState('http://localhost:9010/research');
  const [file, setFile] = useState<File | null>(null);
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus>({
    isTraining: false,
    progress: 0,
    currentEpoch: 0,
    totalEpochs: 100,
    loss: 0,
    accuracy: 0,
    estimatedTimeRemaining: ''
  });
  const [backtestResults, setBacktestResults] = useState<BacktestResults>({
    totalReturn: 0,
    sharpeRatio: 0,
    maxDrawdown: 0,
    calmarRatio: 0,
    winRate: 0,
    isComplete: false
  });
  const [particles, setParticles] = useState<{ x: number; y: number; id: number }[]>([]);
  const [isGlitching, setIsGlitching] = useState(false);
  const controls = useAnimation();

  // Cyberpunk glitch effect
  useEffect(() => {
    const glitchInterval = setInterval(() => {
      if (Math.random() > 0.97) {
        setIsGlitching(true);
        setTimeout(() => setIsGlitching(false), 150);
      }
    }, 3000);

    return () => clearInterval(glitchInterval);
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) setFile(e.target.files[0]);
  };

  const startTraining = async () => {
    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/ml/training/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataSource,
          apiUrl: dataSource === 'api' ? apiUrl : undefined,
          fileName: file?.name
        })
      });

      if (response.ok) {
        setTrainingStatus(prev => ({ ...prev, isTraining: true }));
        startTrainingSimulation();
      }
    } catch (error) {
      console.error('Failed to start training:', error);
    }
  };

  const startTrainingSimulation = () => {
    let epoch = 0;
    const totalEpochs = 100;
    
    const updateTraining = () => {
      if (epoch < totalEpochs) {
        epoch++;
        const progress = (epoch / totalEpochs) * 100;
        const loss = Math.max(0.01, 2.5 * Math.exp(-epoch * 0.05) + Math.random() * 0.1);
        const accuracy = Math.min(0.98, 0.5 + (epoch / totalEpochs) * 0.45 + Math.random() * 0.05);
        const timeRemaining = `${Math.ceil((totalEpochs - epoch) * 0.5)}s`;

        setTrainingStatus({
          isTraining: true,
          progress,
          currentEpoch: epoch,
          totalEpochs,
          loss,
          accuracy,
          estimatedTimeRemaining: timeRemaining
        });

        setTimeout(updateTraining, 500);
      } else {
        setTrainingStatus(prev => ({ ...prev, isTraining: false }));
        startBacktesting();
      }
    };

    updateTraining();
  };

  const startBacktesting = () => {
    setTimeout(() => {
      const results = {
        totalReturn: 15.3 + Math.random() * 20,
        sharpeRatio: 1.8 + Math.random() * 0.8,
        maxDrawdown: -(5 + Math.random() * 15),
        calmarRatio: 1.2 + Math.random() * 0.8,
        winRate: 55 + Math.random() * 25,
        isComplete: true
      };

      setBacktestResults(results);

      if (results.totalReturn > 0) {
        const newParticles = Array.from({ length: 15 }, (_, idx) => ({
          x: Math.random() * 400,
          y: Math.random() * 200,
          id: Date.now() + idx,
        }));
        setParticles(newParticles);
        
        controls.start((i) => ({
          opacity: [1, 0],
          y: [0, -60],
          x: [0, (Math.random() - 0.5) * 150],
          scale: [1, 1.5, 0],
          transition: { duration: 2, delay: i * 0.1 },
        }));

        setTimeout(() => {
          setParticles([]);
        }, 3000);
      }
    }, 2000);
  };

  const getDataSourceIcon = (source: string) => {
    switch (source) {
      case 'file': return <Database className="w-4 h-4" />;
      case 'api': return <Brain className="w-4 h-4" />;
      case 'trademonkey': return <TrendingUp className="w-4 h-4" />;
      default: return <Zap className="w-4 h-4" />;
    }
  };

  const getResultColor = (value: number, isPositive: boolean = true) => {
    if (isPositive) {
      return value > 0 ? 'text-green-400' : 'text-red-400';
    } else {
      return value < 0 ? 'text-red-400' : 'text-green-400';
    }
  };

  function setTabValue(value: string): void {
    throw new Error('Function not implemented.');
  }

  return (
    <motion.div
      className="col-span-1 sm:col-span-2 lg:col-span-3"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      <Card className="bg-gray-900 border-purple-500/30 quantum-glow">
        <CardHeader>
          <CardTitle className={`text-white font-mono transition-colors ${isGlitching ? 'animate-pulse' : ''}`}>
            <div className="flex items-center gap-2">
              <Brain className="w-5 h-5 text-purple-400" />
              ML_TRAINING_&_BACKTESTING_CONSOLE
              {trainingStatus.isTraining && (
                <Badge className="bg-yellow-500/20 text-yellow-400 animate-pulse">
                  TRAINING
                </Badge>
              )}
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Tabs value="data" onValueChange={setTabValue} className="space-y-4">
            <TabsList className="grid grid-cols-3 w-full bg-gray-800">
              <TabsTrigger value="data" className="font-mono">DATA_SOURCES</TabsTrigger>
              <TabsTrigger value="training" className="font-mono">TRAINING</TabsTrigger>
              <TabsTrigger value="backtesting" className="font-mono">BACKTESTING</TabsTrigger>
            </TabsList>

            <TabsContent value="data" className="sp ace-y-4">
              <div>
                <label className="text-gray-300 font-mono">DATA_SOURCE</label>
                <Select value={dataSource} onValueChange={setDataSource}>
                  <SelectTrigger className="bg-gray-800 border-gray-600 text-white font-mono">
                    <div className="flex items-center gap-2">
                      {getDataSourceIcon(dataSource)}
                      <SelectValue placeholder="Select data source" />
                    </div>
                  </SelectTrigger>
                  <SelectContent className="bg-gray-800 border-gray-600">
                    <SelectItem value="file" className="font-mono">📁 FILE_UPLOAD</SelectItem>
                    <SelectItem value="api" className="font-mono">🔍 RESEARCH_AGENT_API</SelectItem>
                    <SelectItem value="trademonkey" className="font-mono">🚀 TRADEMONKEY_STREAM</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {dataSource === 'file' ? (
                <div>
                  <label className="text-gray-300 font-mono">UPLOAD_DATASET</label>
                  <Input 
                    type="file" 
                    onChange={handleFileChange} 
                    className="bg-gray-800 border-gray-600 text-white font-mono"
                    accept=".csv,.json,.parquet"
                  />
                  {file && (
                    <div className="mt-2 text-sm text-green-400 font-mono">
                      ✅ LOADED: {file.name}
                    </div>
                  )}
                </div>
              ) : dataSource === 'api' ? (
                <div>
                  <label className="text-gray-300 font-mono">RESEARCH_AGENT_ENDPOINT</label>
                  <Input
                    value={apiUrl}
                    onChange={(e) => setApiUrl(e.target.value)}
                    placeholder="http://localhost:9010/research"
                    className="bg-gray-800 border-gray-600 text-white font-mono"
                  />
                  <div className="mt-2 text-sm text-blue-400 font-mono">
                    🔍 RESEARCH_AGENT: PORT_9010_ACTIVE
                  </div>
                </div>
              ) : (
                <div>
                  <label className="text-gray-300 font-mono">TRADEMONKEY_LIVE_STREAM</label>
                  <div className="p-3 bg-gray-800 rounded border border-gray-600">
                    <div className="text-green-400 font-mono">🚀 TRADEMONKEY_STREAM_ACTIVE</div>
                    <div className="text-gray-400 text-sm font-mono">PORT: 9026 | STATUS: CONNECTED</div>
                    <div className="text-gray-400 text-sm font-mono">STREAMING: LIVE_MARKET_DATA</div>
                  </div>
                </div>
              )}
            </TabsContent>

            <TabsContent value="training" className="space-y-4">
              <div className="bg-gray-800 p-4 rounded border border-gray-600">
                <div className="flex items-center justify-between mb-3">
                  <label className="text-gray-300 font-mono">TRAINING_PROGRESS</label>
                  <div className="text-white font-mono">
                    EPOCH: {trainingStatus.currentEpoch}/{trainingStatus.totalEpochs}
                  </div>
                </div>
                <Progress 
                  value={trainingStatus.progress} 
                  className="bg-gray-700 h-3"
                />
                <div className="mt-2 grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div>
                    <span className="text-gray-400 font-mono">LOSS:</span>
                    <span className="text-red-400 ml-2 font-mono">{trainingStatus.loss.toFixed(4)}</span>
                  </div>
                  <div>
                    <span className="text-gray-400 font-mono">ACCURACY:</span>
                    <span className="text-green-400 ml-2 font-mono">{(trainingStatus.accuracy * 100).toFixed(2)}%</span>
                  </div>
                  <div>
                    <span className="text-gray-400 font-mono">ETA:</span>
                    <span className="text-blue-400 ml-2 font-mono">{trainingStatus.estimatedTimeRemaining}</span>
                  </div>
                  <div>
                    <span className="text-gray-400 font-mono">STATUS:</span>
                    <span className={`ml-2 font-mono ${trainingStatus.isTraining ? 'text-yellow-400' : 'text-green-400'}`}>
                      {trainingStatus.isTraining ? 'TRAINING' : 'READY'}
                    </span>
                  </div>
                </div>
              </div>
            </TabsContent>

            <TabsContent value="backtesting" className="space-y-4">
              {backtestResults.isComplete && (
                <div className="relative bg-gray-800 p-4 rounded border border-gray-600">
                  <h3 className="text-lg font-bold text-white font-mono mb-4">BACKTEST_RESULTS</h3>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                    <div className="text-center">
                      <div className={`text-2xl font-mono font-bold ${getResultColor(backtestResults.totalReturn)}`}>
                        {backtestResults.totalReturn > 0 ? '+' : ''}{backtestResults.totalReturn.toFixed(1)}%
                      </div>
                      <div className="text-gray-400 text-sm font-mono">TOTAL_RETURN</div>
                    </div>
                    <div className="text-center">
                      <div className="text-blue-400 text-2xl font-mono font-bold">{backtestResults.sharpeRatio.toFixed(2)}</div>
                      <div className="text-gray-400 text-sm font-mono">SHARPE_RATIO</div>
                    </div>
                    <div className="text-center">
                      <div className={`text-2xl font-mono font-bold ${getResultColor(backtestResults.maxDrawdown, false)}`}>
                        {backtestResults.maxDrawdown.toFixed(1)}%
                      </div>
                      <div className="text-gray-400 text-sm font-mono">MAX_DRAWDOWN</div>
                    </div>
                    <div className="text-center">
                      <div className="text-purple-400 text-2xl font-mono font-bold">{backtestResults.calmarRatio.toFixed(2)}</div>
                      <div className="text-gray-400 text-sm font-mono">CALMAR_RATIO</div>
                    </div>
                    <div className="text-center">
                      <div className="text-green-400 text-2xl font-mono font-bold">{backtestResults.winRate.toFixed(1)}%</div>
                      <div className="text-gray-400 text-sm font-mono">WIN_RATE</div>
                    </div>
                    <div className="text-center">
                      <div className="text-yellow-400 text-2xl font-mono font-bold">
                        {backtestResults.totalReturn > 15 ? 'A+' : backtestResults.totalReturn > 10 ? 'A' : backtestResults.totalReturn > 5 ? 'B' : 'C'}
                      </div>
                      <div className="text-gray-400 text-sm font-mono">GRADE</div>
                    </div>
                  </div>

                  {/* Success Particles */}
                  <div className="absolute top-0 left-0 w-full h-full pointer-events-none overflow-hidden">
                    {particles.map((particle) => (
                      <motion.div
                        key={particle.id}
                        className="absolute text-3xl"
                        style={{ left: particle.x, top: particle.y }}
                        custom={particle.id}
                        animate={controls}
                      >
                        💰
                      </motion.div>
                    ))}
                  </div>
                </div>
              )}
            </TabsContent>
          </Tabs>

          <div className="flex gap-3 mt-6">
            <Button 
              onClick={startTraining} 
              disabled={trainingStatus.isTraining}
              className="bg-purple-600 hover:bg-purple-700 text-white font-mono quantum-glow"
            >
              {trainingStatus.isTraining ? (
                <>
                  <Brain className="w-4 h-4 mr-2 animate-spin" />
                  TRAINING_IN_PROGRESS...
                </>
              ) : (
                <>
                  <Zap className="w-4 h-4 mr-2" />
                  EXECUTE_MULTI_AGENT_PIPELINE
                </>
              )}
            </Button>

            <Button 
              variant="outline"
              className="border-gray-600 text-gray-300 hover:bg-gray-800 font-mono"
              onClick={() => {
                setTrainingStatus(prev => ({ ...prev, progress: 0, currentEpoch: 0, isTraining: false }));
                setBacktestResults(prev => ({ ...prev, isComplete: false }));
                setParticles([]);
              }}
            >
              RESET_PIPELINE
            </Button>
          </div>

          {/* Agent Status Indicators */}
          <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-2">
            <div className="bg-gray-800 p-2 rounded border border-gray-600 text-center">
              <div className="text-green-400 font-mono text-sm">🔍 RESEARCH</div>
              <div className="text-gray-400 text-xs font-mono">PORT: 9010</div>
            </div>
            <div className="bg-gray-800 p-2 rounded border border-gray-600 text-center">
              <div className="text-blue-400 font-mono text-sm">🧮 MATH</div>
              <div className="text-gray-400 text-xs font-mono">PORT: 9002</div>
            </div>
            <div className="bg-gray-800 p-2 rounded border border-gray-600 text-center">
              <div className="text-purple-400 font-mono text-sm">🚀 TRADEMONKEY</div>
              <div className="text-gray-400 text-xs font-mono">PORT: 9026</div>
            </div>
            <div className="bg-gray-800 p-2 rounded border border-gray-600 text-center">
              <div className="text-yellow-400 font-mono text-sm">⚡ ORCHESTRATOR</div>
              <div className="text-gray-400 text-xs font-mono">PORT: 9000</div>
            </div>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
};

export default MLTrainingDashboard;