import { motion, useAnimation } from 'framer-motion';
import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';

const MLTrainingDashboard = () => {
  const [dataSource, setDataSource] = useState('file');
  const [apiUrl, setApiUrl] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [backtestProgress, setBacktestProgress] = useState(0);
  const [backtestResults, setBacktestResults] = useState({
    return: 0,
    sharpe: 0,
    drawdown: 0,
    calmar: 0,
    winRate: 0,
  });
  const [particles, setParticles] = useState<{ x: number; y: number; id: number }[]>([]);
  const controls = useAnimation();

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) setFile(e.target.files[0]);
  };

  const runMultiAgentPipeline = async () => {
    setTrainingProgress(0);
    setBacktestProgress(0);

    for (let i = 0; i <= 100; i += 10) {
      await new Promise((resolve) => setTimeout(resolve, 300));
      setTrainingProgress(i);
    }

    for (let i = 0; i <= 100; i += 10) {
      await new Promise((resolve) => setTimeout(resolve, 300));
      setBacktestProgress(i);
    }

    const results = {
      return: 25.3,
      sharpe: 2.1,
      drawdown: -15.7,
      calmar: 1.6,
      winRate: 68.4,
    };
    setBacktestResults(results);

    if (results.return > 0) {
      const newParticles = Array.from({ length: 10 }, (_, idx) => ({
        x: Math.random() * 300,
        y: Math.random() * 200,
        id: idx,
      }));
      setParticles(newParticles);
      controls.start((i) => ({
        opacity: [1, 0],
        y: [0, -50],
        x: [0, (Math.random() - 0.5) * 100],
        transition: { duration: 1, delay: i * 0.1 },
      }));
    }
  };

  return (
    <motion.div
      className="col-span-1 sm:col-span-2 lg:col-span-3"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      <Card className="holo-card">
        <CardHeader>
          <CardTitle className="text-white">ML Training & Backtesting Dashboard</CardTitle>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="data" className="space-y-4">
            <TabsList className="grid grid-cols-3 w-full">
              <TabsTrigger value="data" className="cyber-button">Data Sources</TabsTrigger>
              <TabsTrigger value="training" className="cyber-button">Training</TabsTrigger>
              <TabsTrigger value="backtesting" className="cyber-button">Backtesting</TabsTrigger>
            </TabsList>
            <TabsContent value="data" className="space-y-4">
              <div>
                <label className="text-gray-300">Data Source</label>
                <Select value={dataSource} onValueChange={setDataSource}>
                  <SelectTrigger className="cyber-input">
                    <SelectValue placeholder="Select data source" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="file">File Upload</SelectItem>
                    <SelectItem value="api">API Call (Research Agent)</SelectItem>
                    <SelectItem value="trademonkey">TradeMonkey Stream</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              {dataSource === 'file' ? (
                <div>
                  <label className="text-gray-300">Upload Dataset</label>
                  <Input type="file" onChange={handleFileChange} className="cyber-input" />
                </div>
              ) : dataSource === 'api' ? (
                <div>
                  <label className="text-gray-300">API Endpoint (Research Agent: 9010)</label>
                  <Input
                    value={apiUrl}
                    onChange={(e) => setApiUrl(e.target.value)}
                    placeholder="Enter API URL"
                    className="cyber-input"
                  />
                </div>
              ) : (
                <div>
                  <label className="text-gray-300">TradeMonkey Stream (Port: 9026)</label>
                  <p className="text-gray-400">Streaming live market data...</p>
                </div>
              )}
            </TabsContent>
            <TabsContent value="training" className="space-y-4">
              <div>
                <label className="text-gray-300">Training Progress</label>
                <Progress value={trainingProgress} className="bg-gray-700" />
                <p className="text-gray-400">{trainingProgress}% (Research Agent fetching data...)</p>
              </div>
            </TabsContent>
            <TabsContent value="backtesting" className="space-y-4">
              <div>
                <label className="text-gray-300">Backtesting Progress</label>
                <Progress value={backtestProgress} className="bg-gray-700" />
                <p className="text-gray-400">{backtestProgress}% (TradeMonkey analyzing...)</p>
              </div>
              {backtestResults.return !== 0 && (
                <div className="relative mt-4 p-4 bg-gray-800 rounded-lg">
                  <h3 className="text-lg font-bold">Backtest Results</h3>
                  <p>Total Return: {backtestResults.return}%</p>
                  <p>Sharpe Ratio: {backtestResults.sharpe}</p>
                  <p>Max Drawdown: {backtestResults.drawdown}%</p>
                  <p>Calmar Ratio: {backtestResults.calmar}</p>
                  <p>Win Rate: {backtestResults.winRate}%</p>
                  <div className="absolute top-0 left-0 w-full h-full pointer-events-none">
                    {particles.map((particle) => (
                      <motion.div
                        key={particle.id}
                        className="absolute text-yellow-400 text-2xl"
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
          <Button onClick={runMultiAgentPipeline} className="cyber-button mt-4">
            Run Multi-Agent Pipeline
          </Button>
        </CardContent>
      </Card>
    </motion.div>
  );
};

export default MLTrainingDashboard;