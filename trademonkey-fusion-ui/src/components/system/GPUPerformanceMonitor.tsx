import React from 'react';
import { motion } from 'framer-motion';
import { useRealtimeData } from '@/hooks/useRealtimeData';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Zap, Cpu } from 'lucide-react';

interface GPUData {
  memory_usage: number;
  processing_speed: number;
  queue_throughput: number;
}

const GPUPerformanceMonitor: React.FC = () => {
  const { data: gpuData, error, isLoading } = useRealtimeData<GPUData>('gpu', '/api/gpu/metrics');

  if (isLoading) {
    return (
      <Card className="p-4 bg-background matrix-glow">
        <div className="text-center text-muted-foreground">Loading GPU metrics...</div>
      </Card>
    );
  }

  if (error) {
    return (
      <Card className="p-4 bg-background matrix-glow">
        <div className="text-center text-destructive">Error loading GPU metrics: {error.message}</div>
      </Card>
    );
  }

  return (
    <Card className="bg-background border-border matrix-glow">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-quantum-blue transition-colors">
          <Cpu className="w-5 h-5" />
          <span className="font-mono">GPU_PERFORMANCE</span>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="p-3 border border-border rounded-lg">
          <div className="text-sm text-quantum-blue font-mono mb-2 transition-colors">MEMORY_USAGE</div>
          <Progress 
            value={gpuData?.memory_usage || 0} 
            className="h-3 bg-muted border border-border"
          />
          <div className="text-sm text-matrix-green mt-1 transition-colors">
            {gpuData?.memory_usage.toFixed(1)}%
          </div>
        </div>
        <div className="p-3 border border-border rounded-lg">
          <div className="text-sm text-quantum-blue font-mono mb-2 transition-colors">PROCESSING_SPEED</div>
          <motion.div 
            className="text-2xl font-mono text-matrix-green"
            animate={{ scale: [1, 1.05, 1] }}
            transition={{ duration: 1, repeat: Infinity }}
          >
            {gpuData?.processing_speed.toFixed(0)} ops/s
          </motion.div>
        </div>
        <div className="p-3 border border-border rounded-lg">
          <div className="text-sm text-quantum-blue font-mono mb-2 transition-colors">QUEUE_THROUGHPUT</div>
          <div className="text-2xl font-mono text-matrix-green transition-colors">
            {gpuData?.queue_throughput.toFixed(0)} tx/s
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default GPUPerformanceMonitor;