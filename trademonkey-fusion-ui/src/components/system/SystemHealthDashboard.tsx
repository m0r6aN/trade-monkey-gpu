import React from 'react';
import { motion } from 'framer-motion';
import { useRealtimeData } from '@/hooks/useRealtimeData';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { AlertTriangle, Heart } from 'lucide-react';

interface HealthData {
  health_score: number;
  uptime: number;
  error_rates: number[];
}

const SystemHealthDashboard: React.FC = () => {
  const { data: healthData, error, isLoading } = useRealtimeData<HealthData>('health', '/api/system/health');

  if (isLoading) {
    return (
      <Card className="p-4 bg-background quantum-glow">
        <div className="text-center text-muted-foreground">Loading system health...</div>
      </Card>
    );
  }

  if (error) {
    return (
      <Card className="p-4 bg-background quantum-glow">
        <div className="text-center text-destructive">Error loading system health: {error.message}</div>
      </Card>
    );
  }

  return (
    <Card className="bg-background border-border quantum-glow">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-quantum-blue transition-colors">
          <Heart className="w-5 h-5" />
          <span className="font-mono">SYSTEM_HEALTH</span>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="p-3 border border-border rounded-lg">
          <div className="text-sm text-quantum-blue font-mono mb-2 transition-colors">HEALTH_SCORE</div>
          <motion.div 
            className={`text-2xl font-mono ${healthData?.health_score > 95 ? 'text-matrix-green' : 'text-laser-red'} transition-colors`}
            animate={{ opacity: [1, 0.8, 1] }}
            transition={{ duration: 2, repeat: Infinity }}
          >
            {healthData?.health_score.toFixed(1)}%
          </motion.div>
        </div>
        <div className="p-3 border border-border rounded-lg">
          <div className="text-sm text-quantum-blue font-mono mb-2 transition-colors">UPTIME</div>
          <div className="text-2xl font-mono text-matrix-green transition-colors">
            {healthData?.uptime.toFixed(2)}%
          </div>
        </div>
        <div className="p-3 border border-border rounded-lg">
          <div className="text-sm text-quantum-blue font-mono mb-2 transition-colors">ERROR_RATE</div>
          <div className="text-sm text-laser-red transition-colors">
            {(healthData?.error_rates[0] * 100).toFixed(2)}% (last period)
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default SystemHealthDashboard;