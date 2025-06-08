import React from 'react';
import { motion } from 'framer-motion';
import { useRealtimeData } from '@/hooks/useRealtimeData';
import { ActivityEvent } from '@/types/trading';
import { Card } from '@/components/ui/card';

const ActivityFeed: React.FC = () => {
  const { data: activities, error, isLoading } = useRealtimeData<ActivityEvent[]>('activities', '/api/activities/feed');

  if (isLoading) {
    return (
      <Card className="p-4 bg-background matrix-glow">
        <div className="text-center text-muted-foreground">Loading activities...</div>
      </Card>
    );
  }

  if (error) {
    return (
      <Card className="p-4 bg-background matrix-glow">
        <div className="text-center text-destructive">Error loading activities: {error.message}</div>
      </Card>
    );
  }

  return (
    <Card className="p-4 bg-background matrix-glow">
      <h2 className="text-lg font-semibold text-foreground mb-4">Activity Feed</h2>
      <div className="space-y-2 max-h-[400px] overflow-y-auto">
        {activities?.map((event) => (
          <motion.div
            key={event.event_id}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
            className="p-2 rounded bg-muted text-muted-foreground text-sm"
          >
            <div className="flex justify-between">
              <span>{event.description}</span>
              <span className="text-xs opacity-75">{new Date(event.timestamp).toLocaleTimeString()}</span>
            </div>
            {event.metadata && (
              <div className="text-xs mt-1 opacity-75">
                {JSON.stringify(event.metadata, null, 2)}
              </div>
            )}
          </motion.div>
        ))}
      </div>
    </Card>
  );
};

export default ActivityFeed;