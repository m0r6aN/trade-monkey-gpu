import React, { useState, useMemo } from 'react';
import { FixedSizeList } from 'react-window';
import { motion } from 'framer-motion';
import { useRealtimeData } from '@/hooks/useRealtimeData';

interface ActivityEvent {
  id: string;
  timestamp: string;
  type: 'sentiment' | 'gpu' | 'trade' | 'signal' | 'system';
  message: string;
  severity: 'info' | 'success' | 'warning' | 'error';
  icon: string;
}

type FilterType = ActivityEvent['type'] | 'all';

interface ActivityFeedProps {
  demoMode?: boolean;
}

interface ListItemProps {
  index: number;
  style: React.CSSProperties;
  data: ActivityEvent[];
}

const ActivityFeed: React.FC<ActivityFeedProps> = ({ demoMode = false }) => {
  const { data } = useRealtimeData();
  const [filter, setFilter] = useState<FilterType>('all');

  const events: ActivityEvent[] = useMemo(() => {
    const liveEvents = (data.activities || []).map(event => ({
      id: event.id,
      timestamp: new Date(event.timestamp).toISOString(),
      type: event.type,
      message: event.message,
      severity: event.severity,
      icon: {
        sentiment: '🧠',
        gpu: '⚡',
        trade: '💰',
        signal: '🎯',
        system: '🏥',
      }[event.type] || '📊',
    }));
    return liveEvents
      .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
      .filter(event => filter === 'all' || event.type === filter);
  }, [data.activities, filter]);

  const filterButtons: Array<{ type: FilterType; label: string; emoji: string }> = [
    { type: 'all', label: 'All', emoji: '📊' },
    { type: 'sentiment', label: 'Sentiment', emoji: '🧠' },
    { type: 'gpu', label: 'GPU', emoji: '⚡' },
    { type: 'trade', label: 'Trade', emoji: '💰' },
    { type: 'signal', label: 'Signal', emoji: '🎯' },
    { type: 'system', label: 'System', emoji: '🏥' },
  ];

  const getSeverityColor = (severity: ActivityEvent['severity']): string => {
    switch (severity) {
      case 'success': return 'text-green-400';
      case 'warning': return 'text-yellow-400';
      case 'error': return 'text-red-400';
      default: return 'text-gray-300';
    }
  };

  const ListItem: React.FC<ListItemProps> = ({ index, style, data }) => {
    const event = data[index];
    return (
      <motion.div
        style={style}
        className="flex items-center gap-3 px-2 py-1 hover:bg-gray-800/50 transition-colors"
        initial={{ opacity: 0, x: -10 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ delay: index * 0.05 }}
      >
        <span className="text-lg flex-shrink-0">{event.icon}</span>
        <div className="flex-1 min-w-0">
          <span className={`text-sm ${getSeverityColor(event.severity)} block truncate`}>
            {event.message}
          </span>
        </div>
        <span className="text-xs text-gray-500 flex-shrink-0">
          {new Date(event.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
        </span>
      </motion.div>
    );
  };

  return (
    <motion.div 
      className="p-4 rounded-lg bg-gray-900 border border-gray-700"
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.5 }}
    >
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-bold text-white flex items-center gap-2">
          📈 Activity Feed
          <span className="text-sm text-gray-400 font-normal">
            ({events.length} events)
          </span>
        </h2>
        <div className="flex gap-1 overflow-x-auto">
          {filterButtons.map((button) => (
            <motion.button
              key={button.type}
              onClick={() => setFilter(button.type)}
              className={`px-3 py-1 rounded text-xs whitespace-nowrap transition-all ${
                filter === button.type 
                  ? 'bg-blue-600 text-white shadow-lg' 
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              {button.emoji} {button.label}
            </motion.button>
          ))}
        </div>
      </div>
      <div className="border border-gray-700 rounded bg-gray-800/50">
        {events.length > 0 ? (
          <FixedSizeList
            height={300}
            width="100%"
            itemCount={events.length}
            itemSize={48}
            itemData={events}
          >
            {ListItem}
          </FixedSizeList>
        ) : (
          <div className="h-300 flex items-center justify-center text-gray-500">
            <div className="text-center">
              <span className="text-4xl block mb-2">🔍</span>
              No events found for "{filter}" filter
            </div>
          </div>
        )}
      </div>
    </motion.div>
  );
};

export default ActivityFeed;
