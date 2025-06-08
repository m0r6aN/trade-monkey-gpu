// File: src/components/ui/progress.tsx
import React from 'react';
import { cn } from '@/lib/utils';

interface ProgressProps {
  value: number;
  className?: string;
  indicatorClassName?: string;
}

export const Progress: React.FC<ProgressProps> = ({ 
  value, 
  className, 
  indicatorClassName 
}) => {
  return (
    <div className={cn(
      'relative h-4 w-full overflow-hidden rounded-full bg-gray-700',
      className
    )}>
      <div
        className={cn(
          'h-full w-full flex-1 bg-primary transition-all',
          'bg-gradient-to-r from-blue-600 to-blue-400',
          indicatorClassName
        )}
        style={{ transform: `translateX(-${100 - (value || 0)}%)` }}
      />
    </div>
  );
};