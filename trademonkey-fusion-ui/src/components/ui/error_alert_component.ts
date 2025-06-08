// File: src/components/ui/ErrorAlert.tsx
import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { AlertTriangle, Wifi, WifiOff, RefreshCw, X } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { LoadingSpinner, GlitchText } from './cyberpunk-animations';

interface ErrorAlertProps {
  error: Error | string | null;
  isConnected?: boolean;
  onRetry?: () => void;
  onDismiss?: () => void;
  autoRetry?: boolean;
  retryInterval?: number;
}

export const ErrorAlert: React.FC<ErrorAlertProps> = ({
  error,
  isConnected = true,
  onRetry,
  onDismiss,
  autoRetry = false,
  retryInterval = 5000
}) => {
  const [isRetrying, setIsRetrying] = useState(false);
  const [retryCount, setRetryCount] = useState(0);
  const [timeToRetry, setTimeToRetry] = useState(0);

  // Auto retry logic
  useEffect(() => {
    if (autoRetry && error && onRetry) {
      const interval = setInterval(() => {
        setTimeToRetry(prev => {
          if (prev <= 1) {
            handleRetry();
            return retryInterval / 1000;
          }
          return prev - 1;
        });
      }, 1000);

      setTimeToRetry(retryInterval / 1000);
      
      return () => clearInterval(interval);
    }
  }, [error, autoRetry, onRetry, retryInterval]);

  const handleRetry = async () => {
    if (!onRetry) return;
    
    setIsRetrying(true);
    setRetryCount(prev => prev + 1);
    
    try {
      await onRetry();
    } catch (err) {
      console.error('Retry failed:', err);
    } finally {
      setIsRetrying(false);
    }
  };

  const getErrorType = () => {
    if (!isConnected) return 'connection';
    if (typeof error === 'string' && error.includes('fetch')) return 'network';
    if (typeof error === 'string' && error.includes('timeout')) return 'timeout';
    return 'general';
  };

  const getErrorMessage = () => {
    const errorType = getErrorType();
    const errorText = typeof error === 'string' ? error : error?.message || 'Unknown error';
    
    switch (errorType) {
      case 'connection':
        return 'WebSocket connection lost. Attempting to reconnect...';
      case 'network':
        return 'Network error. Check your connection and try again.';
      case 'timeout':
        return 'Request timed out. The server may be busy.';
      default:
        return errorText;
    }
  };

  const getErrorIcon = () => {
    const errorType = getErrorType();
    
    switch (errorType) {
      case 'connection':
        return <WifiOff className="w-5 h-5" />;
      case 'network':
        return <Wifi className="w-5 h-5" />;
      default:
        return <AlertTriangle className="w-5 h-5" />;
    }
  };

  const getErrorColor = () => {
    const errorType = getErrorType();
    
    switch (errorType) {
      case 'connection':
        return 'border-yellow-500/50 bg-yellow-900/20';
      case 'network':
        return 'border-blue-500/50 bg-blue-900/20';
      default:
        return 'border-red-500/50 bg-red-900/20';
    }
  };

  if (!error) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, y: -50, scale: 0.9 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        exit={{ opacity: 0, y: -50, scale: 0.9 }}
        transition={{ duration: 0.3, ease: 'easeOut' }}
        className="fixed top-4 right-4 z-50 max-w-md"
      >
        <Card className={`border-2 ${getErrorColor()} backdrop-blur-sm`}>
          <CardContent className="p-4">
            <div className="flex items-start gap-3">
              <motion.div
                className="flex-shrink-0 text-red-400"
                animate={{ 
                  rotate: [0, 5, -5, 0],
                  scale: [1, 1.1, 1]
                }}
                transition={{ 
                  duration: 0.5, 
                  repeat: Infinity, 
                  repeatDelay: 2 
                }}
              >
                {getErrorIcon()}
              </motion.div>
              
              <div className="flex-1 min-w-0">
                <GlitchText className="text-red-400 font-semibold text-sm mb-1">
                  SYSTEM_ERROR
                </GlitchText>
                
                <p className="text-gray-300 text-sm mb-3">
                  {getErrorMessage()}
                </p>
                
                {retryCount > 0 && (
                  <p className="text-gray-400 text-xs mb-2">
                    Retry attempts: {retryCount}
                  </p>
                )}
                
                {autoRetry && timeToRetry > 0 && (
                  <div className="flex items-center gap-2 mb-3">
                    <div className="flex-1 bg-gray-700 rounded-full h-1">
                      <motion.div
                        className="bg-blue-500 h-1 rounded-full"
                        animate={{ width: `${((retryInterval / 1000) - timeToRetry) / (retryInterval / 1000) * 100}%` }}
                        transition={{ duration: 0.1 }}
                      />
                    </div>
                    <span className="text-gray-400 text-xs font-mono">
                      {timeToRetry}s
                    </span>
                  </div>
                )}
                
                <div className="flex items-center gap-2">
                  {onRetry && (
                    <Button
                      onClick={handleRetry}
                      disabled={isRetrying}
                      size="sm"
                      className="bg-blue-600 hover:bg-blue-700 text-white font-mono text-xs"
                    >
                      {isRetrying ? (
                        <>
                          <LoadingSpinner size="sm" />
                          <span className="ml-2">RETRYING</span>
                        </>
                      ) : (
                        <>
                          <RefreshCw className="w-3 h-3 mr-1" />
                          RETRY
                        </>
                      )}
                    </Button>
                  )}
                  
                  {onDismiss && (
                    <Button
                      onClick={onDismiss}
                      size="sm"
                      variant="outline"
                      className="border-gray-600 text-gray-300 hover:bg-gray-800 font-mono text-xs"
                    >
                      <X className="w-3 h-3 mr-1" />
                      DISMISS
                    </Button>
                  )}
                </div>
              </div>
            </div>
            
            {/* Glitch effect overlay */}
            <motion.div
              className="absolute inset-0 bg-red-500/10 pointer-events-none"
              animate={{ 
                opacity: [0, 0.3, 0],
                scale: [1, 1.02, 1]
              }}
              transition={{ 
                duration: 0.2, 
                repeat: Infinity, 
                repeatDelay: 3 
              }}
            />
          </CardContent>
        </Card>
      </motion.div>
    </AnimatePresence>
  );
};

// Connection Status Indicator
export const ConnectionStatus: React.FC<{
  isConnected: boolean;
  lastUpdate?: Date;
  className?: string;
}> = ({ isConnected, lastUpdate, className = '' }) => {
  const [timeSinceUpdate, setTimeSinceUpdate] = useState<string>('');

  useEffect(() => {
    if (!lastUpdate) return;

    const interval = setInterval(() => {
      const now = new Date();
      const diff = Math.floor((now.getTime() - lastUpdate.getTime()) / 1000);
      
      if (diff < 60) {
        setTimeSinceUpdate(`${diff}s ago`);
      } else if (diff < 3600) {
        setTimeSinceUpdate(`${Math.floor(diff / 60)}m ago`);
      } else {
        setTimeSinceUpdate(`${Math.floor(diff / 3600)}h ago`);
      }
    }, 1000);

    return () => clearInterval(interval);
  }, [lastUpdate]);

  return (
    <motion.div
      className={`flex items-center gap-2 text-xs font-mono ${className}`}
      animate={isConnected ? {} : { opacity: [1, 0.5, 1] }}
      transition={{ duration: 1, repeat: isConnected ? 0 : Infinity }}
    >
      <motion.div
        className={`w-2 h-2 rounded-full ${
          isConnected ? 'bg-green-400' : 'bg-red-400'
        }`}
        animate={isConnected ? { scale: [1, 1.2, 1] } : {}}
        transition={{ duration: 2, repeat: Infinity }}
      />
      
      <span className={isConnected ? 'text-green-400' : 'text-red-400'}>
        {isConnected ? 'CONNECTED' : 'DISCONNECTED'}
      </span>
      
      {lastUpdate && (
        <span className="text-gray-400">
          • {timeSinceUpdate}
        </span>
      )}
    </motion.div>
  );
};

// Global Error Boundary Component
export const CyberpunkErrorBoundary: React.FC<{
  children: React.ReactNode;
  fallback?: React.ReactNode;
}> = ({ children, fallback }) => {
  const [hasError, setHasError] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    const handleError = (event: ErrorEvent) => {
      setHasError(true);
      setError(new Error(event.message));
    };

    const handleUnhandledRejection = (event: PromiseRejectionEvent) => {
      setHasError(true);
      setError(new Error(event.reason?.message || 'Promise rejection'));
    };

    window.addEventListener('error', handleError);
    window.addEventListener('unhandledrejection', handleUnhandledRejection);

    return () => {
      window.removeEventListener('error', handleError);
      window.removeEventListener('unhandledrejection', handleUnhandledRejection);
    };
  }, []);

  const handleReset = () => {
    setHasError(false);
    setError(null);
    window.location.reload();
  };

  if (hasError) {
    return fallback || (
      <div className="min-h-screen bg-gray-950 flex items-center justify-center p-4">
        <Card className="max-w-md w-full border-red-500/50 bg-red-900/20">
          <CardContent className="p-6 text-center">
            <motion.div
              className="text-red-400 mb-4"
              animate={{ 
                rotate: [0, 5, -5, 0],
                scale: [1, 1.1, 1]
              }}
              transition={{ 
                duration: 0.5, 
                repeat: Infinity, 
                repeatDelay: 2 
              }}
            >
              <AlertTriangle className="w-16 h-16 mx-auto" />
            </motion.div>
            
            <GlitchText className="text-red-400 text-xl font-bold mb-2">
              SYSTEM_FAILURE
            </GlitchText>
            
            <p className="text-gray-300 mb-4">
              The trading system has encountered a critical error and needs to restart.
            </p>
            
            <p className="text-gray-400 text-sm mb-6 font-mono">
              Error: {error?.message || 'Unknown system error'}
            </p>
            
            <Button
              onClick={handleReset}
              className="bg-red-600 hover:bg-red-700 text-white font-mono"
            >
              <RefreshCw className="w-4 h-4 mr-2" />
              RESTART_SYSTEM
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  return <>{children}</>;
};