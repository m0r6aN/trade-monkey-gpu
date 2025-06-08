// File: src/components/dashboard/HeaderBar.tsx (Enhanced)
"use client";

import { motion } from 'framer-motion';
import { useState, useEffect } from 'react';
import { Switch } from '@/components/ui/switch';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { Badge } from '@/components/ui/badge';
import { Settings, Wifi, WifiOff, Activity, TrendingUp } from 'lucide-react';
import { cn } from '@/lib/utils';

interface HeaderBarProps {
  isConnected?: boolean;
  demoMode?: boolean;
  onDemoToggle?: () => void;
  marketMode?: 'live' | 'paper' | 'demo';
  onMarketModeChange?: (mode: 'live' | 'paper' | 'demo') => void;
}

const HeaderBar: React.FC<HeaderBarProps> = ({
  isConnected = true,
  demoMode = false,
  onDemoToggle,
  marketMode = 'demo',
  onMarketModeChange
}) => {
  const [isMatrixTheme, setIsMatrixTheme] = useState(false);
  const [isClient, setIsClient] = useState(false);
  const [notifications, setNotifications] = useState(0);

  // Hydration fix - only access localStorage on client
  useEffect(() => {
    setIsClient(true);
    
    // Load saved preferences
    const savedTheme = localStorage.getItem('trademonkey-theme');
    const savedMarketMode = localStorage.getItem('trademonkey-market-mode');
    
    if (savedTheme === 'matrix') {
      setIsMatrixTheme(true);
      document.documentElement.classList.add('theme-matrix');
    }
    
    if (savedMarketMode && onMarketModeChange) {
      onMarketModeChange(savedMarketMode as 'live' | 'paper' | 'demo');
    }
  }, [onMarketModeChange]);

  // Save theme preference
  useEffect(() => {
    if (!isClient) return;
    
    localStorage.setItem('trademonkey-theme', isMatrixTheme ? 'matrix' : 'quantum');
    
    if (isMatrixTheme) {
      document.documentElement.classList.add('theme-matrix');
      document.documentElement.classList.remove('theme-quantum');
    } else {
      document.documentElement.classList.add('theme-quantum');
      document.documentElement.classList.remove('theme-matrix');
    }
  }, [isMatrixTheme, isClient]);

  // Save market mode preference
  useEffect(() => {
    if (!isClient) return;
    localStorage.setItem('trademonkey-market-mode', marketMode);
  }, [marketMode, isClient]);

  // Simulate notifications
  useEffect(() => {
    const interval = setInterval(() => {
      if (Math.random() > 0.7) {
        setNotifications(prev => prev + 1);
      }
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  const getMarketModeColor = (mode: string) => {
    switch (mode) {
      case 'live': return 'bg-red-500 text-white';
      case 'paper': return 'bg-yellow-500 text-black';
      case 'demo': return 'bg-green-500 text-white';
      default: return 'bg-gray-500 text-white';
    }
  };

  // Prevent hydration mismatch by not rendering dynamic content until client-side
  if (!isClient) {
    return (
      <header className="bg-gray-900/95 backdrop-blur-sm p-4 flex justify-between items-center sticky top-0 z-50 border-b border-gray-700">
        <div className="flex items-center gap-2">
          <span className="text-2xl">🦍</span>
          <h1 className="text-xl font-bold text-white">TradeMonkey Fusion</h1>
        </div>
        <div className="flex items-center gap-4">
          <div className="w-8 h-8 bg-gray-800 rounded-full animate-pulse" />
        </div>
      </header>
    );
  }

  return (
    <motion.header
      initial={{ y: -64, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      className="bg-gray-900/95 backdrop-blur-sm p-4 flex justify-between items-center sticky top-0 z-50 border-b border-gray-700"
    >
      {/* Left Section - Logo & Status */}
      <div className="flex items-center gap-4">
        <motion.div 
          className="flex items-center gap-2"
          whileHover={{ scale: 1.05 }}
        >
          <motion.span 
            className="text-2xl"
            animate={{ 
              rotate: demoMode ? [0, 10, -10, 0] : 0,
              scale: demoMode ? [1, 1.1, 1] : 1 
            }}
            transition={{ duration: 2, repeat: demoMode ? Infinity : 0 }}
          >
            🦍
          </motion.span>
          <h1 className={`text-xl font-bold transition-colors ${
            isMatrixTheme ? 'text-green-400' : 'text-white'
          }`}>
            TradeMonkey Fusion
          </h1>
          <Badge variant="outline" className="text-xs">
            v1.0
          </Badge>
        </motion.div>

        {/* Connection Status */}
        <motion.div 
          className="flex items-center gap-2"
          animate={{ 
            scale: isConnected ? 1 : [1, 1.1, 1],
            opacity: isConnected ? 1 : [1, 0.5, 1]
          }}
          transition={{ duration: 1, repeat: isConnected ? 0 : Infinity }}
        >
          {isConnected ? (
            <Wifi className="w-4 h-4 text-green-400" />
          ) : (
            <WifiOff className="w-4 h-4 text-red-400" />
          )}
          <span className={`text-xs font-mono ${
            isConnected ? 'text-green-400' : 'text-red-400'
          }`}>
            {isConnected ? 'CONNECTED' : 'DISCONNECTED'}
          </span>
        </motion.div>

        {/* Market Mode Selector */}
        <div className="flex items-center gap-2">
          <span className="text-gray-400 text-sm">Mode:</span>
          <select 
            value={marketMode}
            onChange={(e) => onMarketModeChange?.(e.target.value as any)}
            className="bg-gray-800 border border-gray-600 rounded px-2 py-1 text-white text-sm"
          >
            <option value="demo">Demo</option>
            <option value="paper">Paper</option>
            <option value="live">Live</option>
          </select>
          <Badge className={getMarketModeColor(marketMode)}>
            {marketMode.toUpperCase()}
          </Badge>
        </div>
      </div>

      {/* Right Section - Controls & Profile */}
      <div className="flex items-center gap-4">
        {/* Theme Toggle */}
        <div className="flex items-center gap-2">
          <span className="text-gray-300 text-sm">Matrix Theme</span>
          <Switch
            checked={isMatrixTheme}
            onCheckedChange={setIsMatrixTheme}
            className="data-[state=checked]:bg-green-500"
          />
        </div>

        {/* Demo Mode Toggle */}
        <motion.button
          onClick={onDemoToggle}
          className={`px-3 py-1 rounded text-sm font-medium transition-all ${
            demoMode 
              ? 'bg-red-600 hover:bg-red-700 text-white' 
              : 'bg-purple-600 hover:bg-purple-700 text-white'
          }`}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          {demoMode ? '🛑 Stop Demo' : '▶️ Demo Mode'}
        </motion.button>

        {/* Notifications */}
        <motion.div 
          className="relative"
          whileHover={{ scale: 1.1 }}
        >
          <Activity className="w-5 h-5 text-gray-400 hover:text-white cursor-pointer" />
          {notifications > 0 && (
            <motion.span
              className="absolute -top-1 -right-1 bg-red-500 text-white text-xs rounded-full w-4 h-4 flex items-center justify-center"
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ type: "spring", stiffness: 500 }}
            >
              {notifications > 9 ? '9+' : notifications}
            </motion.span>
          )}
        </motion.div>

        {/* Settings */}
        <motion.button
          className="p-2 rounded hover:bg-gray-800 text-gray-400 hover:text-white transition-colors"
          whileHover={{ rotate: 90 }}
        >
          <Settings className="w-5 h-5" />
        </motion.button>

        {/* User Avatar */}
        <motion.div whileHover={{ scale: 1.1 }}>
          <Avatar className="w-8 h-8">
            <AvatarImage src="/user.png" alt="User" />
            <AvatarFallback className="bg-purple-600 text-white">TM</AvatarFallback>
          </Avatar>
        </motion.div>
      </div>
    </motion.header>
  );
};

export default HeaderBar;