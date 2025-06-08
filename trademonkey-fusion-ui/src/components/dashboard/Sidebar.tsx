// File: src/components/dashboard/Sidebar.tsx
'use client';

import { motion } from 'framer-motion';
import { useState } from 'react';
import { Home, TrendingUp, BarChart3, Settings, Menu, X } from 'lucide-react';
import { cn } from '@/lib/utils';

const Sidebar = () => {
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [activeItem, setActiveItem] = useState('dashboard');

  const menuItems = [
    { id: 'dashboard', icon: Home, label: 'Dashboard', href: '/' },
    { id: 'positions', icon: TrendingUp, label: 'Positions', href: '/positions' },
    { id: 'backtest', icon: BarChart3, label: 'Backtest', href: '/backtest' },
    { id: 'settings', icon: Settings, label: 'Settings', href: '/settings' },
  ];

  return (
    <motion.aside
      className={cn(
        "h-screen bg-gray-900 border-r border-gray-700 flex flex-col transition-all duration-300",
        isCollapsed ? "w-16" : "w-64"
      )}
      initial={{ x: -100, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      {/* Header */}
      <div className="p-4 border-b border-gray-700">
        <div className="flex items-center justify-between">
          {!isCollapsed && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.2 }}
              className="flex items-center gap-2"
            >
              <span className="text-2xl">🐵</span>
              <span className="text-white font-bold">TradeMonkey</span>
            </motion.div>
          )}
          <motion.button
            onClick={() => setIsCollapsed(!isCollapsed)}
            className="p-2 rounded-lg hover:bg-gray-800 text-gray-400 hover:text-white transition-colors"
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.95 }}
          >
            {isCollapsed ? <Menu className="w-4 h-4" /> : <X className="w-4 h-4" />}
          </motion.button>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-4">
        <div className="space-y-2">
          {menuItems.map((item, index) => (
            <motion.a
              key={item.id}
              href={item.href}
              onClick={(e) => {
                e.preventDefault();
                setActiveItem(item.id);
              }}
              className={cn(
                "flex items-center gap-3 p-3 rounded-lg transition-all duration-200",
                "hover:bg-gray-800 hover:text-white group",
                activeItem === item.id
                  ? "bg-blue-600 text-white"
                  : "text-gray-400"
              )}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.1 }}
              whileHover={{ x: 4 }}
            >
              <item.icon className="w-5 h-5 flex-shrink-0" />
              {!isCollapsed && (
                <motion.span
                  className="font-medium"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.1 }}
                >
                  {item.label}
                </motion.span>
              )}
              
              {/* Active indicator */}
              {activeItem === item.id && (
                <motion.div
                  className="ml-auto w-2 h-2 bg-white rounded-full"
                  layoutId="activeIndicator"
                  transition={{ duration: 0.2 }}
                />
              )}
            </motion.a>
          ))}
        </div>
      </nav>

      {/* Footer */}
      <div className="p-4 border-t border-gray-700">
        {!isCollapsed && (
          <motion.div
            className="text-center text-xs text-gray-500"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.3 }}
          >
            <div>TradeMonkey Fusion v1.0</div>
            <div className="text-green-400">🟢 OMEGA Network Active</div>
          </motion.div>
        )}
      </div>
    </motion.aside>
  );
};

export default Sidebar;