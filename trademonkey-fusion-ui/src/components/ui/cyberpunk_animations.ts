// File: src/components/ui/cyberpunk-animations.tsx
import { motion } from 'framer-motion';
import React from 'react';

export const GlitchText: React.FC<{ children: React.ReactNode; className?: string }> = ({ 
  children, 
  className = '' 
}) => (
  <motion.div
    className={`font-mono ${className}`}
    animate={{
      textShadow: [
        '0 0 0px rgba(0, 255, 255, 0)',
        '2px 0 0px rgba(255, 0, 0, 0.5), -2px 0 0px rgba(0, 255, 255, 0.5)',
        '0 0 0px rgba(0, 255, 255, 0)'
      ]
    }}
    transition={{ duration: 0.1, repeat: Infinity, repeatDelay: 3 }}
  >
    {children}
  </motion.div>
);

export const PulseGlow: React.FC<{ children: React.ReactNode; color?: string }> = ({ 
  children, 
  color = 'purple' 
}) => (
  <motion.div
    animate={{
      boxShadow: [
        `0 0 5px rgba(139, 92, 246, 0.2)`,
        `0 0 20px rgba(139, 92, 246, 0.6)`,
        `0 0 5px rgba(139, 92, 246, 0.2)`
      ]
    }}
    transition={{ duration: 2, repeat: Infinity }}
    className="rounded"
  >
    {children}
  </motion.div>
);

export const SuccessParticles: React.FC<{ trigger: boolean; onComplete?: () => void }> = ({ 
  trigger, 
  onComplete 
}) => {
  const particles = Array.from({ length: 8 }, (_, i) => i);

  return trigger ? (
    <div className="absolute inset-0 pointer-events-none overflow-hidden">
      {particles.map((i) => (
        <motion.div
          key={i}
          className="absolute text-2xl"
          initial={{ 
            x: '50%', 
            y: '50%', 
            scale: 0, 
            opacity: 1 
          }}
          animate={{ 
            x: `${50 + (Math.cos(i * 45 * Math.PI / 180) * 100)}%`,
            y: `${50 + (Math.sin(i * 45 * Math.PI / 180) * 100)}%`,
            scale: [0, 1.5, 0],
            opacity: [1, 1, 0]
          }}
          transition={{ 
            duration: 1.5, 
            delay: i * 0.1,
            ease: 'easeOut'
          }}
          onAnimationComplete={() => i === particles.length - 1 && onComplete?.()}
        >
          💰
        </motion.div>
      ))}
    </div>
  ) : null;
};

export const LoadingSpinner: React.FC<{ size?: 'sm' | 'md' | 'lg' }> = ({ size = 'md' }) => {
  const sizeClasses = {
    sm: 'w-4 h-4',
    md: 'w-6 h-6', 
    lg: 'w-8 h-8'
  };

  return (
    <motion.div
      className={`${sizeClasses[size]} border-2 border-purple-500 border-t-transparent rounded-full`}
      animate={{ rotate: 360 }}
      transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
    />
  );
};

export const MatrixRain: React.FC<{ intensity?: number }> = ({ intensity = 20 }) => {
  const characters = '01アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン';
  const drops = Array.from({ length: intensity }, (_, i) => i);

  return (
    <div className="absolute inset-0 pointer-events-none overflow-hidden">
      {drops.map((i) => (
        <motion.div
          key={i}
          className="absolute text-green-400 text-xs font-mono opacity-60"
          style={{ left: `${(i / intensity) * 100}%` }}
          animate={{
            y: ['-10%', '110%'],
            opacity: [0, 1, 0]
          }}
          transition={{
            duration: Math.random() * 3 + 2,
            repeat: Infinity,
            delay: Math.random() * 2,
            ease: 'linear'
          }}
        >
          {characters.charAt(Math.floor(Math.random() * characters.length))}
        </motion.div>
      ))}
    </div>
  );
};

export const HolographicCard: React.FC<{ children: React.ReactNode; className?: string }> = ({
  children,
  className = ''
}) => (
  <motion.div
    className={`relative overflow-hidden ${className}`}
    whileHover={{ scale: 1.02 }}
    style={{
      background: 'linear-gradient(45deg, rgba(139, 92, 246, 0.1), rgba(16, 185, 129, 0.1))',
      borderImage: 'linear-gradient(45deg, #8b5cf6, #10b981) 1'
    }}
  >
    <motion.div
      className="absolute inset-0 opacity-20"
      animate={{
        background: [
          'linear-gradient(0deg, transparent, rgba(139, 92, 246, 0.3), transparent)',
          'linear-gradient(180deg, transparent, rgba(139, 92, 246, 0.3), transparent)'
        ]
      }}
      transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
    />
    {children}
  </motion.div>
);

export const DataStream: React.FC<{ direction?: 'horizontal' | 'vertical' }> = ({ 
  direction = 'horizontal' 
}) => {
  const streamData = ['█', '▓', '▒', '░'];
  
  return (
    <div className="absolute inset-0 pointer-events-none overflow-hidden opacity-30">
      {Array.from({ length: 10 }).map((_, i) => (
        <motion.div
          key={i}
          className="absolute text-blue-400 text-xs"
          style={{
            [direction === 'horizontal' ? 'top' : 'left']: `${(i / 10) * 100}%`,
            [direction === 'horizontal' ? 'left' : 'top']: '-10%'
          }}
          animate={{
            [direction === 'horizontal' ? 'x' : 'y']: ['0%', '110vw']
          }}
          transition={{
            duration: Math.random() * 5 + 3,
            repeat: Infinity,
            delay: Math.random() * 2,
            ease: 'linear'
          }}
        >
          {streamData[Math.floor(Math.random() * streamData.length)]}
        </motion.div>
      ))}
    </div>
  );
};

export const QuantumParticles: React.FC<{ count?: number; color?: string }> = ({ 
  count = 50, 
  color = '#8b5cf6' 
}) => (
  <div className="absolute inset-0 pointer-events-none overflow-hidden">
    {Array.from({ length: count }).map((_, i) => (
      <motion.div
        key={i}
        className="absolute w-1 h-1 rounded-full"
        style={{ backgroundColor: color }}
        initial={{
          x: Math.random() * window.innerWidth,
          y: Math.random() * window.innerHeight,
          opacity: 0
        }}
        animate={{
          x: Math.random() * window.innerWidth,
          y: Math.random() * window.innerHeight,
          opacity: [0, 1, 0],
          scale: [0, 1, 0]
        }}
        transition={{
          duration: Math.random() * 10 + 5,
          repeat: Infinity,
          delay: Math.random() * 5,
          ease: 'easeInOut'
        }}
      />
    ))}
  </div>
);

export const CyberpunkButton: React.FC<{
  children: React.ReactNode;
  onClick?: () => void;
  variant?: 'primary' | 'secondary' | 'danger';
  isLoading?: boolean;
  className?: string;
}> = ({ 
  children, 
  onClick, 
  variant = 'primary', 
  isLoading = false,
  className = '' 
}) => {
  const variants = {
    primary: 'bg-purple-600 hover:bg-purple-700 border-purple-500',
    secondary: 'bg-blue-600 hover:bg-blue-700 border-blue-500',
    danger: 'bg-red-600 hover:bg-red-700 border-red-500'
  };

  return (
    <motion.button
      onClick={onClick}
      disabled={isLoading}
      className={`
        relative px-6 py-3 font-mono text-white border-2 rounded
        ${variants[variant]} ${className}
        transition-all duration-200 overflow-hidden
        disabled:opacity-50 disabled:cursor-not-allowed
      `}
      whileHover={{ scale: isLoading ? 1 : 1.05 }}
      whileTap={{ scale: isLoading ? 1 : 0.95 }}
    >
      <motion.div
        className="absolute inset-0 bg-gradient-to-r from-transparent via-white to-transparent opacity-20"
        animate={{
          x: ['-100%', '100%']
        }}
        transition={{
          duration: 2,
          repeat: Infinity,
          ease: 'linear'
        }}
      />
      <div className="relative z-10 flex items-center justify-center gap-2">
        {isLoading && <LoadingSpinner size="sm" />}
        {children}
      </div>
    </motion.button>
  );
};