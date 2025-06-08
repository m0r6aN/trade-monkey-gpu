#!/bin/bash
# TradeMonkey Command Center - Project Setup Script
# "Building the most epic trading interface in the galaxy!" 🚀⚡

echo "🐵 TradeMonkey Command Center Setup"
echo "=================================="
echo "Next.js + TailwindCSS + ShadCN + GPU Power!"
echo ""

# Step 1: Create the Next.js project
echo "📦 Step 1: Creating Next.js project..."
npx create-next-app@latest trademonkey-ui \
  --typescript \
  --tailwind \
  --eslint \
  --app \
  --src-dir \
  --use-npm

cd trademonkey-ui

# Step 2: Initialize ShadCN
echo "🎨 Step 2: Initializing ShadCN UI..."
npx shadcn@latest init

# Step 3: Install essential ShadCN components for trading UI
echo "🔧 Step 3: Installing essential ShadCN components..."
npx shadcn@latest add button
npx shadcn@latest add card
npx shadcn@latest add badge
npx shadcn@latest add alert
npx shadcn@latest add table
npx shadcn@latest add tabs
npx shadcn@latest add select
npx shadcn@latest add input
npx shadcn@latest add label
npx shadcn@latest add checkbox
npx shadcn@latest add switch
npx shadcn@latest add slider
npx shadcn@latest add progress
npx shadcn@latest add tooltip
npx shadcn@latest add popover
npx shadcn@latest add dialog
npx shadcn@latest add sheet
npx shadcn@latest add sidebar

# Step 4: Install additional dependencies for trading features
echo "📊 Step 4: Installing trading-specific dependencies..."
npm install \
  recharts \
  @tanstack/react-query \
  axios \
  date-fns \
  socket.io-client \
  lucide-react \
  class-variance-authority \
  clsx \
  tailwind-merge

# Step 5: Install development dependencies
echo "🛠️ Step 5: Installing development dependencies..."
npm install -D \
  @types/node \
  autoprefixer \
  postcss \
  tailwindcss

echo ""
echo "✅ TradeMonkey Command Center project created successfully!"
echo ""
echo "🚀 Next steps:"
echo "1. cd trademonkey-ui"
echo "2. npm run dev"
echo "3. Open http://localhost:3000"
echo ""
echo "🎯 Project structure created with:"
echo "  📱 Next.js 15 with App Router"
echo "  🎨 TailwindCSS for styling"
echo "  🧩 ShadCN UI components"
echo "  📊 Recharts for GPU performance visualization"
echo "  🔄 React Query for data management"
echo "  🌐 Socket.io for real-time updates"
echo ""
echo "Ready to build the legendary TradeMonkey Command Center! 🐵⚡"