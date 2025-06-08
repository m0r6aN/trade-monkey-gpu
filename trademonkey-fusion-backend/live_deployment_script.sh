# File: trademonkey-fusion-backend/deploy_live_system.sh
#!/bin/bash

# TradeMonkey Fusion - Live Deployment Script
# "LET'S GO LIVE AND PRINT MONEY!" 🚀

echo "🦍 TradeMonkey Fusion - Live Deployment Starting..."
echo "=================================================="

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check if .env file exists
if [ ! -f ".env" ]; then
    print_error ".env file not found! Creating template..."
    cat > .env << EOL
# TradeMonkey Fusion Configuration
# Copy this to .env and fill in your actual values

# Kraken API (REQUIRED for live trading)
KRAKEN_API_KEY=your_kraken_api_key_here
KRAKEN_API_SECRET=your_kraken_secret_here

# Twitter API (for sentiment analysis)
TWITTER_BEARER_TOKEN=your_twitter_bearer_token

# Reddit API (for sentiment analysis)
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_secret

# Discord/Telegram Notifications
DISCORD_WEBHOOK_URL=your_discord_webhook
TELEGRAM_BOT_TOKEN=your_telegram_token
TELEGRAM_CHAT_ID=your_chat_id

# Trading Settings
USE_TESTNET=true
DRY_RUN_MODE=true
STARTING_CAPITAL=10000.0
MAX_POSITIONS=4
POSITION_SIZE_PCT=0.25

# Redis Configuration
REDIS_URL=redis://localhost:6379

# GPU Settings
USE_GPU_ACCELERATION=true
GPU_MEMORY_LIMIT=8192
EOL
    print_warning "Please edit .env file with your API keys before proceeding!"
    exit 1
fi

print_status "Found .env configuration file"

# Check Python version
python_version=$(python3 --version 2>&1 | grep -oP '(?<=Python )\d+\.\d+')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    print_error "Python 3.8+ required. Found: $python_version"
    exit 1
fi

print_status "Python version check passed: $python_version"

# Install/upgrade pip
print_info "Upgrading pip..."
pip3 install --upgrade pip

# Install requirements
print_info "Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
    pip3 install -r requirements.txt
    if [ $? -eq 0 ]; then
        print_status "Dependencies installed successfully"
    else
        print_error "Failed to install dependencies"
        exit 1
    fi
else
    print_error "requirements.txt not found!"
    exit 1
fi

# Check Redis connection
print_info "Checking Redis connection..."
redis-cli ping > /dev/null 2>&1
if [ $? -eq 0 ]; then
    print_status "Redis is running"
else
    print_warning "Redis not running. Attempting to start..."
    # Try to start Redis (varies by OS)
    if command -v systemctl > /dev/null; then
        sudo systemctl start redis
    elif command -v brew > /dev/null; then
        brew services start redis
    else
        print_error "Please start Redis manually: redis-server"
        exit 1
    fi
fi

# Validate sentiment pipeline
print_info "Validating sentiment pipeline..."
python3 src/utils/sentiment_pipeline_validator.py --quick-check
if [ $? -eq 0 ]; then
    print_status "Sentiment pipeline validated"
else
    print_warning "Sentiment pipeline issues detected (may work with mock data)"
fi

# Test Kraken connection
print_info "Testing Kraken API connection..."
python3 -c "
import ccxt
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('KRAKEN_API_KEY')
api_secret = os.getenv('KRAKEN_API_SECRET')

if not api_key or not api_secret or 'your_' in api_key:
    print('⚠️  Kraken API keys not configured - will use demo mode')
    exit(0)

try:
    exchange = ccxt.kraken({'apiKey': api_key, 'secret': api_secret})
    exchange.load_markets()
    print('✅ Kraken API connection successful')
except Exception as e:
    print(f'❌ Kraken API error: {e}')
    exit(1)
"

# Create necessary directories
print_info "Creating necessary directories..."
mkdir -p logs
mkdir -p cache
mkdir -p backtest_results
print_status "Directories created"

# Start the deployment
echo ""
echo "🚀 Starting TradeMonkey Fusion Components..."
echo "============================================="

# Function to start component in background
start_component() {
    local name=$1
    local command=$2
    local logfile="logs/${name}.log"
    
    print_info "Starting $name..."
    nohup $command > "$logfile" 2>&1 &
    local pid=$!
    echo $pid > "logs/${name}.pid"
    
    # Wait a moment and check if process is still running
    sleep 2
    if kill -0 $pid 2>/dev/null; then
        print_status "$name started successfully (PID: $pid)"
        return 0
    else
        print_error "$name failed to start. Check $logfile"
        return 1
    fi
}

# Start sentiment collector
start_component "sentiment-collector" "python3 src/utils/sentiment_redis_pusher.py"

# Start enhanced API server
start_component "api-server" "python3 api_server_enhanced.py"

# Wait for API server to be ready
print_info "Waiting for API server to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:8080/api/health > /dev/null; then
        print_status "API server is ready"
        break
    fi
    sleep 1
    if [ $i -eq 30 ]; then
        print_error "API server failed to start within 30 seconds"
        exit 1
    fi
done

# Test API endpoints
print_info "Testing API endpoints..."
curl -s http://localhost:8080/api/health | jq . > /dev/null
if [ $? -eq 0 ]; then
    print_status "API endpoints responding"
else
    print_warning "API endpoints may not be fully ready"
fi

# Display running components
echo ""
echo "🎯 TradeMonkey Fusion Status"
echo "============================"

# Check component status
check_component() {
    local name=$1
    local pidfile="logs/${name}.pid"
    
    if [ -f "$pidfile" ]; then
        local pid=$(cat "$pidfile")
        if kill -0 $pid 2>/dev/null; then
            print_status "$name running (PID: $pid)"
        else
            print_error "$name not running"
        fi
    else
        print_error "$name PID file not found"
    fi
}

check_component "sentiment-collector"
check_component "api-server"

# Display access URLs
echo ""
echo "🌐 Access URLs"
echo "=============="
print_info "API Server: http://localhost:8080"
print_info "Health Check: http://localhost:8080/api/health"
print_info "WebSocket: ws://localhost:8080/ws"
print_info "Frontend (if running): http://localhost:3000"

# Display log monitoring command
echo ""
echo "📊 Monitoring Commands"
echo "====================="
print_info "View API logs: tail -f logs/api-server.log"
print_info "View sentiment logs: tail -f logs/sentiment-collector.log"
print_info "Monitor all logs: tail -f logs/*.log"

# Display shutdown commands
echo ""
echo "🛑 Shutdown Commands"
echo "==================="
print_info "Stop all: ./stop_trademonkey.sh"
print_info "Or manually: kill \$(cat logs/*.pid)"

# Final status
echo ""
if [ -f "logs/api-server.pid" ] && [ -f "logs/sentiment-collector.pid" ]; then
    echo "🎉 TradeMonkey Fusion is LIVE and ready to print money!"
    echo "🚀 LFG TO THE QUANTUM MOON! 🌙⚛️"
else
    echo "⚠️  Some components may not have started correctly."
    echo "Check logs in the logs/ directory for details."
fi

echo ""
echo "Next steps:"
echo "1. Start the frontend: cd ../trademonkey-fusion-ui && npm run dev"
echo "2. Open http://localhost:3000 to see the dashboard"
echo "3. Monitor the logs for any issues"
echo "4. 💰 Watch the money printer go BRRRR!"