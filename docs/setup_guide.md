# 🐵 TradeMonkey Lite Setup Guide

> *"A journey of a thousand trades begins with a single setup"* - Ancient Crypto Wisdom

Welcome to the complete setup guide for TradeMonkey Lite! This guide will walk you through every step needed to get your automated crypto trading bot up and running with Kraken.

## 🚀 Quick Start (TL;DR)

```bash
# 1. Clone/download the project
# 2. Install dependencies
pip install -r requirements.txt

# 3. Setup environment
cp .env.example .env
# Edit .env with your Kraken API keys

# 4. Test configuration
python test_config.py

# 5. Start paper trading
python main.py --paper
```

## 📋 Prerequisites

Before we begin, make sure you have:

- **Python 3.8+** installed
- **Kraken account** (create at [kraken.com](https://kraken.com))
- **Basic understanding of crypto trading** (or at least willingness to learn!)
- **$100+ in USDT/USD** for testing (start small!)
- **Diamond hands** 💎🙌 (metaphorical, but recommended)

## 🔧 Step 1: Environment Setup

### Install Python Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Or install individually if you prefer:
pip install ccxt pandas numpy ta aiohttp python-dotenv coloredlogs
```

### Verify Installation

```bash
# Run our test suite
python test_config.py
```

You should see all green checkmarks ✅. If not, install any missing packages.

## 🔑 Step 2: Kraken API Setup

### Create Kraken Account

1. Go to [kraken.com](https://kraken.com) and create an account
2. Complete identity verification (required for futures trading)
3. Deposit some funds (start with $100-500 for testing)

### Generate API Keys

1. Log into Kraken
2. Go to **Settings** → **API**
3. Click **Generate New Key**
4. Set permissions:
   - ✅ **Query Funds**
   - ✅ **Query Open Orders & Trades**
   - ✅ **Query Closed Orders & Trades**
   - ✅ **Query Ledger Entries**
   - ✅ **Create & Modify Orders**
   - ✅ **Cancel Orders**
   - ✅ **Futures Trading** (IMPORTANT!)
5. **Save your API Key and Secret** - you won't see the secret again!

### Configure Environment

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your favorite editor
nano .env
# or
code .env
```

Add your Kraken credentials:

```bash
# Required - Your Kraken API credentials
KRAKEN_API_KEY=your_actual_api_key_here
KRAKEN_API_SECRET=your_actual_api_secret_here

# Recommended - Set to true for testing
USE_TESTNET=true

# Optional - Notification settings
DISCORD_WEBHOOK_URL=your_discord_webhook_url
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id
```

## 🧪 Step 3: Testing & Validation

### Test Configuration

```bash
# Verify everything is setup correctly
python test_config.py
```

### Check Available Symbols

```bash
# See what trading pairs are available
python main.py --symbols
```

### View Configuration

```bash
# Review your current settings
python main.py --config
```

## 🎯 Step 4: First Run (Paper Trading)

**ALWAYS start with paper trading!** This lets you test the bot without risking real money.

```bash
# Start paper trading
python main.py --paper
```

You should see:
- ✅ Configuration summary
- ✅ Connection to Kraken
- ✅ Market data loading
- ✅ Signal monitoring begins

## 📊 Step 5: Set Up Notifications (Optional but Recommended)

### Discord Notifications

1. Create a Discord server (or use existing)
2. Go to **Server Settings** → **Integrations** → **Webhooks**
3. Click **New Webhook**
4. Copy the webhook URL
5. Add to your `.env` file:

```bash
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/your_webhook_url_here
```

### Telegram Notifications

1. Message [@BotFather](https://t.me/botfather) on Telegram
2. Create a new bot with `/newbot`
3. Save the bot token
4. Message [@userinfobot](https://t.me/userinfobot) to get your chat ID
5. Add to your `.env` file:

```bash
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
```

## ⚙️ Step 6: Configuration Customization

### Edit Trading Parameters

Modify `config/strategies.json` to customize:

- **Trading symbols**: Which coins to trade
- **Timeframes**: Chart timeframes to analyze
- **Risk management**: Stop losses, take profits
- **Position sizing**: How much to risk per trade
- **Technical indicators**: Signal generation parameters

### Key Settings to Review

```json
{
  "symbols": ["BTC/USD", "ETH/USD", "SOL/USD"],
  "risk_management": {
    "max_positions": 4,
    "position_size_pct": 0.25,
    "initial_leverage": 2.0,
    "trailing_stop_pct": 0.05
  }
}
```

## 🚨 Step 7: Going Live (When Ready)

**⚠️ IMPORTANT: Only go live after thorough testing!**

### Pre-Live Checklist

- [ ] Bot runs successfully in paper mode for at least 24 hours
- [ ] You understand the strategy and risks
- [ ] Notifications are working
- [ ] You have appropriate capital allocated
- [ ] You've read all the warnings in the code

### Switch to Live Trading

```bash
# Option 1: Edit .env file
USE_TESTNET=false

# Option 2: Use command line override
python main.py --live
```

The bot will ask for confirmation:

```
Type 'RELEASE THE KRAKEN' to continue:
```

## 📈 Step 8: Monitoring & Management

### Daily Monitoring

- Check Discord/Telegram notifications
- Review position performance in Kraken
- Monitor bot logs for any errors
- Adjust strategy parameters as needed

### Log Files

Bot logs are stored in the `logs/` directory:
- `trademonkey.log` - General application logs
- `trades.log` - Trade execution logs
- `errors.log` - Error logs

### Performance Tracking

Track your performance:
- Keep a trading journal
- Monitor monthly P&L
- Adjust risk parameters based on results
- Scale position sizes as you gain confidence

## 🛠️ Troubleshooting

### Common Issues

**"Missing Kraken API credentials"**
- Check your `.env` file has the correct API key and secret
- Ensure no extra spaces or quotes around the values

**"Configuration validation failed"**
- Run `python test_config.py` to see specific errors
- Check all required fields in `.env` are filled

**"No signals generated"**
- Market may be choppy - this is normal
- Try adding more volatile trading pairs
- Adjust signal sensitivity in `strategies.json`

**"Insufficient balance"**
- Ensure you have USD/USDT in your Kraken futures wallet
- Reduce position size percentage if needed

### Getting Help

1. Check the logs in `logs/` directory
2. Run `python main.py --validate` to check config
3. Search GitHub issues for similar problems
4. Create a new issue with logs and error details

## ⚠️ Important Warnings

- **This is NOT financial advice** - We're just code enthusiasts who like automation
- **Start small** - Use money you can afford to lose
- **Leverage is risky** - Even 2x can liquidate you on bad days
- **Markets are unpredictable** - Past performance ≠ future results
- **Monitor regularly** - Automated doesn't mean "set and forget"

## 🎓 Learning Resources

- [Kraken Futures Guide](https://www.kraken.com/features/futures)
- [Technical Analysis Basics](https://www.investopedia.com/technical-analysis-4689657)
- [Risk Management](https://www.babypips.com/learn/forex/risk-management)
- [CCXT Documentation](https://docs.ccxt.com/en/latest/)

## 🎬 Final Words

*"The market can remain irrational longer than you can remain solvent."* - John Maynard Keynes

*"But with proper risk management and automated execution, we can remain solvent longer than most expect!"* - TradeMonkey Lite Philosophy

---

**Ready to release the Kraken?** 🐙⚡

Remember: We're not just building a trading bot, we're building a money printer! 🖨️💰

Stay safe, trade smart, and may your candles be forever green! 🟢

Built with 💪 and ☕ by developers who believe in the power of automation and compound gains.

**LFG! 🚀🚀🚀**