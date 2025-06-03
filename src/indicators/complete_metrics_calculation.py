def _calculate_proper_metrics(self, trades: List[Trade], equity_curve: List[EquityPoint], 
                            config: BacktestConfig, df: pd.DataFrame) -> FixedBacktestResult:
    """Calculate ALL metrics properly - THE GOLD STANDARD IMPLEMENTATION! 🏆"""
    
    if not trades:
        logger.warning("No trades executed - returning empty result")
        return self._empty_result(config, df)
    
    # Basic trade statistics
    total_trades = len(trades)
    winning_trades = len([t for t in trades if t.net_profit > 0])
    losing_trades = total_trades - winning_trades
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    # Return calculations
    final_capital = equity_curve[-1].capital
    total_return_pct = (final_capital - self.initial_capital) / self.initial_capital
    total_profit_usd = final_capital - self.initial_capital
    
    # Time calculations
    start_date = equity_curve[0].timestamp
    end_date = equity_curve[-1].timestamp
    trading_days = (end_date - start_date).days
    years = trading_days / 365.25
    
    # CAGR calculation
    if years > 0 and final_capital > 0:
        cagr = (final_capital / self.initial_capital) ** (1/years) - 1
    else:
        cagr = 0
    
    # PROPER Max Drawdown calculation
    max_drawdown_pct = max([ep.drawdown_pct for ep in equity_curve]) if equity_curve else 0
    
    # Max drawdown duration (FIXED edge case handling)
    in_drawdown = False
    drawdown_start = None
    max_dd_duration = 0
    
    for ep in equity_curve:
        if ep.drawdown_pct > 0.01:  # In drawdown (>1%)
            if not in_drawdown:
                drawdown_start = ep.timestamp
                in_drawdown = True
        else:  # Out of drawdown
            if in_drawdown and drawdown_start:
                duration = (ep.timestamp - drawdown_start).days
                max_dd_duration = max(max_dd_duration, duration)
                in_drawdown = False
    
    # EDGE CASE: Check if we end in drawdown
    if in_drawdown and drawdown_start:
        duration = (end_date - drawdown_start).days
        max_dd_duration = max(max_dd_duration, duration)
    
    # 🚀 GOLD STANDARD: Daily Equity Returns for Sharpe/Sortino/Volatility
    # Convert equity curve to DataFrame for resampling
    equity_df = pd.DataFrame([
        {'timestamp': ep.timestamp, 'capital': ep.capital} 
        for ep in equity_curve
    ])
    equity_df = equity_df.set_index('timestamp')
    
    # Resample to daily and calculate returns
    daily_equity = equity_df['capital'].resample('D').last().dropna()
    daily_returns = daily_equity.pct_change().dropna()
    
    # Market returns (daily resampled)
    market_df = df[['close']].copy()
    daily_market = market_df['close'].resample('D').last().dropna()
    market_returns = daily_market.pct_change().dropna()
    
    # Align dates for proper comparison
    common_dates = daily_returns.index.intersection(market_returns.index)
    aligned_portfolio_returns = daily_returns.reindex(common_dates)
    aligned_market_returns = market_returns.reindex(common_dates)
    
    # Remove any NaN values after alignment
    valid_mask = ~(aligned_portfolio_returns.isna() | aligned_market_returns.isna())
    aligned_portfolio_returns = aligned_portfolio_returns[valid_mask]
    aligned_market_returns = aligned_market_returns[valid_mask]
    
    # 🎯 GOLD STANDARD: Consistent annualization (crypto 24/7/365 vs traditional 252)
    # For crypto, we use 365 days for true 24/7 markets
    ANNUALIZATION_FACTOR = 365 if 'crypto' in str(config.symbol).lower() else 252
    
    # 🎯 PROPER Sharpe ratio calculation (daily returns, annualized)
    if len(aligned_portfolio_returns) > 1 and aligned_portfolio_returns.std() > 0:
        daily_risk_free = self.risk_free_rate / ANNUALIZATION_FACTOR  # Consistent daily risk-free rate
        excess_returns = aligned_portfolio_returns - daily_risk_free
        sharpe_ratio = excess_returns.mean() / aligned_portfolio_returns.std() * np.sqrt(ANNUALIZATION_FACTOR)
    else:
        sharpe_ratio = 0
    
    # 🎯 PROPER Sortino ratio (downside deviation)
    if len(aligned_portfolio_returns) > 1:
        negative_returns = aligned_portfolio_returns[aligned_portfolio_returns < 0]
        if len(negative_returns) > 0:
            downside_deviation = negative_returns.std() * np.sqrt(ANNUALIZATION_FACTOR)
            daily_risk_free = self.risk_free_rate / ANNUALIZATION_FACTOR
            excess_return_annual = (aligned_portfolio_returns.mean() - daily_risk_free) * ANNUALIZATION_FACTOR
            sortino_ratio = excess_return_annual / downside_deviation
        else:
            sortino_ratio = float('inf') if aligned_portfolio_returns.mean() > 0 else 0
    else:
        sortino_ratio = 0
    
    # 🎯 PROPER Annualized Volatility
    if len(aligned_portfolio_returns) > 1:
        volatility_annualized = aligned_portfolio_returns.std() * np.sqrt(ANNUALIZATION_FACTOR)
    else:
        volatility_annualized = 0
    
    # Calmar ratio
    calmar_ratio = cagr / max_drawdown_pct if max_drawdown_pct > 0 else float('inf')
    
    # PROPER Profit Factor calculation
    gross_profits = [t.gross_profit for t in trades if t.gross_profit > 0]
    gross_losses = [abs(t.gross_profit) for t in trades if t.gross_profit < 0]
    
    total_gross_profit = sum(gross_profits)
    total_gross_loss = sum(gross_losses)
    profit_factor = total_gross_profit / total_gross_loss if total_gross_loss > 0 else float('inf')
    
    # Trade statistics
    trade_returns = [t.return_pct for t in trades]
    avg_trade_return = np.mean(trade_returns)
    best_trade = max(trade_returns) if trade_returns else 0
    worst_trade = min(trade_returns) if trade_returns else 0
    
    winning_returns = [t.return_pct for t in trades if t.net_profit > 0]
    losing_returns = [t.return_pct for t in trades if t.net_profit < 0]
    
    avg_win = np.mean(winning_returns) if winning_returns else 0
    avg_loss = np.mean(losing_returns) if losing_returns else 0
    
    # Winning/losing streaks
    streaks = []
    current_streak = 0
    current_type = None
    
    for trade in trades:
        trade_type = 'win' if trade.net_profit > 0 else 'loss'
        if trade_type == current_type:
            current_streak += 1
        else:
            if current_streak > 0:
                streaks.append((current_type, current_streak))
            current_streak = 1
            current_type = trade_type
    
    if current_streak > 0:
        streaks.append((current_type, current_streak))
    
    max_winning_streak = max([s[1] for s in streaks if s[0] == 'win'], default=0)
    max_losing_streak = max([s[1] for s in streaks if s[0] == 'loss'], default=0)
    
    # Trading frequency
    trades_per_week = total_trades / (trading_days / 7) if trading_days > 0 else 0
    avg_duration = np.mean([t.duration_hours for t in trades]) if trades else 0
    
    # Market comparison
    if len(daily_market) > 0:
        market_return = (daily_market.iloc[-1] - daily_market.iloc[0]) / daily_market.iloc[0]
    else:
        market_return = 0
    
    alpha = total_return_pct - market_return
    
    # 🚀 GOLD STANDARD: Beta calculation with perfect alignment
    if len(aligned_portfolio_returns) > 1 and len(aligned_market_returns) > 1:
        if aligned_market_returns.var() > 1e-10:  # Avoid division by near-zero variance
            # Covariance between portfolio and market returns
            covariance = np.cov(aligned_portfolio_returns, aligned_market_returns)[0, 1]
            market_variance = aligned_market_returns.var()
            beta = covariance / market_variance
        else:
            beta = np.nan  # Market has no variance - beta undefined
    else:
        beta = np.nan  # Insufficient data for correlation
    
    # Cost analysis
    total_fees_paid = sum([t.fees_paid for t in trades])
    fees_as_pct_of_profit = (total_fees_paid / total_profit_usd * 100) if total_profit_usd > 0 else 0
    
    # 🎉 ASSEMBLE THE FINAL RESULT
    return FixedBacktestResult(
        config=config,
        
        # Trade statistics
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        win_rate=win_rate,
        
        # Return metrics
        total_return_pct=total_return_pct,
        total_profit_usd=total_profit_usd,
        cagr=cagr,
        
        # Risk metrics - PROPERLY CALCULATED!
        max_drawdown_pct=max_drawdown_pct,
        max_drawdown_duration_days=max_dd_duration,
        volatility_annualized=volatility_annualized,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        calmar_ratio=calmar_ratio,
        
        # Performance metrics
        profit_factor=profit_factor,
        avg_trade_return_pct=avg_trade_return,
        best_trade_pct=best_trade,
        worst_trade_pct=worst_trade,
        avg_win_pct=avg_win,
        avg_loss_pct=avg_loss,
        largest_winning_streak=max_winning_streak,
        largest_losing_streak=max_losing_streak,
        
        # Trading frequency
        trades_per_week=trades_per_week,
        avg_trade_duration_hours=avg_duration,
        
        # Benchmark comparison
        market_return_pct=market_return,
        alpha=alpha,
        beta=beta,
        
        # Cost analysis
        total_fees_paid=total_fees_paid,
        fees_as_pct_of_profit=fees_as_pct_of_profit,
        
        # Timing
        start_date=start_date,
        end_date=end_date,
        trading_days=trading_days,
        
        # Data for visualization
        equity_curve=equity_curve,
        trades=trades
    )

def _empty_result(self, config: BacktestConfig, df: pd.DataFrame) -> FixedBacktestResult:
    """Return a properly initialized empty result when no trades are executed"""
    
    start_date = df.index[0] if len(df) > 0 else datetime.now()
    end_date = df.index[-1] if len(df) > 0 else datetime.now()
    trading_days = (end_date - start_date).days if end_date > start_date else 1
    
    # Market return for comparison
    market_return = 0
    if len(df) > 0:
        market_return = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]
    
    # Empty equity curve (just starting capital)
    equity_curve = [EquityPoint(start_date, self.initial_capital, 0.0, self.initial_capital)]
    
    return FixedBacktestResult(
        config=config,
        
        # Trade statistics
        total_trades=0,
        winning_trades=0,
        losing_trades=0,
        win_rate=0.0,
        
        # Return metrics
        total_return_pct=0.0,
        total_profit_usd=0.0,
        cagr=0.0,
        
        # Risk metrics
        max_drawdown_pct=0.0,
        max_drawdown_duration_days=0.0,
        volatility_annualized=0.0,
        sharpe_ratio=0.0,
        sortino_ratio=0.0,
        calmar_ratio=0.0,
        
        # Performance metrics
        profit_factor=0.0,
        avg_trade_return_pct=0.0,
        best_trade_pct=0.0,
        worst_trade_pct=0.0,
        avg_win_pct=0.0,
        avg_loss_pct=0.0,
        largest_winning_streak=0,
        largest_losing_streak=0,
        
        # Trading frequency
        trades_per_week=0.0,
        avg_trade_duration_hours=0.0,
        
        # Benchmark comparison
        market_return_pct=market_return,
        alpha=-market_return,  # Negative alpha since we made 0% while market moved
        beta=0.0,
        
        # Cost analysis
        total_fees_paid=0.0,
        fees_as_pct_of_profit=0.0,
        
        # Timing
        start_date=start_date,
        end_date=end_date,
        trading_days=trading_days,
        
        # Empty data
        equity_curve=equity_curve,
        trades=[]
    )