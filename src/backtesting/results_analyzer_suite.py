#!/usr/bin/env python3
"""
TradeMonkey Lite - Results Analysis & Visualization Suite
"From raw data to wisdom - the transformation is complete!" 🧠📊⚡

This is where 972 configurations become actionable intelligence!
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Dict, Optional, Callable, Any, Tuple
import logging
from datetime import datetime
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import our backtesting components
# from your_backtester_module import FixedBacktestResult, BacktestConfig, MarketCondition

logger = logging.getLogger('ResultsAnalyzer')

class ResultsAnalyzer:
    """
    The INTELLIGENCE LAYER! 🧠
    
    Transforms raw backtest results into actionable trading wisdom.
    Makes hedge fund quants weep with envy!
    """
    
    def __init__(self, results: List, save_path: Optional[str] = None):
        """Initialize with results and optional save path for persistence"""
        if not results:
            raise ValueError("Results list cannot be empty - need data to analyze!")
        
        self.raw_results = results
        self.results_df = self._results_to_dataframe(results)
        self.save_path = Path(save_path) if save_path else Path("backtest_results")
        self.save_path.mkdir(exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("viridis")
        
        logger.info(f"🧠 ResultsAnalyzer initialized with {len(results)} backtest results")
        logger.info(f"📊 Total configurations analyzed: {len(self.results_df)}")
        logger.info(f"💾 Results will be saved to: {self.save_path}")
        
    def _results_to_dataframe(self, results: List) -> pd.DataFrame:
        """Convert FixedBacktestResult objects to a comprehensive DataFrame"""
        data = []
        
        for res in results:
            # Flatten config and results into analysis-ready format
            row = {
                # Configuration parameters
                'symbol': res.config.symbol,
                'timeframe': res.config.timeframe,
                'test_period_name': res.config.test_period.name,
                'market_condition': res.config.test_period.expected_condition.value,
                'strategy_name': res.config.strategy_name,
                'signal_threshold': res.config.signal_threshold,
                'position_size_pct': res.config.position_size_pct,
                'use_stop_loss': res.config.use_stop_loss,
                'use_take_profit': res.config.use_take_profit,
                'leverage': res.config.leverage,
                'atr_stop_multiplier': res.config.atr_stop_multiplier,
                'atr_profit_multiplier': res.config.atr_profit_multiplier,
                
                # Performance metrics
                'total_return_pct': res.total_return_pct,
                'cagr': res.cagr,
                'max_drawdown_pct': res.max_drawdown_pct,
                'max_drawdown_duration_days': res.max_drawdown_duration_days,
                'volatility_annualized': res.volatility_annualized,
                'sharpe_ratio': res.sharpe_ratio,
                'sortino_ratio': res.sortino_ratio,
                'calmar_ratio': res.calmar_ratio,
                
                # Trading statistics
                'profit_factor': res.profit_factor,
                'win_rate': res.win_rate,
                'total_trades': res.total_trades,
                'winning_trades': res.winning_trades,
                'losing_trades': res.losing_trades,
                'avg_trade_return_pct': res.avg_trade_return_pct,
                'best_trade_pct': res.best_trade_pct,
                'worst_trade_pct': res.worst_trade_pct,
                'avg_win_pct': res.avg_win_pct,
                'avg_loss_pct': res.avg_loss_pct,
                'largest_winning_streak': res.largest_winning_streak,
                'largest_losing_streak': res.largest_losing_streak,
                
                # Market comparison
                'alpha': res.alpha,
                'beta': res.beta,
                'market_return_pct': res.market_return_pct,
                
                # Frequency & timing
                'trades_per_week': res.trades_per_week,
                'avg_trade_duration_hours': res.avg_trade_duration_hours,
                'trading_days': res.trading_days,
                
                # Cost analysis
                'total_fees_paid': res.total_fees_paid,
                'fees_as_pct_of_profit': res.fees_as_pct_of_profit,
                
                # Custom composite scores
                'risk_adjusted_return': res.cagr / max(res.max_drawdown_pct, 0.01),  # Custom metric
                'efficiency_score': res.total_return_pct / max(res.total_trades, 1),  # Return per trade
                'consistency_score': res.win_rate * res.profit_factor,  # Combined metric
            }
            data.append(row)
            
        df = pd.DataFrame(data)
        
        # Handle infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        logger.info(f"✅ Converted {len(df)} results to DataFrame")
        return df
    
    def save_results(self, filename: str = "backtest_results.csv"):
        """Save results DataFrame to CSV for external analysis"""
        filepath = self.save_path / filename
        self.results_df.to_csv(filepath, index=False)
        logger.info(f"💾 Results saved to {filepath}")
        
    def get_summary_stats(self, group_by_cols: List[str] = None, 
                         metric_col: str = 'sharpe_ratio') -> pd.DataFrame:
        """Calculate comprehensive summary statistics"""
        
        if group_by_cols is None:
            # Overall summary
            stats = self.results_df[metric_col].describe()
            stats['count_positive'] = (self.results_df[metric_col] > 0).sum()
            stats['count_negative'] = (self.results_df[metric_col] <= 0).sum()
            return stats
        
        # Grouped summary
        missing_cols = [col for col in group_by_cols if col not in self.results_df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
            
        grouped = self.results_df.groupby(group_by_cols)[metric_col].agg([
            'count', 'mean', 'median', 'std', 'min', 'max',
            lambda x: (x > 0).sum(),  # count_positive
            lambda x: (x <= 0).sum()  # count_negative
        ])
        
        grouped.columns = ['count', 'mean', 'median', 'std', 'min', 'max', 'count_positive', 'count_negative']
        return grouped.sort_values('mean', ascending=False)
    
    def find_sweet_spots(self, metric: str = 'sharpe_ratio', 
                        top_n: int = 10, min_trades: int = 5) -> pd.DataFrame:
        """Find the parameter combinations that consistently perform well"""
        
        # Filter for meaningful results
        filtered = self.results_df[self.results_df['total_trades'] >= min_trades].copy()
        
        if filtered.empty:
            logger.warning("No results meet the minimum trades requirement")
            return pd.DataFrame()
        
        # Group by key parameters and calculate performance
        param_cols = ['symbol', 'timeframe', 'signal_threshold', 'atr_stop_multiplier', 
                     'atr_profit_multiplier', 'position_size_pct', 'leverage']
        
        sweet_spots = filtered.groupby(param_cols).agg({
            metric: ['mean', 'std', 'count'],
            'total_trades': 'sum',
            'win_rate': 'mean',
            'max_drawdown_pct': 'mean',
            'cagr': 'mean'
        }).round(4)
        
        # Flatten column names
        sweet_spots.columns = ['_'.join(col).strip() for col in sweet_spots.columns]
        
        # Calculate robustness score (high mean, low std)
        sweet_spots['robustness_score'] = (
            sweet_spots[f'{metric}_mean'] / (sweet_spots[f'{metric}_std'] + 0.01)
        )
        
        # Sort by robustness and return top N
        return sweet_spots.sort_values('robustness_score', ascending=False).head(top_n)
    
    def analyze_regime_performance(self, metric: str = 'sharpe_ratio') -> pd.DataFrame:
        """Analyze performance across different market conditions"""
        
        regime_analysis = self.results_df.groupby('market_condition').agg({
            metric: ['count', 'mean', 'median', 'std', 'min', 'max'],
            'cagr': ['mean', 'median'],
            'max_drawdown_pct': ['mean', 'median'],
            'win_rate': ['mean', 'median'],
            'total_trades': ['sum', 'mean']
        }).round(4)
        
        # Flatten column names
        regime_analysis.columns = ['_'.join(col).strip() for col in regime_analysis.columns]
        
        return regime_analysis.sort_values(f'{metric}_mean', ascending=False)
    
    def plot_performance_heatmap(self, param1: str = 'atr_stop_multiplier', 
                               param2: str = 'atr_profit_multiplier',
                               metric: str = 'sharpe_ratio', 
                               aggfunc: Callable = np.mean,
                               figsize: Tuple[int, int] = (12, 8),
                               **filters):
        """Create beautiful heatmaps of parameter performance"""
        
        # Apply filters
        filtered_df = self.results_df.copy()
        for key, value in filters.items():
            if key in filtered_df.columns:
                filtered_df = filtered_df[filtered_df[key] == value]
        
        if filtered_df.empty:
            logger.warning(f"No data after filtering: {filters}")
            return
        
        # Create pivot table
        pivot_table = pd.pivot_table(
            filtered_df, 
            values=metric, 
            index=param1, 
            columns=param2, 
            aggfunc=aggfunc
        )
        
        # Create the plot
        plt.figure(figsize=figsize)
        
        # Use a diverging colormap centered on 0 for metrics like Sharpe
        if 'sharpe' in metric.lower() or 'alpha' in metric.lower():
            center = 0
            cmap = 'RdYlGn'
        else:
            center = None
            cmap = 'viridis'
            
        sns.heatmap(
            pivot_table, 
            annot=True, 
            fmt='.3f', 
            cmap=cmap,
            center=center,
            cbar_kws={'label': metric}
        )
        
        title = f'{metric.replace("_", " ").title()} vs {param1} & {param2}'
        if filters:
            title += f'\nFilters: {filters}'
            
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel(param2.replace('_', ' ').title(), fontsize=12)
        plt.ylabel(param1.replace('_', ' ').title(), fontsize=12)
        plt.tight_layout()
        
        # Save plot
        filename = f"heatmap_{metric}_{param1}_{param2}.png"
        plt.savefig(self.save_path / filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"💾 Heatmap saved as {filename}")
    
    def plot_risk_return_scatter(self, x_metric: str = 'max_drawdown_pct',
                               y_metric: str = 'cagr',
                               color_metric: str = 'sharpe_ratio',
                               size_metric: Optional[str] = None,
                               figsize: Tuple[int, int] = (14, 10),
                               **filters):
        """Create risk-return scatter plots with multiple dimensions"""
        
        # Apply filters
        filtered_df = self.results_df.copy()
        for key, value in filters.items():
            if key in filtered_df.columns:
                filtered_df = filtered_df[filtered_df[key] == value]
        
        if filtered_df.empty:
            logger.warning(f"No data after filtering: {filters}")
            return
        
        plt.figure(figsize=figsize)
        
        # Prepare size data
        sizes = None
        if size_metric and size_metric in filtered_df.columns:
            # Normalize sizes for visualization
            size_data = filtered_df[size_metric]
            sizes = 50 + 200 * (size_data - size_data.min()) / (size_data.max() - size_data.min())
        else:
            sizes = 100
        
        # Create scatter plot
        scatter = plt.scatter(
            filtered_df[x_metric],
            filtered_df[y_metric],
            c=filtered_df[color_metric],
            s=sizes,
            alpha=0.7,
            cmap='viridis',
            edgecolors='black',
            linewidth=0.5
        )
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label(color_metric.replace('_', ' ').title(), fontsize=12)
        
        # Labels and title
        plt.xlabel(x_metric.replace('_', ' ').title(), fontsize=12)
        plt.ylabel(y_metric.replace('_', ' ').title(), fontsize=12)
        
        title = f'{y_metric.replace("_", " ").title()} vs {x_metric.replace("_", " ").title()}'
        if filters:
            title += f'\nFilters: {filters}'
        plt.title(title, fontsize=14, fontweight='bold')
        
        # Add grid
        plt.grid(True, alpha=0.3)
        
        # Add quadrant lines if appropriate
        if x_metric == 'max_drawdown_pct' and y_metric == 'cagr':
            plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            plt.axvline(x=0.1, color='red', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        # Save plot
        filename = f"scatter_{y_metric}_vs_{x_metric}.png"
        plt.savefig(self.save_path / filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"💾 Scatter plot saved as {filename}")
    
    def plot_regime_boxplots(self, metrics: List[str] = None, 
                           figsize: Tuple[int, int] = (16, 12)):
        """Create box plots showing performance across market regimes"""
        
        if metrics is None:
            metrics = ['sharpe_ratio', 'cagr', 'max_drawdown_pct', 'win_rate']
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics[:4]):
            sns.boxplot(
                data=self.results_df,
                x='market_condition',
                y=metric,
                ax=axes[i]
            )
            
            axes[i].set_title(f'{metric.replace("_", " ").title()} by Market Condition', 
                            fontsize=12, fontweight='bold')
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        filename = "regime_performance_boxplots.png"
        plt.savefig(self.save_path / filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"💾 Regime boxplots saved as {filename}")
    
    def create_comprehensive_report(self, save_html: bool = True) -> str:
        """Generate a comprehensive HTML report with all key insights"""
        
        report_sections = []
        
        # Executive Summary
        report_sections.append("""
        <h1>🚀 TradeMonkey Backtesting Intelligence Report</h1>
        <h2>📊 Executive Summary</h2>
        """)
        
        # Overall statistics
        total_configs = len(self.results_df)
        profitable_configs = (self.results_df['total_return_pct'] > 0).sum()
        avg_sharpe = self.results_df['sharpe_ratio'].mean()
        best_sharpe = self.results_df['sharpe_ratio'].max()
        
        report_sections.append(f"""
        <ul>
            <li><strong>Total Configurations Tested:</strong> {total_configs:,}</li>
            <li><strong>Profitable Configurations:</strong> {profitable_configs:,} ({profitable_configs/total_configs:.1%})</li>
            <li><strong>Average Sharpe Ratio:</strong> {avg_sharpe:.3f}</li>
            <li><strong>Best Sharpe Ratio:</strong> {best_sharpe:.3f}</li>
        </ul>
        """)
        
        # Sweet spots analysis
        sweet_spots = self.find_sweet_spots()
        if not sweet_spots.empty:
            report_sections.append("""
            <h2>🎯 Top Performing Configurations</h2>
            """)
            report_sections.append(sweet_spots.head().to_html())
        
        # Regime analysis
        regime_perf = self.analyze_regime_performance()
        report_sections.append("""
        <h2>🌍 Performance by Market Regime</h2>
        """)
        report_sections.append(regime_perf.to_html())
        
        # Combine all sections
        full_report = "".join(report_sections)
        
        if save_html:
            report_path = self.save_path / "comprehensive_report.html"
            with open(report_path, 'w') as f:
                f.write(f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>TradeMonkey Intelligence Report</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 40px; }}
                        h1 {{ color: #2E8B57; }}
                        h2 {{ color: #4682B4; }}
                        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                        th {{ background-color: #f2f2f2; }}
                    </style>
                </head>
                <body>
                {full_report}
                </body>
                </html>
                """)
            
            logger.info(f"📄 Comprehensive report saved to {report_path}")
        
        return full_report
    
    def export_top_strategies(self, metric: str = 'sharpe_ratio', 
                            top_n: int = 5) -> Dict[str, Any]:
        """Export the top strategies for live trading implementation"""
        
        # Find top performers
        top_strategies = self.results_df.nlargest(top_n, metric)
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'selection_metric': metric,
            'top_strategies': []
        }
        
        for _, row in top_strategies.iterrows():
            strategy_config = {
                'symbol': row['symbol'],
                'timeframe': row['timeframe'],
                'signal_threshold': row['signal_threshold'],
                'atr_stop_multiplier': row['atr_stop_multiplier'],
                'atr_profit_multiplier': row['atr_profit_multiplier'],
                'position_size_pct': row['position_size_pct'],
                'leverage': row['leverage'],
                'performance_metrics': {
                    'sharpe_ratio': row['sharpe_ratio'],
                    'cagr': row['cagr'],
                    'max_drawdown_pct': row['max_drawdown_pct'],
                    'win_rate': row['win_rate'],
                    'total_trades': row['total_trades']
                }
            }
            export_data['top_strategies'].append(strategy_config)
        
        # Save to JSON
        export_path = self.save_path / f"top_{top_n}_strategies.json"
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"🏆 Top {top_n} strategies exported to {export_path}")
        return export_data


# Example usage function
def run_comprehensive_analysis(results_list, output_dir: str = "analysis_output"):
    """
    Run the complete analysis suite on backtest results
    
    Usage:
        results = [...]  # Your list of FixedBacktestResult objects
        run_comprehensive_analysis(results)
    """
    
    # Initialize analyzer
    analyzer = ResultsAnalyzer(results_list, output_dir)
    
    # Save raw data
    analyzer.save_results()
    
    # Generate sweet spots analysis
    print("🎯 Finding parameter sweet spots...")
    sweet_spots = analyzer.find_sweet_spots()
    print(sweet_spots.head())
    
    # Regime analysis
    print("\n🌍 Analyzing regime performance...")
    regime_perf = analyzer.analyze_regime_performance()
    print(regime_perf)
    
    # Generate visualizations
    print("\n📊 Creating visualizations...")
    
    # Heatmap of ATR parameters vs Sharpe ratio
    analyzer.plot_performance_heatmap(
        param1='atr_stop_multiplier',
        param2='atr_profit_multiplier',
        metric='sharpe_ratio',
        symbol='BTC/USD'
    )
    
    # Risk-return scatter
    analyzer.plot_risk_return_scatter(
        x_metric='max_drawdown_pct',
        y_metric='cagr',
        color_metric='sharpe_ratio'
    )
    
    # Regime boxplots
    analyzer.plot_regime_boxplots()
    
    # Generate comprehensive report
    print("\n📄 Generating comprehensive report...")
    analyzer.create_comprehensive_report()
    
    # Export top strategies
    print("\n🏆 Exporting top strategies...")
    top_strategies = analyzer.export_top_strategies()
    
    print(f"\n✅ Analysis complete! Results saved to {output_dir}/")
    return analyzer

# BONUS: Parameter optimization helper
def suggest_next_parameters(analyzer: ResultsAnalyzer, 
                          current_best_metric: str = 'sharpe_ratio') -> Dict[str, Any]:
    """
    AI-powered parameter suggestion for the next round of testing
    Based on current results, suggests promising parameter ranges to explore
    """
    
    # Find the top 10% of configurations
    top_percentile = analyzer.results_df.quantile(0.9)[current_best_metric]
    top_configs = analyzer.results_df[
        analyzer.results_df[current_best_metric] >= top_percentile
    ]
    
    if top_configs.empty:
        return {"message": "No strong patterns found - continue broad exploration"}
    
    # Analyze parameter distributions in top performers
    param_analysis = {}
    key_params = ['signal_threshold', 'atr_stop_multiplier', 'atr_profit_multiplier', 
                  'position_size_pct', 'leverage']
    
    for param in key_params:
        param_stats = top_configs[param].describe()
        param_analysis[param] = {
            'optimal_range': [param_stats['25%'], param_stats['75%']],
            'best_value': top_configs.loc[top_configs[current_best_metric].idxmax(), param],
            'suggested_values': np.linspace(param_stats['25%'], param_stats['75%'], 5).tolist()
        }
    
    suggestions = {
        'analysis_metric': current_best_metric,
        'top_configs_analyzed': len(top_configs),
        'parameter_suggestions': param_analysis,
        'next_test_recommendation': "Focus testing around the optimal ranges identified"
    }
    
    return suggestions