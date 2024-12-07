import numpy as np

def calculate_metrics(results):
    metrics = {}
    
    # Calculate total return from equity curve
    if 'Equity Curve' in results.columns and len(results) > 0:
        initial_value = float(results['Equity Curve'].iloc[0])
        final_value = float(results['Equity Curve'].iloc[-1])
        
        if initial_value > 0:
            total_return = ((final_value - initial_value) / initial_value) * 100
            metrics['total_return'] = total_return
        else:
            metrics['total_return'] = 0
            
    # Calculate Sharpe Ratio (assuming risk-free rate of 0.02)
    if 'Returns' in results.columns:
        returns = results['Returns'].dropna()
        if len(returns) > 0:
            excess_returns = returns - 0.02/252  # Daily risk-free rate
            sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()
            metrics['sharpe_ratio'] = sharpe_ratio
        else:
            metrics['sharpe_ratio'] = 0
    else:
        metrics['sharpe_ratio'] = 0
        
    # Calculate Maximum Drawdown
    if 'Equity Curve' in results.columns:
        equity = results['Equity Curve']
        rolling_max = equity.expanding().max()
        drawdowns = (equity - rolling_max) / rolling_max * 100
        metrics['max_drawdown'] = abs(drawdowns.min()) if len(drawdowns) > 0 else 0
    else:
        metrics['max_drawdown'] = 0
        
    # Calculate Win Rate
    if 'Returns' in results.columns:
        returns = results['Returns'].dropna()
        if len(returns) > 0:
            wins = len(returns[returns > 0])
            total_trades = len(returns[returns != 0])
            metrics['win_rate'] = (wins / total_trades * 100) if total_trades > 0 else 0
        else:
            metrics['win_rate'] = 0
    else:
        metrics['win_rate'] = 0
    
    # Add Sortino Ratio
    if 'Returns' in results.columns:
        returns = results['Returns'].dropna()
        if len(returns) > 0:
            negative_returns = returns[returns < 0]
            downside_std = negative_returns.std()
            if downside_std != 0:
                sortino_ratio = np.sqrt(252) * returns.mean() / downside_std
                metrics['sortino_ratio'] = sortino_ratio
            else:
                metrics['sortino_ratio'] = np.nan
    
    # Add Maximum Consecutive Losses
    if 'Returns' in results.columns:
        returns = results['Returns'].dropna()
        losing_streak = 0
        max_losing_streak = 0
        for ret in returns:
            if ret < 0:
                losing_streak += 1
                max_losing_streak = max(losing_streak, max_losing_streak)
            else:
                losing_streak = 0
        metrics['max_consecutive_losses'] = max_losing_streak
    
    # Calmar Ratio (Return/Max Drawdown)
    if metrics['max_drawdown'] != 0:
        metrics['calmar_ratio'] = metrics['total_return'] / abs(metrics['max_drawdown'])
    
    # Value at Risk (VaR)
    returns = results['Returns'].dropna()
    metrics['var_95'] = np.percentile(returns, 5)
    metrics['var_99'] = np.percentile(returns, 1)
    
    # Maximum Consecutive Wins
    winning_streak = 0
    max_winning_streak = 0
    for ret in returns:
        if ret > 0:
            winning_streak += 1
            max_winning_streak = max(winning_streak, max_winning_streak)
        else:
            winning_streak = 0
    metrics['max_consecutive_wins'] = max_winning_streak
    
    return metrics 