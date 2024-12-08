import numpy as np

def calculate_metrics(results):
    """Calculate trading strategy performance metrics."""
    
    # Calculate returns if not already calculated
    if 'Returns' not in results.columns:
        results['Returns'] = results['Total Value'].pct_change()
    
    # Calculate total return
    total_return = ((results['Total Value'].iloc[-1] / results['Total Value'].iloc[0]) - 1) * 100
    
    # Calculate Sharpe Ratio (assuming risk-free rate of 0.01)
    try:
        excess_returns = results['Returns'] - 0.01/252  # Daily risk-free rate
        returns_std = excess_returns.std()
        if returns_std != 0:
            sharpe_ratio = np.sqrt(252) * excess_returns.mean() / returns_std
        else:
            sharpe_ratio = 0
    except:
        sharpe_ratio = 0
    
    # Calculate Maximum Drawdown
    try:
        cumulative_returns = (1 + results['Returns']).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns / rolling_max - 1
        max_drawdown = drawdowns.min() * 100  # Will be negative for losses
    except:
        max_drawdown = 0
    
    # Calculate Win Rate
    winning_days = len(results[results['Returns'] > 0])
    total_days = len(results[results['Returns'] != 0])  # Only count days with trades
    win_rate = (winning_days / total_days * 100) if total_days > 0 else 0
    
    # Print detailed metrics for debugging
    print("\nDetailed Metrics:")
    print(f"Final Portfolio Value: ${results['Total Value'].iloc[-1]:,.2f}")
    print(f"Initial Portfolio Value: ${results['Total Value'].iloc[0]:,.2f}")
    print(f"Calculated Total Return: {total_return:.2f}%")
    print(f"Average Daily Return: {results['Returns'].mean()*100:.4f}%")
    print(f"Return Std Dev: {results['Returns'].std()*100:.4f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Maximum Drawdown: {max_drawdown:.2f}%")
    print(f"Winning Days: {winning_days}")
    print(f"Total Trading Days: {total_days}")
    print(f"Win Rate: {win_rate:.2f}%")
    
    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate
    } 