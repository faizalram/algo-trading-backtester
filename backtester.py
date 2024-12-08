import pandas as pd
import numpy as np

class Backtester:
    def __init__(self, strategy, initial_capital=100000, position_size=1.0, max_position_size=0.2):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.max_position_size = max_position_size  # Maximum % of capital per trade
        self.commission = 0.001  # 0.1% commission per trade
        self.slippage = 0.001   # 0.1% slippage per trade
    
    def calculate_position_size(self, capital, price):
        max_shares = (capital * self.max_position_size) / price
        return int(max_shares * self.position_size)
    
    def run(self, data):
        """
        Run the backtest simulation.
        
        Args:
            data (pd.DataFrame): Historical price data with OHLCV columns
            
        Returns:
            pd.DataFrame: Results of the backtest including signals and portfolio value
        """
        # Generate trading signals
        signals = self.strategy.generate_signals(data)
        
        # Initialize results DataFrame
        results = pd.DataFrame(index=data.index)
        results['Signal'] = signals['Signal']
        results['Price'] = data['Close']
        
        # Initialize portfolio metrics
        results['Position'] = 0  # Current position (1: long, -1: short, 0: neutral)
        results['Shares'] = 0  # Number of shares held
        results['Cash'] = self.initial_capital  # Available cash
        results['Holdings'] = 0  # Value of holdings
        results['Total Value'] = self.initial_capital  # Total portfolio value
        
        # Track portfolio changes
        position = 0
        
        # Simulate trading
        for i in range(len(results)):
            # Update position based on signal
            if i > 0:  # Copy previous day's position and values
                results.iloc[i, results.columns.get_loc('Position')] = results.iloc[i-1, results.columns.get_loc('Position')]
                results.iloc[i, results.columns.get_loc('Shares')] = results.iloc[i-1, results.columns.get_loc('Shares')]
                results.iloc[i, results.columns.get_loc('Cash')] = results.iloc[i-1, results.columns.get_loc('Cash')]
            
            # Check for trade signals
            if results.iloc[i, results.columns.get_loc('Signal')] == 1 and position <= 0:  # Buy signal
                # Calculate number of shares to buy
                available_cash = results.iloc[i, results.columns.get_loc('Cash')]
                price = results.iloc[i, results.columns.get_loc('Price')]
                shares_to_buy = (available_cash * (1 - self.transaction_costs)) // price
                
                if shares_to_buy > 0:
                    # Update position and holdings
                    results.iloc[i, results.columns.get_loc('Position')] = 1
                    results.iloc[i, results.columns.get_loc('Shares')] = shares_to_buy
                    results.iloc[i, results.columns.get_loc('Cash')] -= shares_to_buy * price * (1 + self.transaction_costs)
                    position = 1
                    
            elif results.iloc[i, results.columns.get_loc('Signal')] == -1 and position >= 0:  # Sell signal
                shares_to_sell = results.iloc[i, results.columns.get_loc('Shares')]
                if shares_to_sell > 0:
                    # Update position and cash
                    price = results.iloc[i, results.columns.get_loc('Price')]
                    results.iloc[i, results.columns.get_loc('Position')] = -1
                    results.iloc[i, results.columns.get_loc('Shares')] = 0
                    results.iloc[i, results.columns.get_loc('Cash')] += shares_to_sell * price * (1 - self.transaction_costs)
                    position = -1
            
            # Calculate holdings value and total portfolio value
            results.iloc[i, results.columns.get_loc('Holdings')] = (
                results.iloc[i, results.columns.get_loc('Shares')] * 
                results.iloc[i, results.columns.get_loc('Price')]
            )
            results.iloc[i, results.columns.get_loc('Total Value')] = (
                results.iloc[i, results.columns.get_loc('Holdings')] + 
                results.iloc[i, results.columns.get_loc('Cash')]
            )
        
        # Calculate returns
        results['Returns'] = results['Total Value'].pct_change()
        
        # Calculate equity curve
        results['Equity Curve'] = (1 + results['Returns']).cumprod() * self.initial_capital
        
        return results