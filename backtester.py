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
        # Create a copy of the data to avoid modifying the original
        results = pd.DataFrame(index=data.index)
        
        # Generate signals - ensure we get a Series
        signals = self.strategy.generate_signals(data)
        if isinstance(signals, pd.DataFrame):
            results['Signal'] = signals['Signal']
        else:
            results['Signal'] = signals
        
        # Initialize columns
        results['Position'] = 0  # Current position
        results['Shares'] = 0    # Number of shares held
        results['Cash'] = self.initial_capital  # Available cash
        results['Holdings'] = 0   # Value of holdings
        results['Total Value'] = self.initial_capital  # Total portfolio value
        results['Returns'] = 0.0  # Period returns
        
        for i in range(1, len(results)):
            current_price = data['Close'].iloc[i]
            prev_price = data['Close'].iloc[i-1]
            current_signal = results['Signal'].iloc[i]
            prev_signal = results['Signal'].iloc[i-1]
            
            # Copy previous values
            results.loc[results.index[i], 'Cash'] = results['Cash'].iloc[i-1]
            results.loc[results.index[i], 'Shares'] = results['Shares'].iloc[i-1]
            results.loc[results.index[i], 'Position'] = results['Position'].iloc[i-1]
            
            # Check for position changes
            if current_signal != prev_signal:
                # Close existing position if any
                if results['Shares'].iloc[i-1] != 0:
                    # Calculate transaction costs for closing
                    close_value = abs(results['Shares'].iloc[i-1] * current_price)
                    transaction_cost = close_value * (self.commission + self.slippage)
                    
                    # Update cash
                    if results['Position'].iloc[i-1] > 0:  # Long position
                        results.loc[results.index[i], 'Cash'] += close_value - transaction_cost
                    else:  # Short position
                        results.loc[results.index[i], 'Cash'] -= close_value + transaction_cost
                    
                    # Reset shares
                    results.loc[results.index[i], 'Shares'] = 0
                    results.loc[results.index[i], 'Position'] = 0
                
                # Open new position if signal is not zero
                if current_signal != 0:
                    # Calculate position size
                    available_capital = results['Cash'].iloc[i]
                    shares = self.calculate_position_size(available_capital, current_price)
                    
                    # Calculate transaction costs for opening
                    open_value = abs(shares * current_price)
                    transaction_cost = open_value * (self.commission + self.slippage)
                    
                    if current_signal > 0:  # Long position
                        if open_value + transaction_cost <= available_capital:
                            results.loc[results.index[i], 'Cash'] -= (open_value + transaction_cost)
                            results.loc[results.index[i], 'Shares'] = shares
                            results.loc[results.index[i], 'Position'] = 1
                    else:  # Short position
                        results.loc[results.index[i], 'Cash'] += (open_value - transaction_cost)
                        results.loc[results.index[i], 'Shares'] = -shares
                        results.loc[results.index[i], 'Position'] = -1
            
            # Calculate holdings value
            results.loc[results.index[i], 'Holdings'] = results['Shares'].iloc[i] * current_price
            
            # Calculate total value
            results.loc[results.index[i], 'Total Value'] = results['Cash'].iloc[i] + results['Holdings'].iloc[i]
            
            # Calculate returns
            results.loc[results.index[i], 'Returns'] = (
                results['Total Value'].iloc[i] / results['Total Value'].iloc[i-1] - 1
            )
        
        # Calculate equity curve from returns
        results['Equity Curve'] = (1 + results['Returns']).cumprod() * self.initial_capital
        
        # Fill any NaN values in Equity Curve with initial_capital
        results['Equity Curve'] = results['Equity Curve'].fillna(self.initial_capital)
        
        return results