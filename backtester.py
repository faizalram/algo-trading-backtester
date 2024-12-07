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
            # If signals is a DataFrame, extract the signal column
            results['Signal'] = signals['Signal']
        else:
            # If signals is already a Series
            results['Signal'] = signals
        
        # Calculate position changes
        position_changes = results['Signal'].diff()
        
        # Calculate transaction costs
        transaction_costs = abs(position_changes) * (self.commission + self.slippage)
        
        # Calculate position sizes and returns
        results['Position Size'] = 0.0
        current_capital = self.initial_capital
        
        for i in range(len(results)):
            if position_changes.iloc[i] != 0:
                # Calculate new position size when position changes
                price = data['Close'].iloc[i]
                position_size = self.calculate_position_size(current_capital, price)
                results.loc[results.index[i], 'Position Size'] = position_size * results['Signal'].iloc[i]
            else:
                # Maintain previous position size
                results.loc[results.index[i], 'Position Size'] = results['Position Size'].iloc[i-1] if i > 0 else 0
        
        # Calculate returns including transaction costs
        price_changes = data['Close'].pct_change()
        results['Returns'] = (results['Position Size'].shift(1) / self.initial_capital) * price_changes - transaction_costs
        
        # Initialize the equity curve with initial capital
        results['Equity Curve'] = self.initial_capital
        
        # Calculate equity curve
        results['Equity Curve'] = self.initial_capital * (1 + results['Returns']).cumprod()
        
        # Fill any NaN values in Equity Curve with initial_capital
        results['Equity Curve'] = results['Equity Curve'].fillna(self.initial_capital)
        
        return results