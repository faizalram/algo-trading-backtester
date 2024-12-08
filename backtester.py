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
        self.transaction_costs = self.commission + self.slippage  # Combined transaction costs
    
    def calculate_position_size(self, capital, price):
        """Calculate the number of shares to trade based on position sizing rules"""
        # Use total capital instead of available cash for position sizing
        position_value = capital * self.max_position_size * self.position_size
        shares = int(position_value / price)
        return max(shares, 0)  # Ensure non-negative number of shares
    
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
        results['Trade'] = 0  # Track when trades occur
        
        # Track portfolio changes
        position = 0
        trades_made = 0
        
        # Add debug counters
        buy_attempts = 0
        sell_attempts = 0
        successful_buys = 0
        successful_sells = 0
        
        # Simulate trading
        for i in range(len(results)):
            # Update position based on signal
            if i > 0:  # Copy previous day's position and values
                results.iloc[i, results.columns.get_loc('Position')] = results.iloc[i-1, results.columns.get_loc('Position')]
                results.iloc[i, results.columns.get_loc('Shares')] = results.iloc[i-1, results.columns.get_loc('Shares')]
                results.iloc[i, results.columns.get_loc('Cash')] = results.iloc[i-1, results.columns.get_loc('Cash')]
            
            current_cash = results.iloc[i, results.columns.get_loc('Cash')]
            current_price = results.iloc[i, results.columns.get_loc('Price')]
            current_total = current_cash  # Include current holdings in total capital
            
            if i > 0:
                current_total += results.iloc[i-1, results.columns.get_loc('Holdings')]
            
            # Check for trade signals
            if results.iloc[i, results.columns.get_loc('Signal')] == 1 and position <= 0:  # Buy signal
                buy_attempts += 1
                # Calculate number of shares to buy using total capital for position sizing
                shares_to_buy = self.calculate_position_size(current_total, current_price)
                
                if shares_to_buy > 0:
                    # Calculate total cost including transaction costs
                    total_cost = shares_to_buy * current_price * (1 + self.transaction_costs)
                    
                    if total_cost <= current_cash:  # Check if we have enough cash
                        # Update position and holdings
                        results.iloc[i, results.columns.get_loc('Position')] = 1
                        results.iloc[i, results.columns.get_loc('Shares')] = shares_to_buy
                        results.iloc[i, results.columns.get_loc('Cash')] = current_cash - total_cost
                        results.iloc[i, results.columns.get_loc('Trade')] = 1
                        position = 1
                        successful_buys += 1
                        print(f"\nBuy executed at {data.index[i]}:")
                        print(f"Price: ${current_price:.2f}")
                        print(f"Shares: {shares_to_buy}")
                        print(f"Total Cost: ${total_cost:.2f}")
                        
            elif results.iloc[i, results.columns.get_loc('Signal')] == -1 and position >= 0:  # Sell signal
                sell_attempts += 1
                shares_to_sell = results.iloc[i, results.columns.get_loc('Shares')]
                if shares_to_sell > 0:
                    # Calculate total proceeds including transaction costs
                    total_proceeds = shares_to_sell * current_price * (1 - self.transaction_costs)
                    
                    # Update position and cash
                    results.iloc[i, results.columns.get_loc('Position')] = -1
                    results.iloc[i, results.columns.get_loc('Shares')] = 0
                    results.iloc[i, results.columns.get_loc('Cash')] = current_cash + total_proceeds
                    results.iloc[i, results.columns.get_loc('Trade')] = -1
                    position = -1
                    successful_sells += 1
                    print(f"\nSell executed at {data.index[i]}:")
                    print(f"Price: ${current_price:.2f}")
                    print(f"Shares: {shares_to_sell}")
                    print(f"Total Proceeds: ${total_proceeds:.2f}")
            
            # Calculate holdings value and total portfolio value
            current_shares = results.iloc[i, results.columns.get_loc('Shares')]
            holdings_value = current_shares * current_price
            results.iloc[i, results.columns.get_loc('Holdings')] = holdings_value
            
            total_value = holdings_value + results.iloc[i, results.columns.get_loc('Cash')]
            results.iloc[i, results.columns.get_loc('Total Value')] = total_value
        
        # Calculate returns
        results['Returns'] = results['Total Value'].pct_change()
        
        # Calculate equity curve
        results['Equity Curve'] = (1 + results['Returns']).cumprod() * self.initial_capital
        
        # Print summary statistics
        print(f"Total trades made: {trades_made}")
        print(f"Final portfolio value: ${results['Total Value'].iloc[-1]:,.2f}")
        print(f"Total return: {((results['Total Value'].iloc[-1] / self.initial_capital) - 1) * 100:.2f}%")
        
        # Add debug information
        if trades_made == 0:
            print("\nDebug information:")
            print(f"Number of buy signals: {len(results[results['Signal'] == 1])}")
            print(f"Number of sell signals: {len(results[results['Signal'] == -1])}")
            print("Check if strategy is generating signals correctly.")
        
        # Print trading statistics
        print("\nTrading Statistics:")
        print(f"Buy attempts: {buy_attempts}")
        print(f"Successful buys: {successful_buys}")
        print(f"Sell attempts: {sell_attempts}")
        print(f"Successful sells: {successful_sells}")
        print(f"Success rate: {((successful_buys + successful_sells) / (buy_attempts + sell_attempts) * 100):.2f}% if signals were generated")
        
        return results