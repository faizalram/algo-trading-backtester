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
        """Run the backtest simulation."""
        try:
            # Generate trading signals
            print("\nGenerating signals...")
            signals = self.strategy.generate_signals(data)
            
            # Convert signals to numeric values and handle NaN
            signals['Signal'] = pd.to_numeric(signals['Signal'], errors='coerce').fillna(0)
            
            # Initialize results DataFrame
            results = pd.DataFrame(index=data.index)
            results['Signal'] = signals['Signal'].astype(float)  # Ensure float type
            results['Price'] = data['Close'].astype(float)  # Ensure float type
            
            # Initialize portfolio metrics with explicit types
            results['Position'] = 0.0  # Use float instead of int
            results['Shares'] = 0.0
            results['Cash'] = float(self.initial_capital)
            results['Holdings'] = 0.0
            results['Total Value'] = float(self.initial_capital)
            results['Trade'] = 0.0
            
            # Track portfolio changes
            position = 0.0
            trades_made = 0
            
            # Add debug counters
            buy_attempts = 0
            sell_attempts = 0
            successful_buys = 0
            successful_sells = 0
            
            # Simulate trading
            for i in range(len(results)):
                try:
                    # Update position based on signal
                    if i > 0:  # Copy previous day's position and values
                        results.iloc[i, results.columns.get_loc('Position')] = float(results.iloc[i-1, results.columns.get_loc('Position')])
                        results.iloc[i, results.columns.get_loc('Shares')] = float(results.iloc[i-1, results.columns.get_loc('Shares')])
                        results.iloc[i, results.columns.get_loc('Cash')] = float(results.iloc[i-1, results.columns.get_loc('Cash')])
                    
                    current_cash = float(results.iloc[i, results.columns.get_loc('Cash')])
                    current_price = float(results.iloc[i, results.columns.get_loc('Price')])
                    current_total = current_cash
                    
                    if i > 0:
                        current_total += float(results.iloc[i-1, results.columns.get_loc('Holdings')])
                    
                    # Get current signal
                    current_signal = float(results.iloc[i, results.columns.get_loc('Signal')])
                    
                    # Check for trade signals
                    if current_signal == 1.0 and position <= 0:  # Buy signal
                        buy_attempts += 1
                        shares_to_buy = self.calculate_position_size(current_total, current_price)
                        
                        if shares_to_buy > 0:
                            total_cost = shares_to_buy * current_price * (1 + self.transaction_costs)
                            
                            if total_cost <= current_cash:
                                results.iloc[i, results.columns.get_loc('Position')] = 1.0
                                results.iloc[i, results.columns.get_loc('Shares')] = float(shares_to_buy)
                                results.iloc[i, results.columns.get_loc('Cash')] = current_cash - total_cost
                                results.iloc[i, results.columns.get_loc('Trade')] = 1.0
                                position = 1.0
                                successful_buys += 1
                                trades_made += 1
                    
                    elif current_signal == -1.0 and position >= 0:  # Sell signal
                        sell_attempts += 1
                        shares_to_sell = float(results.iloc[i, results.columns.get_loc('Shares')])
                        
                        if shares_to_sell > 0:
                            total_proceeds = shares_to_sell * current_price * (1 - self.transaction_costs)
                            results.iloc[i, results.columns.get_loc('Position')] = -1.0
                            results.iloc[i, results.columns.get_loc('Shares')] = 0.0
                            results.iloc[i, results.columns.get_loc('Cash')] = current_cash + total_proceeds
                            results.iloc[i, results.columns.get_loc('Trade')] = -1.0
                            position = -1.0
                            successful_sells += 1
                            trades_made += 1
                    
                    # Calculate holdings value and total portfolio value
                    current_shares = float(results.iloc[i, results.columns.get_loc('Shares')])
                    holdings_value = current_shares * current_price
                    results.iloc[i, results.columns.get_loc('Holdings')] = holdings_value
                    total_value = holdings_value + float(results.iloc[i, results.columns.get_loc('Cash')])
                    results.iloc[i, results.columns.get_loc('Total Value')] = total_value
                    
                except Exception as e:
                    print(f"Error at index {i}: {str(e)}")
                    raise
            
            # Calculate returns
            results['Returns'] = results['Total Value'].pct_change().fillna(0)
            
            # Calculate equity curve
            results['Equity Curve'] = (1 + results['Returns']).cumprod() * self.initial_capital
            
            # Print trading statistics
            print("\nTrading Statistics:")
            print(f"Buy attempts: {buy_attempts}")
            print(f"Successful buys: {successful_buys}")
            print(f"Sell attempts: {sell_attempts}")
            print(f"Successful sells: {successful_sells}")
            print(f"Total trades made: {trades_made}")
            print(f"Final portfolio value: ${results['Total Value'].iloc[-1]:,.2f}")
            print(f"Total return: {((results['Total Value'].iloc[-1] / self.initial_capital) - 1) * 100:.2f}%")
            
            return results
            
        except Exception as e:
            print(f"Error in backtester: {str(e)}")
            raise