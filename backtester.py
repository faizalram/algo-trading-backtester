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
        try:
            # Use total capital instead of available cash for position sizing
            position_value = float(capital) * float(self.max_position_size) * float(self.position_size)
            shares = int(position_value / float(price))
            return max(shares, 0)  # Ensure non-negative number of shares
        except Exception as e:
            print(f"Error in position sizing: {str(e)}")
            return 0
    
    def run(self, data):
        """Run the backtest simulation."""
        try:
            # Validate input data
            print("\nValidating input data...")
            if data is None or len(data) == 0:
                raise ValueError("Empty or invalid input data")
            
            print(f"Input data shape: {data.shape}")
            print(f"Input data columns: {data.columns.tolist()}")
            print(f"Input data range: {data.index[0]} to {data.index[-1]}")
            
            # Generate trading signals
            print("\nGenerating signals...")
            signals = self.strategy.generate_signals(data)
            
            if signals is None or len(signals) == 0:
                raise ValueError("No signals generated")
            
            print(f"Generated signals shape: {signals.shape}")
            print(f"Signal values: {signals['Signal'].value_counts().to_dict()}")
            
            # Initialize results DataFrame
            results = pd.DataFrame(index=data.index)
            results['Signal'] = signals['Signal'].astype(float)
            results['Price'] = data['Close'].astype(float)
            
            # Initialize portfolio metrics
            results['Position'] = 0.0
            results['Shares'] = 0.0
            results['Cash'] = float(self.initial_capital)
            results['Holdings'] = 0.0
            results['Total Value'] = float(self.initial_capital)
            
            # Track portfolio changes
            position = 0.0
            trades_made = 0
            
            # Simulate trading
            print("\nSimulating trades...")
            for i in range(len(results)):
                try:
                    current_cash = float(results.iloc[i, results.columns.get_loc('Cash')])
                    current_price = float(results.iloc[i, results.columns.get_loc('Price')])
                    current_signal = float(results.iloc[i, results.columns.get_loc('Signal')])
                    
                    # Copy previous position if not first day
                    if i > 0:
                        results.iloc[i, results.columns.get_loc('Position')] = float(results.iloc[i-1, results.columns.get_loc('Position')])
                        results.iloc[i, results.columns.get_loc('Shares')] = float(results.iloc[i-1, results.columns.get_loc('Shares')])
                        results.iloc[i, results.columns.get_loc('Cash')] = float(results.iloc[i-1, results.columns.get_loc('Cash')])
                    
                    # Process signals
                    if current_signal == 1.0 and position <= 0:  # Buy signal
                        shares_to_buy = self.calculate_position_size(current_cash, current_price)
                        if shares_to_buy > 0:
                            total_cost = shares_to_buy * current_price * (1 + self.transaction_costs)
                            if total_cost <= current_cash:
                                results.iloc[i, results.columns.get_loc('Position')] = 1.0
                                results.iloc[i, results.columns.get_loc('Shares')] = float(shares_to_buy)
                                results.iloc[i, results.columns.get_loc('Cash')] = current_cash - total_cost
                                position = 1.0
                                trades_made += 1
                                print(f"\nBuy executed at {data.index[i]}: Price=${current_price:.2f}, Shares={shares_to_buy}")
                    
                    elif current_signal == -1.0 and position >= 0:  # Sell signal
                        shares_to_sell = float(results.iloc[i, results.columns.get_loc('Shares')])
                        if shares_to_sell > 0:
                            total_proceeds = shares_to_sell * current_price * (1 - self.transaction_costs)
                            results.iloc[i, results.columns.get_loc('Position')] = -1.0
                            results.iloc[i, results.columns.get_loc('Shares')] = 0.0
                            results.iloc[i, results.columns.get_loc('Cash')] = current_cash + total_proceeds
                            position = -1.0
                            trades_made += 1
                            print(f"\nSell executed at {data.index[i]}: Price=${current_price:.2f}, Shares={shares_to_sell}")
                    
                    # Update portfolio value
                    current_shares = float(results.iloc[i, results.columns.get_loc('Shares')])
                    holdings_value = current_shares * current_price
                    results.iloc[i, results.columns.get_loc('Holdings')] = holdings_value
                    results.iloc[i, results.columns.get_loc('Total Value')] = holdings_value + float(results.iloc[i, results.columns.get_loc('Cash')])
                
                except Exception as e:
                    print(f"Error processing trade at index {i}: {str(e)}")
                    raise
            
            # Calculate performance metrics
            results['Returns'] = results['Total Value'].pct_change().fillna(0)
            results['Equity Curve'] = (1 + results['Returns']).cumprod() * self.initial_capital
            
            # Print summary
            print("\nBacktest Summary:")
            print(f"Total trades executed: {trades_made}")
            print(f"Final portfolio value: ${results['Total Value'].iloc[-1]:,.2f}")
            print(f"Total return: {((results['Total Value'].iloc[-1] / self.initial_capital) - 1) * 100:.2f}%")
            
            return results
            
        except Exception as e:
            print(f"\nError in backtester: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            print(f"Traceback:\n{traceback.format_exc()}")
            raise