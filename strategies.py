import pandas as pd
import numpy as np

class Strategy:
    def __init__(self, position_size=1.0, stop_loss=None, take_profit=None):
        self.position_size = position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
    
    def generate_signals(self, data):
        raise NotImplementedError("Subclass must implement generate_signals")
    
    def apply_risk_management(self, signals, data):
        """Apply stop loss and take profit levels"""
        position = 0
        entry_price = None
        
        for i in range(len(signals)):
            signal_condition = signals.iloc[i] != 0
            no_position = position == 0
            
            if signal_condition & no_position:
                position = float(signals.iloc[i])
                entry_price = float(data['Close'].iloc[i])
            elif position != 0:
                current_price = float(data['Close'].iloc[i])
                pnl = (current_price - entry_price) / entry_price * 100
                
                # Check stop loss using bitwise operators
                if self.stop_loss is not None:
                    stop_loss_hit = (position * pnl) < -abs(self.stop_loss)
                    if stop_loss_hit:
                        signals.iloc[i] = 0
                        position = 0
                        entry_price = None
                
                # Check take profit using bitwise operators
                if self.take_profit is not None:
                    take_profit_hit = (position * pnl) > abs(self.take_profit)
                    if take_profit_hit:
                        signals.iloc[i] = 0
                        position = 0
                        entry_price = None
        
        return signals * self.position_size

class MovingAverageCrossover(Strategy):
    def __init__(self, short_window, long_window):
        super().__init__()  # Call parent class constructor
        self.short_window = short_window
        self.long_window = long_window
    
    def generate_signals(self, data):
        """Generate trading signals based on moving average crossover."""
        signals = pd.DataFrame(index=data.index)
        
        # Calculate moving averages
        short_ma = data['Close'].rolling(window=self.short_window).mean()
        long_ma = data['Close'].rolling(window=self.long_window).mean()
        
        # Initialize signals
        signals['Signal'] = 0
        signals['Short MA'] = short_ma
        signals['Long MA'] = long_ma
        
        # Generate crossover signals
        for i in range(self.long_window + 1, len(data)):
            # Get current and previous values as floats
            curr_short = float(short_ma.iloc[i])
            curr_long = float(long_ma.iloc[i])
            prev_short = float(short_ma.iloc[i-1])
            prev_long = float(long_ma.iloc[i-1])
            
            # Create boolean conditions using bitwise operators
            prev_condition = prev_short <= prev_long
            curr_condition = curr_short > curr_long
            
            # Check for crossovers using bitwise operators
            if (prev_condition) & (curr_condition):  # Bullish crossover
                signals.iloc[i, signals.columns.get_loc('Signal')] = 1
            elif (~prev_condition) & (~curr_condition):  # Bearish crossover
                signals.iloc[i, signals.columns.get_loc('Signal')] = -1
        
        # Add debug information
        print("\nMoving Average Strategy Debug Information:")
        print(f"Data points: {len(signals)}")
        buy_signals = (signals['Signal'] == 1).sum()
        sell_signals = (signals['Signal'] == -1).sum()
        print(f"Buy signals: {buy_signals}")
        print(f"Sell signals: {sell_signals}")
        print(f"First MA crossover at index: {self.long_window}")
        
        # Print first few signals with prices
        signal_points = signals[signals['Signal'] != 0].copy()
        if not signal_points.empty:
            signal_points['Price'] = data['Close']
            print("\nFirst few signals with prices:")
            print(signal_points[['Signal', 'Price', 'Short MA', 'Long MA']].head())
            print("\nSignal distribution:")
            print(signals['Signal'].value_counts())
            
            # Add more detailed debug info with proper float conversion
            print("\nFirst few crossovers:")
            for idx in signal_points.head().index:
                i = signals.index.get_loc(idx)
                print(f"\nAt {idx}:")
                print(f"Short MA: {float(short_ma.iloc[i-1]):.2f} -> {float(short_ma.iloc[i]):.2f}")
                print(f"Long MA: {float(long_ma.iloc[i-1]):.2f} -> {float(long_ma.iloc[i]):.2f}")
                print(f"Signal: {int(signals.iloc[i]['Signal'])}")
        else:
            print("\nNo signals generated!")
            print("Short MA values:", short_ma.head().to_string())
            print("Long MA values:", long_ma.head().to_string())
        
        return signals

class RSIStrategy(Strategy):
    def __init__(self, window=14, oversold=30, overbought=70):
        super().__init__()  # Call parent class constructor
        self.window = window
        self.oversold = oversold
        self.overbought = overbought
    
    def calculate_rsi(self, data):
        """Calculate RSI without using talib"""
        delta = data['Close'].diff()
        
        # Separate gains and losses
        gain = delta.copy()
        loss = delta.copy()
        gain[gain < 0] = 0
        loss[loss > 0] = 0
        loss = abs(loss)
        
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=self.window, min_periods=1).mean()
        avg_loss = loss.rolling(window=self.window, min_periods=1).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # Handle division by zero
        rsi = rsi.replace([np.inf, -np.inf], np.nan)
        rsi = rsi.fillna(50)  # Fill NaN values with neutral RSI
        
        return rsi
    
    def generate_signals(self, data):
        signals = pd.DataFrame(index=data.index)
        
        # Calculate RSI
        rsi = self.calculate_rsi(data)
        
        # Initialize signals
        signals['Signal'] = 0
        
        # Track position to avoid multiple signals
        position = 0
        
        # Generate signals using bitwise operators
        for i in range(len(signals)):
            oversold_condition = float(rsi.iloc[i]) < self.oversold
            overbought_condition = float(rsi.iloc[i]) > self.overbought
            
            if (position <= 0) & oversold_condition:
                signals.iloc[i, signals.columns.get_loc('Signal')] = 1
                position = 1
            elif (position >= 0) & overbought_condition:
                signals.iloc[i, signals.columns.get_loc('Signal')] = -1
                position = -1
            else:
                signals.iloc[i, signals.columns.get_loc('Signal')] = 0
        
        # Store RSI for plotting
        signals['RSI'] = rsi
        
        # Add debug information using proper counting
        buy_signals = (signals['Signal'] == 1).sum()
        sell_signals = (signals['Signal'] == -1).sum()
        print("\nRSI Strategy Debug Information:")
        print(f"Data points: {len(signals)}")
        print(f"Buy signals: {buy_signals}")
        print(f"Sell signals: {sell_signals}")
        print(f"RSI range: {float(rsi.min()):.2f} to {float(rsi.max()):.2f}")
        print(f"Average RSI: {float(rsi.mean()):.2f}")
        
        return signals

class BollingerBandsStrategy(Strategy):
    def __init__(self, bb_window, num_std):
        self.bb_window = bb_window
        self.num_std = num_std
    
    def generate_signals(self, data):
        signals = pd.DataFrame(index=data.index)
        signals['Signal'] = 0.0
        
        # Calculate Bollinger Bands
        signals['MA'] = data['Close'].rolling(window=self.bb_window).mean()
        signals['STD'] = data['Close'].rolling(window=self.bb_window).std()
        signals['Upper'] = signals['MA'] + (signals['STD'] * self.num_std)
        signals['Lower'] = signals['MA'] - (signals['STD'] * self.num_std)
        
        # Generate signals using bitwise operators
        lower_cross = (data['Close'] < signals['Lower'])
        upper_cross = (data['Close'] > signals['Upper'])
        
        # Apply signals
        signals.loc[lower_cross, 'Signal'] = 1.0
        signals.loc[upper_cross, 'Signal'] = -1.0
        
        # Generate trading orders
        signals['Position'] = signals['Signal'].diff()
        
        # Add debug information
        buy_signals = (signals['Signal'] == 1).sum()
        sell_signals = (signals['Signal'] == -1).sum()
        print("\nBollinger Bands Strategy Debug Information:")
        print(f"Data points: {len(signals)}")
        print(f"Buy signals: {buy_signals}")
        print(f"Sell signals: {sell_signals}")
        
        return signals

class MarketRegimeDetector:
    def __init__(self, window=20):
        self.window = window
    
    def detect_regime(self, data):
        # Calculate volatility
        returns = data['Close'].pct_change()
        volatility = returns.rolling(window=self.window).std()
        vol_mean = volatility.mean()
        
        # Calculate trend using bitwise operators
        sma = data['Close'].rolling(window=self.window).mean()
        trend = data['Close'] > sma
        low_vol = volatility < vol_mean
        
        # Define regimes using bitwise operators
        regimes = pd.Series(index=data.index, data='Unknown')
        regimes[trend & low_vol] = 'Uptrend'
        regimes[trend & ~low_vol] = 'Volatile Uptrend'
        regimes[~trend & low_vol] = 'Downtrend'
        regimes[~trend & ~low_vol] = 'Volatile Downtrend'
        
        return regimes 

class TrendFollowingStrategy(Strategy):
    """
    A trend-following strategy that combines multiple technical indicators
    to identify strong trends and generate trading signals.
    """
    def __init__(self, ema_short=20, ema_long=50, atr_period=14, rsi_period=14):
        super().__init__()
        self.ema_short = ema_short
        self.ema_long = ema_long
        self.atr_period = atr_period
        self.rsi_period = rsi_period
    
    def generate_signals(self, data):
        signals = pd.DataFrame(index=data.index)
        signals['Signal'] = 0.0
        
        # Calculate EMAs
        signals['EMA_short'] = data['Close'].ewm(span=self.ema_short, adjust=False).mean()
        signals['EMA_long'] = data['Close'].ewm(span=self.ema_long, adjust=False).mean()
        
        # Calculate ATR
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        signals['ATR'] = true_range.rolling(window=self.atr_period).mean()
        
        # Calculate RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        signals['RSI'] = 100 - (100 / (1 + rs))
        
        # Generate signals based on multiple conditions
        trend_condition = signals['EMA_short'] > signals['EMA_long']
        momentum_condition = signals['RSI'] > 50
        volatility_condition = signals['ATR'] < signals['ATR'].rolling(window=100).mean()
        
        buy_mask = trend_condition & momentum_condition & volatility_condition
        sell_mask = ~trend_condition & ~momentum_condition
        signals.loc[buy_mask, 'Signal'] = 1.0
        signals.loc[sell_mask, 'Signal'] = -1.0
        
        return signals

class MeanReversionStrategy(Strategy):
    """
    A mean reversion strategy that uses Bollinger Bands and RSI
    to identify overbought/oversold conditions.
    """
    def __init__(self, bb_window=20, bb_std=2.0, rsi_window=14, zscore_window=20):
        super().__init__()
        self.bb_window = bb_window
        self.bb_std = bb_std
        self.rsi_window = rsi_window
        self.zscore_window = zscore_window
    
    def generate_signals(self, data):
        signals = pd.DataFrame(index=data.index)
        signals['Signal'] = 0.0
        
        # Calculate Bollinger Bands
        signals['MA'] = data['Close'].rolling(window=self.bb_window).mean()
        signals['STD'] = data['Close'].rolling(window=self.bb_window).std()
        signals['Upper'] = signals['MA'] + (signals['STD'] * self.bb_std)
        signals['Lower'] = signals['MA'] - (signals['STD'] * self.bb_std)
        
        # Calculate RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_window).mean()
        rs = gain / loss
        signals['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate z-score
        returns = data['Close'].pct_change()
        signals['ZScore'] = (returns - returns.rolling(window=self.zscore_window).mean()) / \
                           returns.rolling(window=self.zscore_window).std()
        
        # Generate signals
        oversold = (data['Close'] < signals['Lower']) & \
                  (signals['RSI'] < 30) & \
                  (signals['ZScore'] < -2)
        
        overbought = (data['Close'] > signals['Upper']) & \
                    (signals['RSI'] > 70) & \
                    (signals['ZScore'] > 2)
        
        signals.loc[oversold, 'Signal'] = 1.0
        signals.loc[overbought, 'Signal'] = -1.0
        
        return signals

class StatisticalArbitrageStrategy(Strategy):
    """
    A pairs trading strategy that looks for statistical arbitrage
    opportunities between correlated assets.
    """
    def __init__(self, window=60, entry_zscore=2.0, exit_zscore=0.0):
        super().__init__()
        self.window = window
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
    
    def calculate_spread(self, price1, price2):
        # Calculate the spread between two price series
        spread = np.log(price1) - np.log(price2)
        zscore = (spread - spread.rolling(window=self.window).mean()) / \
                 spread.rolling(window=self.window).std()
        return zscore
    
    def generate_signals(self, data):
        signals = pd.DataFrame(index=data.index)
        signals['Signal'] = 0.0
        
        # For demonstration, we'll use a simple ratio of High/Low prices
        # In practice, you would use two different but correlated assets
        zscore = self.calculate_spread(data['High'], data['Low'])
        
        # Generate signals
        signals.loc[zscore > self.entry_zscore, 'Signal'] = -1.0  # Short the spread
        signals.loc[zscore < -self.entry_zscore, 'Signal'] = 1.0  # Long the spread
        signals.loc[abs(zscore) < self.exit_zscore, 'Signal'] = 0.0  # Exit position
        
        return signals