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
            if signals.iloc[i] != 0 and position == 0:
                position = signals.iloc[i]
                entry_price = data['Close'].iloc[i]
            elif position != 0:
                current_price = data['Close'].iloc[i]
                pnl = (current_price - entry_price) / entry_price * 100
                
                # Check stop loss
                if self.stop_loss and position * pnl < -abs(self.stop_loss):
                    signals.iloc[i] = 0
                    position = 0
                    entry_price = None
                
                # Check take profit
                if self.take_profit and position * pnl > abs(self.take_profit):
                    signals.iloc[i] = 0
                    position = 0
                    entry_price = None
        
        return signals * self.position_size

class MovingAverageCrossover(Strategy):
    def __init__(self, short_window, long_window):
        self.short_window = short_window
        self.long_window = long_window
    
    def generate_signals(self, data):
        signals = pd.DataFrame(index=data.index)
        signals['Signal'] = 0.0
        
        # Calculate moving averages
        signals['Short MA'] = data['Close'].rolling(window=self.short_window).mean()
        signals['Long MA'] = data['Close'].rolling(window=self.long_window).mean()
        
        # Generate signals
        signals['Signal'] = 0.0
        mask = signals.index >= signals.index[self.long_window]
        condition = signals['Short MA'] > signals['Long MA']
        signals.loc[mask, 'Signal'] = np.where(condition[mask], 1.0, -1.0)
        
        # Generate trading orders
        signals['Position'] = signals['Signal'].diff()
        
        return signals

class RSIStrategy(Strategy):
    def __init__(self, window, oversold, overbought):
        self.window = window
        self.oversold = oversold
        self.overbought = overbought
    
    def generate_signals(self, data):
        signals = pd.DataFrame(index=data.index)
        signals['Signal'] = 0.0
        
        # Calculate RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.window).mean()
        rs = gain / loss
        signals['RSI'] = 100 - (100 / (1 + rs))
        
        # Generate signals
        signals['Signal'] = pd.Series(0.0, index=data.index)
        mask_oversold = signals['RSI'] < self.oversold
        mask_overbought = signals['RSI'] > self.overbought
        signals.loc[mask_oversold, 'Signal'] = 1.0
        signals.loc[mask_overbought, 'Signal'] = -1.0
        
        # Generate trading orders
        signals['Position'] = signals['Signal'].diff()
        
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
        
        # Generate signals
        signals['Signal'] = pd.Series(0.0, index=data.index)
        mask_lower = data['Close'] < signals['Lower']
        mask_upper = data['Close'] > signals['Upper']
        signals.loc[mask_lower, 'Signal'] = 1.0
        signals.loc[mask_upper, 'Signal'] = -1.0
        
        # Generate trading orders
        signals['Position'] = signals['Signal'].diff()
        
        return signals 

class MarketRegimeDetector:
    def __init__(self, window=20):
        self.window = window
    
    def detect_regime(self, data):
        # Calculate volatility
        returns = data['Close'].pct_change()
        volatility = returns.rolling(window=self.window).std()
        
        # Calculate trend
        sma = data['Close'].rolling(window=self.window).mean()
        trend = data['Close'] > sma
        
        # Define regimes
        regimes = pd.Series(index=data.index, data='Unknown')
        regimes[trend & (volatility < volatility.mean())] = 'Uptrend'
        regimes[trend & (volatility >= volatility.mean())] = 'Volatile Uptrend'
        regimes[~trend & (volatility < volatility.mean())] = 'Downtrend'
        regimes[~trend & (volatility >= volatility.mean())] = 'Volatile Downtrend'
        
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