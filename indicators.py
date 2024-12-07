class TechnicalIndicators:
    @staticmethod
    def macd(data, fast=12, slow=26, signal=9):
        exp1 = data['Close'].ewm(span=fast, adjust=False).mean()
        exp2 = data['Close'].ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line
    
    @staticmethod
    def stochastic(data, k_window=14, d_window=3):
        low_min = data['Low'].rolling(window=k_window).min()
        high_max = data['High'].rolling(window=k_window).max()
        k = 100 * (data['Close'] - low_min) / (high_max - low_min)
        d = k.rolling(window=d_window).mean()
        return k, d 