import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from strategies import MovingAverageCrossover, RSIStrategy, BollingerBandsStrategy, TrendFollowingStrategy, MeanReversionStrategy, StatisticalArbitrageStrategy
from backtester import Backtester
from metrics import calculate_metrics
from optimizer import optimize_strategy
import numpy as np
import io

st.set_page_config(page_title="IHSG Trading Strategy Tester", layout="wide")

st.title("IHSG Trading Strategy Tester")

# Sidebar inputs
st.sidebar.header("Strategy Parameters")

# Stock selection
symbol = st.sidebar.text_input("Stock Symbol (e.g., BBCA.JK)", "^JKSE")

# Date range selection
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-03-01"))

# Add input validation
if start_date >= end_date:
    st.error("Start date must be before end date")
    st.stop()

# Strategy selection
strategy_options = {
    "Basic Strategies": [
        "Moving Average Crossover",
        "RSI",
        "Bollinger Bands"
    ],
    "Advanced Strategies": [
        "Trend Following",
        "Mean Reversion",
        "Statistical Arbitrage"
    ]
}

# Flatten options for selectbox while keeping track of categories
strategy_list = []
for category, strategies in strategy_options.items():
    strategy_list.extend([f"{category} - {strategy}" for strategy in strategies])

strategy_type = st.sidebar.selectbox(
    "Select Strategy",
    strategy_list
)

# Extract actual strategy name without category prefix
strategy_type = strategy_type.split(" - ")[1]

# Create detailed strategy descriptions
strategy_info = {
    "Moving Average Crossover": {
        "description": """
        The Moving Average Crossover strategy is a trend-following strategy that uses two moving averages 
        to identify potential trading signals. When the shorter-term MA crosses above the longer-term MA, 
        it generates a buy signal, and when it crosses below, it generates a sell signal.
        """,
        "parameters": {
            "short_window": "Period for the short-term moving average. Shorter periods are more responsive to price changes.",
            "long_window": "Period for the long-term moving average. Longer periods help identify the overall trend."
        },
        "pros": [
            "Simple and easy to understand",
            "Works well in trending markets",
            "Reduces noise in price action"
        ],
        "cons": [
            "Can generate false signals in sideways markets",
            "Lag in signal generation due to moving average calculation",
            "Performance depends heavily on parameter selection"
        ]
    },
    "RSI": {
        "description": """
        The Relative Strength Index (RSI) strategy is a momentum oscillator that measures the speed and 
        magnitude of recent price changes to evaluate overbought or oversold conditions. It generates 
        buy signals when the market is oversold and sell signals when overbought.
        """,
        "parameters": {
            "window": "Period for RSI calculation. Standard is 14 days.",
            "oversold": "Level below which the market is considered oversold (typically 30)",
            "overbought": "Level above which the market is considered overbought (typically 70)"
        },
        "pros": [
            "Effective in identifying potential reversal points",
            "Works well in ranging markets",
            "Provides clear overbought/oversold signals"
        ],
        "cons": [
            "Can stay in overbought/oversold territory during strong trends",
            "May miss strong trends while waiting for reversal signals",
            "Requires careful parameter tuning"
        ]
    },
    "Bollinger Bands": {
        "description": """
        Bollinger Bands is a volatility-based strategy that uses a moving average with upper and lower bands 
        set at standard deviation levels. The strategy generates buy signals when price touches the lower band 
        (potentially oversold) and sell signals when price touches the upper band (potentially overbought).
        """,
        "parameters": {
            "bb_window": "Period for calculating the moving average and standard deviation bands",
            "num_std": "Number of standard deviations for band width - higher values create wider bands"
        },
        "pros": [
            "Adapts automatically to market volatility",
            "Provides dynamic support and resistance levels",
            "Effective in both trending and ranging markets"
        ],
        "cons": [
            "Can generate false signals in strong trends",
            "Requires careful parameter selection based on market conditions",
            "May not work well in low volatility periods"
        ]
    },
    "Trend Following": {
        "description": """
        A comprehensive trend-following strategy that combines multiple technical indicators to identify 
        and follow strong market trends. It uses EMAs for trend direction, ATR for volatility measurement, 
        and RSI for momentum confirmation, creating a robust system for trend identification.
        """,
        "parameters": {
            "ema_short": "Short-term EMA period for quick trend changes",
            "ema_long": "Long-term EMA period for overall trend direction",
            "atr_period": "ATR period for volatility measurement",
            "rsi_period": "RSI period for momentum confirmation"
        },
        "pros": [
            "Captures major market trends effectively",
            "Multiple confirmation signals reduce false signals",
            "Built-in volatility adaptation",
            "Combines trend, momentum, and volatility analysis"
        ],
        "cons": [
            "More complex to optimize and maintain",
            "May lag in rapid market reversals",
            "Multiple parameters require careful balancing",
            "Can underperform in choppy markets"
        ]
    },
    "Mean Reversion": {
        "description": """
        A sophisticated mean reversion strategy that combines Bollinger Bands, RSI, and Z-score analysis 
        to identify statistical extremes in price movements. The strategy looks for oversold conditions 
        to buy and overbought conditions to sell, with multiple confirmations required for signals.
        """,
        "parameters": {
            "bb_window": "Bollinger Bands calculation period",
            "bb_std": "Standard deviations for Bollinger Bands",
            "rsi_window": "RSI calculation period",
            "zscore_window": "Period for Z-score calculations"
        },
        "pros": [
            "Strong statistical foundation",
            "Multiple confirmation signals",
            "Effective in range-bound markets",
            "Built-in risk management through multiple indicators"
        ],
        "cons": [
            "Can face losses in strong trending markets",
            "Requires stable market conditions",
            "Complex parameter optimization needed",
            "May have longer waiting periods between trades"
        ]
    },
    "Statistical Arbitrage": {
        "description": """
        A quantitative trading strategy that identifies and exploits statistical relationships between 
        prices. It uses Z-scores to detect when price relationships deviate significantly from their 
        historical norms, trading on the assumption that these deviations will eventually normalize.
        """,
        "parameters": {
            "window": "Lookback period for statistical calculations",
            "entry_zscore": "Z-score threshold for entering trades",
            "exit_zscore": "Z-score threshold for exiting trades"
        },
        "pros": [
            "Market-neutral strategy",
            "Based on statistical principles",
            "Can work in various market conditions",
            "Generally lower risk when properly implemented"
        ],
        "cons": [
            "Requires stable statistical relationships",
            "May have extended periods of inactivity",
            "Statistical relationships can break down",
            "More complex to understand and implement"
        ]
    }
}

# Function to determine strategy category
def get_strategy_category(strategy_name):
    basic_strategies = ["Moving Average Crossover", "RSI", "Bollinger Bands"]
    if strategy_name in basic_strategies:
        return "Basic Strategies"
    return "Advanced Strategies"

# Create strategy information section
with st.expander(f"ℹ️ About {strategy_type} Strategy"):
    if strategy_type in strategy_info:
        info = strategy_info[strategy_type]
        st.markdown("### Description")
        st.write(info["description"])
        
        st.markdown("### Parameters")
        for param, desc in info["parameters"].items():
            st.markdown(f"**{param}**: {desc}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Advantages")
            for pro in info["pros"]:
                st.markdown(f"✅ {pro}")
        
        with col2:
            st.markdown("### Limitations")
            for con in info["cons"]:
                st.markdown(f"⚠️ {con}")
        
        # Add performance tips
        st.markdown("### Performance Tips")
        st.info("""
        - Consider market conditions when selecting parameters
        - Use in conjunction with other indicators for confirmation
        - Regularly review and adjust parameters based on market changes
        """)

# Strategy specific parameters
if strategy_type == "Moving Average Crossover":
    # Initialize session state for parameters if not exists
    if 'ma_params' not in st.session_state:
        st.session_state.ma_params = {'short_window': 20, 'long_window': 50}
    
    short_window = st.sidebar.slider("Short MA Window", 5, 50, st.session_state.ma_params['short_window'])
    long_window = st.sidebar.slider("Long MA Window", 20, 200, st.session_state.ma_params['long_window'])
    strategy = MovingAverageCrossover(short_window, long_window)
elif strategy_type == "RSI":
    # Initialize session state for parameters if not exists
    if 'rsi_params' not in st.session_state:
        st.session_state.rsi_params = {'window': 14, 'oversold': 30, 'overbought': 70}
    
    rsi_window = st.sidebar.slider("RSI Window", 5, 30, st.session_state.rsi_params['window'])
    oversold = st.sidebar.slider("Oversold Level", 20, 40, st.session_state.rsi_params['oversold'])
    overbought = st.sidebar.slider("Overbought Level", 60, 80, st.session_state.rsi_params['overbought'])
    strategy = RSIStrategy(rsi_window, oversold, overbought)
elif strategy_type == "Trend Following":
    # Initialize session state for parameters if not exists
    if 'trend_following_params' not in st.session_state:
        st.session_state.trend_following_params = {
            'ema_short': 20, 'ema_long': 50,
            'atr_period': 14, 'rsi_period': 14
        }
    
    ema_short = st.sidebar.slider("Short EMA", 5, 50, st.session_state.trend_following_params['ema_short'])
    ema_long = st.sidebar.slider("Long EMA", 20, 200, st.session_state.trend_following_params['ema_long'])
    atr_period = st.sidebar.slider("ATR Period", 5, 30, st.session_state.trend_following_params['atr_period'])
    rsi_period = st.sidebar.slider("RSI Period", 5, 30, st.session_state.trend_following_params['rsi_period'])
    strategy = TrendFollowingStrategy(ema_short, ema_long, atr_period, rsi_period)
elif strategy_type == "Mean Reversion":
    # Initialize session state for parameters if not exists
    if 'mean_reversion_params' not in st.session_state:
        st.session_state.mean_reversion_params = {
            'bb_window': 20, 'bb_std': 2.0,
            'rsi_window': 14, 'zscore_window': 20
        }
    
    bb_window = st.sidebar.slider("BB Window", 5, 50, st.session_state.mean_reversion_params['bb_window'])
    bb_std = st.sidebar.slider("BB Std Dev", 1.0, 3.0, st.session_state.mean_reversion_params['bb_std'])
    rsi_window = st.sidebar.slider("RSI Window", 5, 30, st.session_state.mean_reversion_params['rsi_window'])
    zscore_window = st.sidebar.slider("Z-Score Window", 5, 50, st.session_state.mean_reversion_params['zscore_window'])
    strategy = MeanReversionStrategy(bb_window, bb_std, rsi_window, zscore_window)
elif strategy_type == "Statistical Arbitrage":
    # Initialize session state for parameters if not exists
    if 'stat_arb_params' not in st.session_state:
        st.session_state.stat_arb_params = {
            'window': 60, 'entry_zscore': 2.0, 'exit_zscore': 0.0
        }
    
    window = st.sidebar.slider("Window", 20, 100, st.session_state.stat_arb_params['window'])
    entry_zscore = st.sidebar.slider("Entry Z-Score", 1.0, 3.0, st.session_state.stat_arb_params['entry_zscore'])
    exit_zscore = st.sidebar.slider("Exit Z-Score", 0.0, 1.0, st.session_state.stat_arb_params['exit_zscore'])
    strategy = StatisticalArbitrageStrategy(window, entry_zscore, exit_zscore)
else:  # Bollinger Bands
    # Initialize session state for parameters if not exists
    # Reset session state if old parameter names exist
    if 'bb_params' in st.session_state and 'window' in st.session_state.bb_params:
        del st.session_state.bb_params
    
    if 'bb_params' not in st.session_state:
        st.session_state.bb_params = {'bb_window': 20, 'num_std': 2.0}
    
    bb_window = st.sidebar.slider("Bollinger Bands Window", 5, 30, st.session_state.bb_params['bb_window'])
    num_std = st.sidebar.slider("Number of Standard Deviations", 1.0, 3.0, st.session_state.bb_params['num_std'])
    strategy = BollingerBandsStrategy(bb_window, num_std)

# Cache data fetching
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_stock_data(symbol, start_date, end_date):
    try:
        data = yf.download(symbol, start=start_date, end=end_date)
        if len(data) < 2:
            return None, "Insufficient data available for the selected date range"
        if not all(col in data.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
            return None, "Missing required price data columns"
        return data, None
    except Exception as e:
        return None, f"Error fetching data: {str(e)}"

# Fetch data
data, error = fetch_stock_data(symbol, start_date, end_date)
if error:
    st.error(error)
    st.stop()

# Add optimization settings section
st.sidebar.subheader("Optimization Settings")
show_optimization = st.sidebar.checkbox("Show Optimization Options", False)

# Initialize stop_optimization in session state if not exists
if 'stop_optimization' not in st.session_state:
    st.session_state.stop_optimization = False

# Container for optimization results
optimization_container = st.container()

if show_optimization:
    if strategy_type == "Moving Average Crossover":
        short_range = st.sidebar.slider("Short MA Range", 5, 50, (10, 30))
        long_range = st.sidebar.slider("Long MA Range", 20, 200, (40, 100))
        
        # Add step size to reduce parameter space
        step_size = st.sidebar.slider("Parameter Step Size", 1, 5, 2)
        
        # Create columns for optimization controls
        col1, col2 = st.columns(2)
        start_button = col1.button("Start Optimization")
        stop_button = col2.button("Stop Optimization")
        
        # Handle stop button
        if stop_button:
            st.session_state.stop_optimization = True
        
        if start_button:
            # Reset stop flag
            st.session_state.stop_optimization = False
            
            st.sidebar.info("Optimization in progress...")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Run optimization
            best_params = optimize_strategy(
                strategy_type,
                data,
                {
                    'short_window': range(short_range[0], short_range[1], step_size),
                    'long_window': range(long_range[0], long_range[1], step_size)
                }
            )
            
            if st.session_state.stop_optimization:
                status_text.warning("Optimization stopped by user")
            else:
                status_text.success("Optimization complete!")
            progress_bar.progress(100)
            
            # Store optimization results in session state
            st.session_state.optimization_results = best_params
            st.session_state.current_strategy = strategy_type

    elif strategy_type == "RSI":
        window_range = st.sidebar.slider("RSI Window Range", 5, 30, (10, 20))
        oversold_range = st.sidebar.slider("Oversold Range", 20, 40, (25, 35))
        overbought_range = st.sidebar.slider("Overbought Range", 60, 80, (65, 75))
        
        if st.sidebar.button("Start Optimization"):
            st.sidebar.info("Optimization in progress...")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            best_params = optimize_strategy(
                strategy_type,
                data,
                {
                    'window': range(window_range[0], window_range[1]),
                    'oversold': range(oversold_range[0], oversold_range[1]),
                    'overbought': range(overbought_range[0], overbought_range[1])
                }
            )
            
            progress_bar.progress(100)
            status_text.success("Optimization complete!")
            
            # Store optimization results in session state
            st.session_state.optimization_results = best_params
            st.session_state.current_strategy = strategy_type

    elif strategy_type == "Bollinger Bands":
        window_range = st.sidebar.slider("Window Range", 5, 30, (10, 20))
        std_range = st.sidebar.slider("Std Dev Range", 1.0, 3.0, (1.5, 2.5), 0.1)
        
        if st.sidebar.button("Start Optimization"):
            st.sidebar.info("Optimization in progress...")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            best_params = optimize_strategy(
                strategy_type,
                data,
                {
                    'bb_window': range(window_range[0], window_range[1]),
                    'num_std': np.arange(std_range[0], std_range[1], 0.1)
                }
            )
            
            progress_bar.progress(100)
            status_text.success("Optimization complete!")
            
            # Store optimization results in session state
            st.session_state.optimization_results = best_params
            st.session_state.current_strategy = strategy_type

    elif strategy_type == "Trend Following":
        ema_short_range = st.sidebar.slider("Short EMA Range", 5, 50, (10, 30))
        ema_long_range = st.sidebar.slider("Long EMA Range", 20, 200, (40, 100))
        atr_range = st.sidebar.slider("ATR Period Range", 5, 30, (10, 20))
        rsi_range = st.sidebar.slider("RSI Period Range", 5, 30, (10, 20))
        
        if st.sidebar.button("Start Optimization"):
            st.sidebar.info("Optimization in progress...")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            best_params = optimize_strategy(
                strategy_type,
                data,
                {
                    'ema_short': range(ema_short_range[0], ema_short_range[1]),
                    'ema_long': range(ema_long_range[0], ema_long_range[1]),
                    'atr_period': range(atr_range[0], atr_range[1]),
                    'rsi_period': range(rsi_range[0], rsi_range[1])
                }
            )
            
            progress_bar.progress(100)
            status_text.success("Optimization complete!")
            
            st.session_state.optimization_results = best_params
            st.session_state.current_strategy = strategy_type

    elif strategy_type == "Mean Reversion":
        bb_window_range = st.sidebar.slider("BB Window Range", 5, 50, (10, 30))
        bb_std_range = st.sidebar.slider("BB Std Range", 1.0, 3.0, (1.5, 2.5), 0.1)
        rsi_range = st.sidebar.slider("RSI Window Range", 5, 30, (10, 20))
        zscore_range = st.sidebar.slider("Z-Score Window Range", 5, 50, (10, 30))
        
        if st.sidebar.button("Start Optimization"):
            st.sidebar.info("Optimization in progress...")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            best_params = optimize_strategy(
                strategy_type,
                data,
                {
                    'bb_window': range(bb_window_range[0], bb_window_range[1]),
                    'bb_std': np.arange(bb_std_range[0], bb_std_range[1], 0.1),
                    'rsi_window': range(rsi_range[0], rsi_range[1]),
                    'zscore_window': range(zscore_range[0], zscore_range[1])
                }
            )
            
            progress_bar.progress(100)
            status_text.success("Optimization complete!")
            
            st.session_state.optimization_results = best_params
            st.session_state.current_strategy = strategy_type

    elif strategy_type == "Statistical Arbitrage":
        window_range = st.sidebar.slider("Window Range", 20, 100, (40, 80))
        entry_range = st.sidebar.slider("Entry Z-Score Range", 1.0, 3.0, (1.5, 2.5), 0.1)
        exit_range = st.sidebar.slider("Exit Z-Score Range", 0.0, 1.0, (0.2, 0.8), 0.1)
        
        if st.sidebar.button("Start Optimization"):
            st.sidebar.info("Optimization in progress...")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            best_params = optimize_strategy(
                strategy_type,
                data,
                {
                    'window': range(window_range[0], window_range[1]),
                    'entry_zscore': np.arange(entry_range[0], entry_range[1], 0.1),
                    'exit_zscore': np.arange(exit_range[0], exit_range[1], 0.1)
                }
            )
            
            progress_bar.progress(100)
            status_text.success("Optimization complete!")
            
            st.session_state.optimization_results = best_params
            st.session_state.current_strategy = strategy_type

# Show optimization results if available (outside the optimization section)
if 'optimization_results' in st.session_state:
    with optimization_container:
        st.write("Best Parameters:", st.session_state.optimization_results['parameters'])
        st.write(f"Optimized Return: {st.session_state.optimization_results['return']:.2f}%")
        
        # Ask user if they want to apply the optimized parameters
        if st.button("Apply Optimized Parameters"):
            if st.session_state.current_strategy == "Moving Average Crossover":
                st.session_state.ma_params = st.session_state.optimization_results['parameters']
            elif st.session_state.current_strategy == "RSI":
                st.session_state.rsi_params = st.session_state.optimization_results['parameters']
            elif st.session_state.current_strategy == "Bollinger Bands":
                st.session_state.bb_params = st.session_state.optimization_results['parameters']
            elif st.session_state.current_strategy == "Trend Following":
                if 'trend_following_params' not in st.session_state:
                    st.session_state.trend_following_params = {}
                st.session_state.trend_following_params = st.session_state.optimization_results['parameters']
            elif st.session_state.current_strategy == "Mean Reversion":
                if 'mean_reversion_params' not in st.session_state:
                    st.session_state.mean_reversion_params = {}
                st.session_state.mean_reversion_params = st.session_state.optimization_results['parameters']
            else:  # Statistical Arbitrage
                if 'stat_arb_params' not in st.session_state:
                    st.session_state.stat_arb_params = {}
                st.session_state.stat_arb_params = st.session_state.optimization_results['parameters']
            
            # Clear optimization results after applying
            del st.session_state.optimization_results
            del st.session_state.current_strategy
            st.rerun()

# Add reset button to restore default parameters
if st.sidebar.button("Reset Parameters"):
    if strategy_type == "Moving Average Crossover":
        st.session_state.ma_params = {'short_window': 20, 'long_window': 50}
    elif strategy_type == "RSI":
        st.session_state.rsi_params = {'window': 14, 'oversold': 30, 'overbought': 70}
    elif strategy_type == "Bollinger Bands":
        st.session_state.bb_params = {'bb_window': 20, 'num_std': 2.0}
    elif strategy_type == "Trend Following":
        st.session_state.trend_following_params = {
            'ema_short': 20, 'ema_long': 50,
            'atr_period': 14, 'rsi_period': 14
        }
    elif strategy_type == "Mean Reversion":
        st.session_state.mean_reversion_params = {
            'bb_window': 20, 'bb_std': 2.0,
            'rsi_window': 14, 'zscore_window': 20
        }
    else:  # Statistical Arbitrage
        st.session_state.stat_arb_params = {
            'window': 60, 'entry_zscore': 2.0, 'exit_zscore': 0.0
        }
    st.rerun()

# Get current parameters based on strategy type
def get_current_params():
    if strategy_type == "Moving Average Crossover":
        return {'short_window': short_window, 'long_window': long_window}
    elif strategy_type == "RSI":
        return {'window': rsi_window, 'oversold': oversold, 'overbought': overbought}
    elif strategy_type == "Bollinger Bands":
        return {'bb_window': bb_window, 'num_std': num_std}
    elif strategy_type == "Trend Following":
        return {'ema_short': ema_short, 'ema_long': ema_long, 
                'atr_period': atr_period, 'rsi_period': rsi_period}
    elif strategy_type == "Mean Reversion":
        return {'bb_window': bb_window, 'bb_std': bb_std, 
                'rsi_window': rsi_window, 'zscore_window': zscore_window}
    else:  # Statistical Arbitrage
        return {'window': window, 'entry_zscore': entry_zscore, 'exit_zscore': exit_zscore}

# Cache backtest results
@st.cache_data(ttl=3600)
def run_backtest(strategy_type, params, data):
    strategy = create_strategy(strategy_type, params)
    backtester = Backtester(strategy)
    results = backtester.run(data)
    metrics = calculate_metrics(results)
    signals = strategy.generate_signals(data)
    return results, metrics, signals

# Helper function to create strategy instance
def create_strategy(strategy_type, params):
    if strategy_type == "Moving Average Crossover":
        return MovingAverageCrossover(params['short_window'], params['long_window'])
    elif strategy_type == "RSI":
        return RSIStrategy(params['window'], params['oversold'], params['overbought'])
    elif strategy_type == "Bollinger Bands":
        return BollingerBandsStrategy(params['bb_window'], params['num_std'])
    elif strategy_type == "Trend Following":
        return TrendFollowingStrategy(params['ema_short'], params['ema_long'], 
                                    params['atr_period'], params['rsi_period'])
    elif strategy_type == "Mean Reversion":
        return MeanReversionStrategy(params['bb_window'], params['bb_std'], 
                                   params['rsi_window'], params['zscore_window'])
    else:  # Statistical Arbitrage
        return StatisticalArbitrageStrategy(params['window'], params['entry_zscore'], 
                                          params['exit_zscore'])

# Run backtest with caching
current_params = get_current_params()
results, metrics, signals = run_backtest(strategy_type, current_params, data)

# Add checks for NaN values before displaying metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Return", f"{metrics['total_return']:.2f}%" if pd.notnull(metrics['total_return']) else "N/A")
col2.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}" if pd.notnull(metrics['sharpe_ratio']) else "N/A")
col3.metric("Max Drawdown", f"{metrics['max_drawdown']:.2f}%" if pd.notnull(metrics['max_drawdown']) else "N/A")
col4.metric("Win Rate", f"{metrics['win_rate']:.2f}%" if pd.notnull(metrics['win_rate']) else "N/A")

# Cache plot creation
@st.cache_data(ttl=3600)
def create_signal_plot(data, results):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'],
                            mode='lines', name='Price'))
    
    # Add buy/sell signals
    buy_signals = results[results['Signal'] == 1]
    sell_signals = results[results['Signal'] == -1]
    
    if len(buy_signals) > 0:
        fig.add_trace(go.Scatter(x=buy_signals.index,
                                y=data.loc[buy_signals.index, 'Close'],
                                mode='markers', name='Buy Signal',
                        marker=dict(color='green', size=10)))

    if len(sell_signals) > 0:
        fig.add_trace(go.Scatter(x=sell_signals.index,
                                y=data.loc[sell_signals.index, 'Close'],
                                mode='markers', name='Sell Signal',
                        marker=dict(color='red', size=10)))

    fig.update_layout(title='Trading Signals',
                     xaxis_title='Date',
                     yaxis_title='Price',
                     template='plotly_white')
    return fig

# Create and display plots
signal_fig = create_signal_plot(data, results)
st.plotly_chart(signal_fig, use_container_width=True)

# Plot equity curve
fig_equity = go.Figure()
fig_equity.add_trace(go.Scatter(x=results.index,
                                y=results['Equity Curve'],
                                mode='lines',
                                name='Equity Curve'))

fig_equity.update_layout(title='Equity Curve',
                         xaxis_title='Date',
                         yaxis_title='Portfolio Value',
                         template='plotly_white')

st.plotly_chart(fig_equity, use_container_width=True)

# Add strategy-specific visualizations
if strategy_type == "Moving Average Crossover":
    # Plot moving averages
    fig_ma = go.Figure()
    fig_ma.add_trace(go.Scatter(x=data.index, y=data['Close'],
                               mode='lines',
                               name='Price'))
    fig_ma.add_trace(go.Scatter(x=signals.index, y=signals['Short MA'],
                               mode='lines',
                               name=f'{short_window}-day MA',
                               line=dict(dash='dash')))
    fig_ma.add_trace(go.Scatter(x=signals.index, y=signals['Long MA'],
                               mode='lines',
                               name=f'{long_window}-day MA',
                               line=dict(dash='dash')))
    fig_ma.update_layout(title='Moving Averages',
                         yaxis_title='Price',
                         template='plotly_white')
    st.plotly_chart(fig_ma, use_container_width=True)

elif strategy_type == "RSI":
    # Plot RSI indicator
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=signals.index, y=signals['RSI'],
                                mode='lines',
                                name='RSI'))
    fig_rsi.add_hline(y=oversold, line_dash="dash", line_color="green")
    fig_rsi.add_hline(y=overbought, line_dash="dash", line_color="red")
    fig_rsi.update_layout(title='RSI Indicator',
                         yaxis_title='RSI Value',
                         template='plotly_white')
    st.plotly_chart(fig_rsi, use_container_width=True)

elif strategy_type == "Bollinger Bands":
    # Plot price with Bollinger Bands
    fig_bb = go.Figure()
    fig_bb.add_trace(go.Scatter(x=data.index, y=data['Close'],
                               mode='lines',
                               name='Price'))
    fig_bb.add_trace(go.Scatter(x=signals.index, y=signals['Upper'],
                               mode='lines',
                               name='Upper Band',
                               line=dict(dash='dash')))
    fig_bb.add_trace(go.Scatter(x=signals.index, y=signals['Lower'],
                               mode='lines',
                               name='Lower Band',
                               line=dict(dash='dash')))
    st.plotly_chart(fig_bb, use_container_width=True) 

# Add strategy comparison section
if st.sidebar.checkbox("Compare Strategies"):
    strategies_to_compare = st.sidebar.multiselect(
        "Select strategies to compare",
        {
            "Basic Strategies": ["Moving Average Crossover", "RSI", "Bollinger Bands"],
            "Advanced Strategies": ["Trend Following", "Mean Reversion", "Statistical Arbitrage"]
        }
    )
    
    if strategies_to_compare:
        comparison_results = {}
        for strat_type in strategies_to_compare:
            if strat_type == "Moving Average Crossover":
                strategy = MovingAverageCrossover(20, 50)
            elif strat_type == "RSI":
                strategy = RSIStrategy(14, 30, 70)
            elif strat_type == "Bollinger Bands":
                strategy = BollingerBandsStrategy(20, 2.0)
            elif strat_type == "Trend Following":
                strategy = TrendFollowingStrategy(20, 50, 14, 14)
            elif strat_type == "Mean Reversion":
                strategy = MeanReversionStrategy(20, 2.0, 14, 20)
            else:  # Statistical Arbitrage
                strategy = StatisticalArbitrageStrategy(60, 2.0, 0.0)
            
            backtester = Backtester(strategy)
            results = backtester.run(data)
            metrics = calculate_metrics(results)
            comparison_results[strat_type] = metrics
        
        # Display comparison table
        comparison_df = pd.DataFrame(comparison_results).round(2)
        st.write("Strategy Comparison")
        st.dataframe(comparison_df)
        
        # Add strategy rankings
        rankings = pd.DataFrame()
        rankings['Total Return'] = comparison_df.loc['total_return'].rank(ascending=False)
        rankings['Sharpe Ratio'] = comparison_df.loc['sharpe_ratio'].rank(ascending=False)
        rankings['Max Drawdown'] = comparison_df.loc['max_drawdown'].rank()
        rankings['Overall Rank'] = rankings.mean(axis=1)
        
        st.write("Strategy Rankings (1 is best)")
        st.dataframe(rankings.round(2))

# Add export functionality
if st.button("Export Results"):
    # Create Excel writer
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        # Write trading signals
        results.to_excel(writer, sheet_name='Trading Signals')
        
        # Write metrics
        pd.DataFrame([metrics]).to_excel(writer, sheet_name='Performance Metrics')
        
        # Write parameters
        if strategy_type == "Moving Average Crossover":
            params = {'short_window': short_window, 'long_window': long_window}
        elif strategy_type == "RSI":
            params = {'window': rsi_window, 'oversold': oversold, 'overbought': overbought}
        elif strategy_type == "Bollinger Bands":
            params = {'bb_window': bb_window, 'num_std': num_std}
        elif strategy_type == "Trend Following":
            params = {
                'ema_short': ema_short, 'ema_long': ema_long,
                'atr_period': atr_period, 'rsi_period': rsi_period
            }
        elif strategy_type == "Mean Reversion":
            params = {
                'bb_window': bb_window, 'bb_std': bb_std,
                'rsi_window': rsi_window, 'zscore_window': zscore_window
            }
        else:  # Statistical Arbitrage
            params = {
                'window': window, 'entry_zscore': entry_zscore,
                'exit_zscore': exit_zscore
            }
        pd.DataFrame([params]).to_excel(writer, sheet_name='Strategy Parameters')
    
    # Download button
    st.download_button(
        label="Download Results",
        data=buffer,
        file_name=f"trading_results_{symbol}_{strategy_type}.xlsx",
        mime="application/vnd.ms-excel"
    ) 