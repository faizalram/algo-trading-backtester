import numpy as np
from itertools import product
from backtester import Backtester
import pandas as pd
from metrics import calculate_metrics
from scipy.optimize import minimize
from multiprocessing import Pool
from functools import partial
from strategies import (MovingAverageCrossover, RSIStrategy, BollingerBandsStrategy,
                        TrendFollowingStrategy, MeanReversionStrategy, StatisticalArbitrageStrategy)
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from sklearn.model_selection import ParameterGrid
import streamlit as st

class StrategyOptimizer:
    def __init__(self, strategy_type, data):
        self.strategy_type = strategy_type
        self.data = data
        if 'stop_optimization' not in st.session_state:
            st.session_state.stop_optimization = False
    
    def evaluate_params(self, params):
        """Evaluate a single set of parameters."""
        strategy = create_strategy(self.strategy_type, params)
        backtester = Backtester(strategy)
        results = backtester.run(self.data)
        return (results['Equity Curve'].iloc[-1] / results['Equity Curve'].iloc[0] - 1) * 100
    
    def quick_evaluate(self, params):
        """Quick evaluation using a smaller data sample"""
        sample_data = self.data.tail(252)  # Use only last year of data
        strategy = create_strategy(self.strategy_type, params)
        backtester = Backtester(strategy)
        results = backtester.run(sample_data)
        return (results['Equity Curve'].iloc[-1] / results['Equity Curve'].iloc[0] - 1) * 100
    
    def optimize(self, param_ranges):
        """Main optimization method"""
        if self.strategy_type in ["Moving Average Crossover", "RSI", "Bollinger Bands"]:
            return self.grid_search_optimization(param_ranges)
        else:
            return self.advanced_optimization(param_ranges)
    
    def grid_search_optimization(self, param_ranges):
        """Grid search optimization for simple strategies."""
        # Convert ranges to lists for ParameterGrid
        param_grid = {name: list(range_) for name, range_ in param_ranges.items()}
        all_params = list(ParameterGrid(param_grid))
        
        # Progress tracking
        total_params = len(all_params)
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        best_return = float('-inf')
        best_params = None
        
        # Reset stop flag at start of optimization
        st.session_state.stop_optimization = False
        
        for i, params in enumerate(all_params):
            if st.session_state.stop_optimization:
                break
                
            # Quick evaluation
            current_return = self.quick_evaluate(params)
            
            if current_return > best_return:
                best_return = current_return
                best_params = params
            
            # Update progress
            progress = (i + 1) / total_params
            progress_bar.progress(progress)
            status_text.text(f"Testing parameter set {i+1}/{total_params}")
        
        progress_bar.empty()
        status_text.empty()
        
        if best_params is None:
            best_params = {name: list(range_)[len(range_)//2] 
                         for name, range_ in param_ranges.items()}
            best_return = self.quick_evaluate(best_params)
        
        return {
            'parameters': best_params,
            'return': best_return
        }
    
    def advanced_optimization(self, param_ranges):
        """Advanced optimization using scipy.optimize."""
        param_names = list(param_ranges.keys())
        
        # Generate random initial points
        n_random = 5
        best_return = float('-inf')
        best_params = None
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Reset stop flag at start of optimization
        st.session_state.stop_optimization = False
        
        for i in range(n_random):
            if st.session_state.stop_optimization:
                break
                
            # Generate random parameters
            params = {}
            for name, range_ in param_ranges.items():
                if isinstance(range_, range):
                    params[name] = np.random.randint(range_.start, range_.stop)
                else:
                    params[name] = np.random.uniform(range_[0], range_[-1])
            
            # Evaluate
            current_return = self.evaluate_params(params)
            if current_return > best_return:
                best_return = current_return
                best_params = params
            
            # Update progress
            progress_bar.progress((i + 1) / n_random)
            status_text.text(f"Testing random parameter set {i+1}/{n_random}")
        
        progress_bar.empty()
        status_text.empty()
        
        return {
            'parameters': best_params,
            'return': best_return
        }

def optimize_strategy(strategy_type, data, param_ranges):
    optimizer = StrategyOptimizer(strategy_type, data)
    return optimizer.optimize(param_ranges)

def create_strategy(strategy_type, params):
    """
    Create a strategy instance based on strategy type and parameters.
    
    Args:
        strategy_type (str): Type of strategy to create
        params (dict): Strategy parameters
        
    Returns:
        Strategy: Instance of the specified strategy
    """
    if strategy_type == "Moving Average Crossover":
        return MovingAverageCrossover(
            short_window=params['short_window'],
            long_window=params['long_window']
        )
    elif strategy_type == "RSI":
        return RSIStrategy(
            window=params['window'],
            oversold=params['oversold'],
            overbought=params['overbought']
        )
    elif strategy_type == "Bollinger Bands":
        return BollingerBandsStrategy(
            window=params['window'],
            num_std=params['num_std']
        )
    elif strategy_type == "Trend Following":
        return TrendFollowingStrategy(
            ema_short=params['ema_short'],
            ema_long=params['ema_long'],
            atr_period=params['atr_period'],
            rsi_period=params['rsi_period']
        )
    elif strategy_type == "Mean Reversion":
        return MeanReversionStrategy(
            bb_window=params['bb_window'],
            bb_std=params['bb_std'],
            rsi_window=params['rsi_window'],
            zscore_window=params['zscore_window']
        )
    else:  # Statistical Arbitrage
        return StatisticalArbitrageStrategy(
            window=params['window'],
            entry_zscore=params['entry_zscore'],
            exit_zscore=params['exit_zscore']
        )

def walk_forward_analysis(strategy_type, data, param_ranges, window_size=252, step_size=126):
    results = []
    
    for i in range(0, len(data)-window_size, step_size):
        train_data = data.iloc[i:i+window_size]
        test_data = data.iloc[i+window_size:i+window_size+step_size]
        
        # Optimize on training data
        best_params = optimize_strategy(strategy_type, train_data, param_ranges)
        
        # Test on out-of-sample data
        strategy = create_strategy(strategy_type, best_params['parameters'])
        backtester = Backtester(strategy)
        test_results = backtester.run(test_data)
        
        results.append({
            'period': f"{test_data.index[0].date()} - {test_data.index[-1].date()}",
            'parameters': best_params['parameters'],
            'train_return': best_params['return'],
            'test_return': calculate_metrics(test_results)['total_return']
        })
    
    return pd.DataFrame(results) 