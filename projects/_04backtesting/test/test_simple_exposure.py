"""
Simple test to understand vectorbt exposure calculation
"""

import pandas as pd
import numpy as np
import vectorbt as vbt

# Simple test data
dates = pd.date_range('2020-01-01', periods=50, freq='D')
stocks = ['A', 'B', 'C', 'D', 'E']

# Simple price data
np.random.seed(42)
prices = pd.DataFrame({
    'A': [100, 101, 102, 103, 104] * 10,
    'B': [50, 51, 52, 53, 54] * 10,
    'C': [75, 76, 77, 78, 79] * 10,
    'D': [120, 121, 122, 123, 124] * 10,
    'E': [90, 91, 92, 93, 94] * 10,
}, index=dates)

print(f"Price data shape: {prices.shape}")
print(f"First few prices:\n{prices.head()}")

# Buy-and-hold all 5 stocks with equal weight (20% each)
entry_signals = pd.DataFrame(True, index=dates, columns=stocks)
entry_signals.iloc[1:] = False  # Only buy on first day

# Test different size approaches
print("\n=== Test 1: Fixed 20% per stock ===")
size_matrix_1 = pd.DataFrame(0.2, index=dates, columns=stocks) * entry_signals
portfolio_1 = vbt.Portfolio.from_signals(
    close=prices,
    entries=entry_signals,
    exits=False,
    size=size_matrix_1,
    size_type='percent',
    init_cash=100000,
    fees=0.001,
    freq='D'
)
stats_1 = portfolio_1.stats()
print(f"Max Gross Exposure: {stats_1.get('Max Gross Exposure [%]', 'N/A')}")

print("\n=== Test 2: Single stock 100% ===")
entry_single = pd.DataFrame(False, index=dates, columns=stocks)
entry_single.iloc[0, 0] = True  # Only buy stock A
size_single = pd.DataFrame(0.0, index=dates, columns=stocks)
size_single.iloc[0, 0] = 1.0  # 100% in stock A
portfolio_2 = vbt.Portfolio.from_signals(
    close=prices,
    entries=entry_single,
    exits=False,
    size=size_single,
    size_type='percent',
    init_cash=100000,
    fees=0.001,
    freq='D'
)
stats_2 = portfolio_2.stats()
print(f"Max Gross Exposure: {stats_2.get('Max Gross Exposure [%]', 'N/A')}")

print("\n=== Test 3: Debug portfolio values ===")
# Check actual portfolio value over time
portfolio_value = portfolio_1.value()
print(f"Initial portfolio value: ${portfolio_value.iloc[0]:,.2f}")
print(f"Final portfolio value: ${portfolio_value.iloc[-1]:,.2f}")

# Check cash position
cash = portfolio_1.cash()
print(f"Initial cash: ${cash.iloc[0]:,.2f}")
print(f"Final cash: ${cash.iloc[-1]:,.2f}")

# Check positions
positions = portfolio_1.positions.records_readable
print(f"Number of positions opened: {len(positions)}")
if len(positions) > 0:
    print(f"Position details:\n{positions[['Size', 'Entry Price', 'Entry Val']].head()}")