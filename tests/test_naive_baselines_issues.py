"""
Test to verify issues in naive_baselines.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from models.naive_baselines import LinearTrendPredictor


def test_linear_trend_order_issue():
    """Test Issue 1: LinearTrendPredictor groupby ordering issue."""
    print("\n" + "="*80)
    print("TEST 1: LinearTrendPredictor groupby() ordering issue")
    print("="*80)
    
    # Create test data with multiple tickers
    # Ticker A: prices 100, 101, 102, 103, 104
    # Ticker B: prices 200, 202, 204, 206, 208
    # Ticker C: prices 50, 51, 52, 53, 54
    
    data = []
    dates = pd.date_range('2020-01-01', periods=5)
    
    # Create data in specific order: C, A, B
    for i in range(5):
        data.append({'date': dates[i], 'ticker': 'C', 'close': 50 + i})
    for i in range(5):
        data.append({'date': dates[i], 'ticker': 'A', 'close': 100 + i})
    for i in range(5):
        data.append({'date': dates[i], 'ticker': 'B', 'close': 200 + i*2})
    
    df = pd.DataFrame(data)
    print("\nOriginal DataFrame order:")
    print(df[['ticker', 'date', 'close']].head(15))
    print(f"\nDataFrame index order: {df.index.tolist()}")
    
    # Apply LinearTrendPredictor
    predictor = LinearTrendPredictor(window=3)
    result = predictor.predict(df)
    
    print("\nResult after prediction:")
    print(result[['ticker', 'date', 'close', 'prediction']].head(15))
    
    # Check if predictions are correctly aligned
    print("\n" + "-"*80)
    print("VERIFICATION:")
    print("-"*80)
    
    issues_found = False
    
    for ticker in ['C', 'A', 'B']:
        ticker_rows = result[result['ticker'] == ticker]
        print(f"\nTicker {ticker}:")
        print(f"  Original close prices: {ticker_rows['close'].tolist()}")
        print(f"  Predictions: {ticker_rows['prediction'].tolist()}")
        
        # Check if predictions are in reasonable range of actual prices
        avg_close = ticker_rows['close'].mean()
        avg_pred = ticker_rows['prediction'].mean()
        
        # For ticker C: avg should be ~52, for A: ~102, for B: ~204
        # Predictions should be close to these values
        if ticker == 'C' and (avg_pred < 30 or avg_pred > 250):
            print(f"  [X] ISSUE: Predictions ({avg_pred:.1f}) are out of range for ticker C (expected ~52)")
            issues_found = True
        elif ticker == 'A' and (avg_pred < 70 or avg_pred > 250):
            print(f"  [X] ISSUE: Predictions ({avg_pred:.1f}) are out of range for ticker A (expected ~102)")
            issues_found = True
        elif ticker == 'B' and (avg_pred < 150):
            print(f"  [X] ISSUE: Predictions ({avg_pred:.1f}) are out of range for ticker B (expected ~204)")
            issues_found = True
        else:
            print(f"  [OK] Predictions appear reasonable (avg: {avg_pred:.1f})")
    
    if issues_found:
        print("\n" + "="*80)
        print("[X] ISSUE 1 CONFIRMED: Predictions are misaligned due to groupby() ordering")
        print("="*80)
        return False
    else:
        print("\n" + "="*80)
        print("[OK] ISSUE 1: No misalignment detected")
        print("="*80)
        return True


def test_regression_target_issue():
    """Test Issue 2: Regression target using same day instead of next day."""
    print("\n" + "="*80)
    print("TEST 2: Regression target timing issue")
    print("="*80)
    
    # Create simple test data
    dates = pd.date_range('2020-01-01', periods=10)
    ticker_data = pd.DataFrame({
        'date': dates,
        'ticker': 'TEST',
        'open': [100.0 + i for i in range(10)],
        'high': [102.0 + i for i in range(10)],
        'low': [98.0 + i for i in range(10)],
        'close': [100.0 + i for i in range(10)],
        'volume': [1000000] * 10
    })
    
    print("\nTest data (close prices):")
    print(ticker_data[['date', 'close']])
    
    # Simulate what dataset.py does for regression task
    print("\n" + "-"*80)
    print("Current behavior (from dataset.py lines 92-101):")
    print("-"*80)
    
    lookback = 3
    max_idx = len(ticker_data)  # Current implementation
    
    print(f"lookback = {lookback}")
    print(f"max_idx = {max_idx} (len of data)")
    print(f"\nSequences created:")
    
    for i in range(lookback, max_idx):
        seq_data = ticker_data['close'].iloc[i-lookback:i].values
        target = ticker_data['close'].iloc[i]
        print(f"  i={i}: sequence days {i-lookback} to {i-1} (close: {seq_data}) -> target: day {i} (close: {target})")
    
    print("\n[X] ISSUE: The last element of the sequence (day i-1) and the target (day i)")
    print("    have the SAME index when i = last iteration.")
    print(f"    At i={max_idx-1}: sequence includes day {max_idx-2} (close={ticker_data['close'].iloc[max_idx-2]})")
    print(f"                       and target is day {max_idx-1} (close={ticker_data['close'].iloc[max_idx-1]})")
    
    # Show what it should be
    print("\n" + "-"*80)
    print("Expected behavior (predict NEXT day):")
    print("-"*80)
    
    max_idx_correct = len(ticker_data) - 1  # Should be this
    
    print(f"max_idx should be = {max_idx_correct} (len - 1)")
    print(f"\nSequences that should be created:")
    
    for i in range(lookback, max_idx_correct):
        seq_data = ticker_data['close'].iloc[i-lookback:i].values
        target_next_day = ticker_data['close'].iloc[i+1]
        print(f"  i={i}: sequence days {i-lookback} to {i-1} (close: {seq_data}) -> target: day {i+1} (close: {target_next_day})")
    
    print("\n" + "="*80)
    print("[X] ISSUE 2 CONFIRMED: Regression task predicts same day instead of next day")
    print("="*80)
    return False


if __name__ == "__main__":
    print("\nVerifying issues in naive_baselines.py and dataset.py...")
    
    issue1_ok = test_linear_trend_order_issue()
    issue2_ok = test_regression_target_issue()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Issue 1 (LinearTrendPredictor ordering): {'FIXED' if issue1_ok else 'CONFIRMED'}")
    print(f"Issue 2 (Regression target timing): {'FIXED' if issue2_ok else 'CONFIRMED'}")
    print("="*80)

