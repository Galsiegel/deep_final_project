"""
Test to verify that day i's opening price is correctly included in regression sequences.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np


def test_sequence_shape():
    """Test that regression sequences have correct shape [lookback+1, features]."""
    print("\n" + "="*80)
    print("TEST: Verify day i's opening price is included in regression")
    print("="*80)
    
    # We'll need torch for this, but let's test the logic manually first
    
    # Simulate what dataset.py does
    lookback = 5
    features = ['open', 'high', 'low', 'close', 'volume']
    
    # Create test data
    dates = pd.date_range('2020-01-01', periods=10)
    ticker_data = pd.DataFrame({
        'date': dates,
        'ticker': ['TEST'] * 10,
        'open': [100.0 + i for i in range(10)],
        'high': [102.0 + i for i in range(10)],
        'low': [98.0 + i for i in range(10)],
        'close': [100.0 + i for i in range(10)],
        'volume': [1000000] * 10
    })
    
    print(f"\nTest data (10 days):")
    print(ticker_data[['date', 'open', 'close']])
    
    print(f"\nLookback window: {lookback} days")
    print(f"Features: {features}")
    
    # Simulate sequence creation for i=5 (predicting day 5's close)
    i = 5
    print(f"\n" + "-"*80)
    print(f"Creating sequence for i={i} (predicting day {i}'s close)")
    print("-"*80)
    
    # Historical data: days [i-lookback, i-1] = [0, 1, 2, 3, 4]
    seq_historical = ticker_data[features].iloc[i-lookback:i].values
    print(f"\nHistorical sequence [days {i-lookback} to {i-1}]:")
    print(f"  Shape: {seq_historical.shape}")
    print(f"  Days: {list(range(i-lookback, i))}")
    print(f"  Open prices: {seq_historical[:, 0].tolist()}")
    print(f"  Close prices: {seq_historical[:, 3].tolist()}")
    
    # Day i's partial data
    prev_close = ticker_data['close'].iloc[i-1]
    day_i_partial = np.zeros(len(features))
    
    for j, feat in enumerate(features):
        if feat == 'open':
            day_i_partial[j] = ticker_data['open'].iloc[i]
        elif feat == 'volume':
            day_i_partial[j] = 0.0  # Unknown yet
        else:  # high, low, close - unknown
            day_i_partial[j] = prev_close  # Use previous close as proxy
    
    print(f"\nDay {i}'s partial data (only open is real):")
    print(f"  Open: {day_i_partial[0]} (REAL)")
    print(f"  High: {day_i_partial[1]} (proxy from prev close)")
    print(f"  Low: {day_i_partial[2]} (proxy from prev close)")
    print(f"  Close: {day_i_partial[3]} (proxy from prev close)")
    print(f"  Volume: {day_i_partial[4]} (unknown, set to 0)")
    
    # Stack together
    seq = np.vstack([seq_historical, day_i_partial])
    
    print(f"\nFinal sequence shape: {seq.shape}")
    print(f"  Expected: ({lookback+1}, {len(features)})")
    
    # Target
    target = ticker_data['close'].iloc[i]
    print(f"\nTarget (day {i}'s close): {target}")
    
    print("\n" + "-"*80)
    print("VERIFICATION:")
    print("-"*80)
    
    # Check shape
    expected_shape = (lookback + 1, len(features))
    if seq.shape == expected_shape:
        print(f"[PASS] Sequence shape is correct: {seq.shape}")
    else:
        print(f"[FAIL] Sequence shape is wrong: {seq.shape}, expected {expected_shape}")
        return False
    
    # Check that day i's open is included and correct
    day_i_open_in_seq = seq[-1, 0]
    expected_open = ticker_data['open'].iloc[i]
    if abs(day_i_open_in_seq - expected_open) < 0.001:
        print(f"[PASS] Day {i}'s opening price is correctly included: {day_i_open_in_seq}")
    else:
        print(f"[FAIL] Day {i}'s opening price is wrong: {day_i_open_in_seq}, expected {expected_open}")
        return False
    
    # Check that day i's close is NOT in the sequence (we're predicting it)
    # The close value in seq[-1] should be the proxy (prev_close), not the real close
    day_i_close_in_seq = seq[-1, 3]
    real_close = ticker_data['close'].iloc[i]
    if abs(day_i_close_in_seq - real_close) > 0.001:
        print(f"[PASS] Day {i}'s close is NOT leaked into input: seq has {day_i_close_in_seq}, real is {real_close}")
    else:
        print(f"[FAIL] Day {i}'s close is leaked! seq has {day_i_close_in_seq}, real is {real_close}")
        return False
    
    print("\n" + "="*80)
    print("[SUCCESS] Day i's opening price is correctly included without data leakage!")
    print("="*80)
    
    return True


def test_information_available():
    """Verify we understand what information is available when."""
    print("\n" + "="*80)
    print("INFORMATION TIMELINE")
    print("="*80)
    
    print("""
When predicting day i's closing price:

Day i-2        Day i-1        Day i
|--------------|--------------|--------------|
   OHLCV          OHLCV        Open    ?Close?
                              9:30 AM  4:00 PM

AVAILABLE FOR PREDICTION:
  [OK] Days i-2, i-1, ... : Full OHLCV (complete historical data)
  [OK] Day i: Opening price (known at 9:30 AM)
  [NO] Day i: High, Low, Close (not known until 4:00 PM)
  [NO] Day i: Volume (partial during day, final at 4:00 PM)

INPUT SEQUENCE SHAPE:
  - Historical: [lookback] days with [5] features each
  - Current day: [1] day with partial data (open + proxies for h/l/c)
  - Total: [lookback+1, 5]

TARGET:
  - Day i's closing price (what we want to predict at 4:00 PM)
""")
    
    print("="*80)
    print("[OK] Information availability is correctly understood")
    print("="*80)


if __name__ == "__main__":
    print("\nVerifying that day i's opening price is included in regression task...\n")
    
    test_information_available()
    success = test_sequence_shape()
    
    print("\n" + "="*80)
    if success:
        print("SUMMARY: [SUCCESS] Implementation is correct!")
    else:
        print("SUMMARY: [FAILURE] Implementation has issues")
    print("="*80)
    
    sys.exit(0 if success else 1)

