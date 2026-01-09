"""
Filter FNSPID dataset to only include S&P 500 companies.

This script:
1. Loads S&P 500 ticker list
2. Filters the Hugging Face dataset to only S&P 500 companies
3. Saves filtered data to CSV or processes further

GOOGLE COLAB USAGE:
-------------------
This script works in Google Colab and saves progress in batches!

Basic usage (saves to /content/ - temporary):
    python filter_sp500.py

With Google Drive (persists after session ends):
    python filter_sp500.py --use-drive

Custom output location:
    python filter_sp500.py --output /content/drive/MyDrive/my_data.csv

PROGRESS SAVING & CHECKPOINTING:
- Saves in batches (default: 50,000 rows, configurable with --batch-size)
- Each batch is appended to the CSV file
- Automatically saves checkpoint after each batch (.checkpoint file)
- If Colab disconnects, re-run the script - it will resume from checkpoint
- Use --use-drive to save to Google Drive for permanent storage
- Use --reset-checkpoint to start from beginning (deletes checkpoint)

DATE FILTERING:
- Default: Filters to 2015-01-01 to 2022-12-31
- Use --start-date and --end-date to customize (YYYY-MM-DD format)
- Example: --start-date 2020-01-01 --end-date 2021-12-31

MANUAL START ROW:
- Use --start-from-row N to start processing from a specific source row index
- Overrides checkpoint resume if specified
- Example: --start-from-row 1000000 (starts from row 1,000,000)
"""

from datasets import load_dataset
from collections import Counter
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import json
import itertools

# S&P 500 ticker list (as of common snapshot - you may want to update this)
# Source: Standard S&P 500 list
SP500_TICKERS = {
    'A', 'AA', 'AAL', 'AAP', 'AAPL', 'ABBV', 'ABC', 'ABMD', 'ABT', 'ACGL',
    'ACN', 'ADBE', 'ADI', 'ADM', 'ADP', 'ADSK', 'AEE', 'AEP', 'AES', 'AFL',
    'A', 'AGCO', 'AIG', 'AIZ', 'AJG', 'AKAM', 'ALB', 'ALGN', 'ALK', 'ALL',
    'ALLE', 'AMAT', 'AMCR', 'AMD', 'AME', 'AMGN', 'AMP', 'AMT', 'AMZN', 'ANET',
    'ANSS', 'AON', 'AOS', 'APA', 'APD', 'APH', 'APTV', 'ARE', 'ATO', 'ATVI',
    'AVB', 'AVGO', 'AVY', 'AWK', 'AXP', 'AZO', 'BA', 'BAC', 'BALL', 'BAX',
    'BBWI', 'BBY', 'BEN', 'BF.B', 'BG', 'BIIB', 'BIO', 'BK', 'BKNG', 'BKR',
    'BLK', 'BLL', 'BMY', 'BR', 'BRK.B', 'BSX', 'BWA', 'BXP', 'C', 'CAG',
    'CAH', 'CARR', 'CAT', 'CB', 'CBOE', 'CBRE', 'CCI', 'CCL', 'CDAY', 'CDNS',
    'CDW', 'CE', 'CF', 'CHD', 'CHRW', 'CHTR', 'CI', 'CINF', 'CL', 'CLX',
    'CME', 'CMG', 'CMI', 'CMS', 'CNC', 'CNP', 'COF', 'COO', 'COP', 'COST',
    'CPB', 'CPRT', 'CPT', 'CRL', 'CRM', 'CRWD', 'CSCO', 'CSGP', 'CSX', 'CTAS',
    'CTLT', 'CTSH', 'CTVA', 'CVS', 'CVX', 'CZR', 'D', 'DAL', 'DD', 'DE',
    'DFS', 'DG', 'DGX', 'DHI', 'DHR', 'DIS', 'DISCA', 'DISCK', 'DISH', 'DLR',
    'DLTR', 'DOV', 'DPZ', 'DRE', 'DRI', 'DTE', 'DUK', 'DVA', 'DVN', 'DXCM',
    'EA', 'EBAY', 'ECL', 'ED', 'EFX', 'EIX', 'EL', 'EMN', 'EMR', 'ENPH',
    'ENTG', 'EOG', 'EPAM', 'EQIX', 'EQR', 'ESS', 'ETN', 'ETR', 'ETSY', 'EVRG',
    'EW', 'EXC', 'EXPD', 'EXPE', 'EXR', 'F', 'FANG', 'FAST', 'FBHS', 'FCX',
    'FDS', 'FE', 'FFIV', 'FIS', 'FISV', 'FITB', 'FLT', 'FMC', 'FOX', 'FOXA',
    'FRC', 'FRT', 'FTNT', 'FTV', 'GD', 'GE', 'GILD', 'GIS', 'GL', 'GLW',
    'GM', 'GNRC', 'GOOG', 'GOOGL', 'GPC', 'GPN', 'GRMN', 'GS', 'GWW', 'HAL',
    'HAS', 'HBAN', 'HCA', 'HD', 'HES', 'HIG', 'HII', 'HLT', 'HOLX', 'HON',
    'HPE', 'HPQ', 'HRL', 'HSIC', 'HST', 'HSY', 'HUM', 'HWM', 'IBM', 'ICE',
    'IDXX', 'IEX', 'IFF', 'ILMN', 'INCY', 'INFO', 'INTC', 'INTU', 'INVH', 'IP',
    'IPG', 'IQV', 'IR', 'IRM', 'ISRG', 'IT', 'ITW', 'IVZ', 'J', 'JBHT',
    'JCI', 'JKHY', 'JNJ', 'JNPR', 'JPM', 'K', 'KDP', 'KHC', 'KIM', 'KLAC',
    'KMB', 'KMI', 'KMX', 'KO', 'KR', 'L', 'LDOS', 'LEG', 'LEN', 'LH',
    'LHX', 'LIN', 'LKQ', 'LLY', 'LMT', 'LNC', 'LNT', 'LOW', 'LRCX', 'LULU',
    'LUMN', 'LUV', 'LVS', 'LW', 'LYB', 'LYV', 'MA', 'MAA', 'MAR', 'MAS',
    'MCD', 'MCHP', 'MCK', 'MCO', 'MDLZ', 'MDT', 'MET', 'META', 'MGM', 'MHK',
    'MKC', 'MKTX', 'MLI', 'MLM', 'MMC', 'MMM', 'MNST', 'MO', 'MOH', 'MOS',
    'MPC', 'MPWR', 'MRK', 'MRNA', 'MRO', 'MS', 'MSCI', 'MSFT', 'MSI', 'MTB',
    'MTCH', 'MTD', 'MU', 'NCLH', 'NDAQ', 'NDSN', 'NEE', 'NEM', 'NFLX', 'NI',
    'NKE', 'NLOK', 'NLSN', 'NOC', 'NOV', 'NOW', 'NRG', 'NSC', 'NTAP', 'NTRS',
    'NUE', 'NVDA', 'NVR', 'NWL', 'NWS', 'NWSA', 'NXST', 'O', 'ODFL', 'OGN',
    'OKE', 'OMC', 'ON', 'ORCL', 'ORLY', 'OTIS', 'OXY', 'PARA', 'PAYC', 'PAYX',
    'PCAR', 'PCG', 'PEAK', 'PEG', 'PENN', 'PEP', 'PFE', 'PG', 'PGR', 'PH',
    'PHM', 'PKG', 'PLD', 'PM', 'PNC', 'PNR', 'PNW', 'POOL', 'PPG', 'PPL',
    'PRU', 'PSA', 'PSX', 'PTC', 'PTON', 'PWR', 'PXD', 'PYPL', 'QCOM', 'QRVO',
    'RCL', 'RE', 'REG', 'REGN', 'RF', 'RHI', 'RJF', 'RL', 'RMD', 'ROK',
    'ROL', 'ROP', 'ROST', 'RSG', 'RTX', 'RVTY', 'SBAC', 'SBUX', 'SCHW', 'SCI',
    'SEIC', 'SHW', 'SIVB', 'SJM', 'SLB', 'SNA', 'SNPS', 'SO', 'SPG', 'SPGI',
    'SRE', 'STE', 'STT', 'STX', 'STZ', 'SWK', 'SWKS', 'SYF', 'SYK', 'SYY',
    'T', 'TAP', 'TDG', 'TDY', 'TECH', 'TEL', 'TER', 'TFC', 'TFX', 'TGT',
    'TJX', 'TMO', 'TMUS', 'TPG', 'TROW', 'TRV', 'TSCO', 'TSLA', 'TSN', 'TT',
    'TTWO', 'TXN', 'TXT', 'TYL', 'UA', 'UAA', 'UAL', 'UDR', 'UHS', 'ULTA',
    'UNH', 'UNP', 'UPS', 'URI', 'USB', 'V', 'VFC', 'VICI', 'VLO', 'VMC',
    'VRSK', 'VRSN', 'VRTX', 'VTR', 'VTRS', 'VZ', 'WAB', 'WAT', 'WBA', 'WBD',
    'WDC', 'WEC', 'WELL', 'WFC', 'WHR', 'WLTW', 'WM', 'WMB', 'WMT', 'WRB',
    'WRK', 'WST', 'WTW', 'WY', 'WYNN', 'XEL', 'XOM', 'XRAY', 'XYL', 'YUM',
    'ZBH', 'ZBRA', 'ZION', 'ZTS'
}



def load_sp500_tickers(file_path=None):
    """
    Load S&P 500 tickers from file or use hardcoded list.
    
    Args:
        file_path: Optional path to CSV file with tickers (should have 'ticker' column)
    
    Returns:
        Set of ticker symbols (uppercase)
    """
    if file_path and Path(file_path).exists():
        df = pd.read_csv(file_path)
        if 'ticker' in df.columns:
            tickers = set(df['ticker'].str.upper().str.strip())
            print(f"Loaded {len(tickers)} tickers from {file_path}")
            return tickers
    
    print(f"Using hardcoded S&P 500 list: {len(SP500_TICKERS)} tickers")
    return SP500_TICKERS


def filter_dataset_to_sp500(dataset, sp500_tickers, output_path=None, 
                            max_rows=None, save_batch_size=50000,
                            start_date=None, end_date=None, start_from_row=None):
    """
    Filter Hugging Face dataset to only S&P 500 companies.
    
    Args:
        dataset: Hugging Face dataset object
        sp500_tickers: Set of S&P 500 ticker symbols
        output_path: Path to save filtered CSV (optional)
        max_rows: Maximum rows to process (for testing, None = all)
        save_batch_size: Save in batches of this size
        start_date: Start date for filtering (YYYY-MM-DD format, inclusive)
        end_date: End date for filtering (YYYY-MM-DD format, inclusive)
        start_from_row: Start processing from specific source row index (0-based, overrides checkpoint)
    
    Returns:
        DataFrame with filtered data (or None if saving to file)
    """
    filtered_rows = []
    ticker_counts = Counter()
    total_processed = 0
    total_filtered = 0
    file_exists = False
    batches_saved = 0
    
    # Parse date filters
    if start_date:
        start_date_obj = pd.to_datetime(start_date)
    else:
        start_date_obj = None
    
    if end_date:
        end_date_obj = pd.to_datetime(end_date)
    else:
        end_date_obj = None
    
    # Checkpoint handling
    checkpoint_path = None
    start_from_index = 0
    
    # Manual start row overrides checkpoint
    if start_from_row is not None:
        start_from_index = start_from_row
        print(f"✓ Starting from manually specified row: {start_from_index:,}")
    elif output_path:
        checkpoint_path = str(output_path) + ".checkpoint"
        if Path(checkpoint_path).exists():
            try:
                with open(checkpoint_path, 'r') as f:
                    checkpoint_data = json.load(f)
                    start_from_index = checkpoint_data.get('last_processed_index', 0)
                    # Restore ticker counts if available
                    if 'ticker_counts' in checkpoint_data:
                        ticker_counts.update(checkpoint_data['ticker_counts'])
                    batches_saved = checkpoint_data.get('batches_saved', 0)
                    total_filtered = checkpoint_data.get('total_filtered', 0)
                print(f"✓ Resuming from checkpoint: row {start_from_index:,}")
                print(f"  Already processed: {total_filtered:,} articles, {batches_saved} batches")
            except Exception as e:
                print(f"⚠ Could not load checkpoint: {e}")
                print("  Starting from beginning...")
                start_from_index = 0
        
        # Check if output file exists
        if Path(output_path).exists():
            file_exists = True
            if start_from_index == 0:
                print(f"⚠ Output file exists - will append to it")
    
    print("="*60)
    print("FILTERING DATASET TO S&P 500 COMPANIES")
    print("="*60)
    print(f"S&P 500 tickers: {len(sp500_tickers)}")
    if start_date_obj or end_date_obj:
        date_range = f"{start_date_obj.strftime('%Y-%m-%d') if start_date_obj else 'all'} to {end_date_obj.strftime('%Y-%m-%d') if end_date_obj else 'all'}"
        print(f"Date range: {date_range}")
    print(f"Output: {output_path or 'Return DataFrame only'}")
    print("="*60)
    
    # Process dataset - skip already processed rows if resuming
    dataset_iter = iter(dataset['train'])
    # Skip rows we've already processed
    if start_from_index > 0:
        print(f"Skipping {start_from_index:,} already processed rows...")
        for _ in range(start_from_index):
            try:
                next(dataset_iter)
            except StopIteration:
                break
    
    # Process remaining dataset
    for i, example in enumerate(tqdm(dataset_iter, desc="Processing", initial=start_from_index), start=start_from_index):
        # Check max rows limit
        if max_rows and i >= max_rows:
            break
        
        ticker = example.get('Stock_symbol', '')
        date_str = example.get('Date', '')
        
        # Filter by date if specified
        if start_date_obj or end_date_obj:
            try:
                article_date = pd.to_datetime(date_str)
                # Skip if date is outside range
                if start_date_obj and article_date < start_date_obj:
                    total_processed += 1
                    continue
                if end_date_obj and article_date > end_date_obj:
                    total_processed += 1
                    continue
            except (ValueError, TypeError):
                # Invalid date - skip this row
                total_processed += 1
                continue
        
        # Note: total_processed is incremented at the end of the loop
        
        # Normalize ticker
        if ticker:
            ticker = str(ticker).strip().upper()
            
            # Check if ticker is in S&P 500
            if ticker in sp500_tickers:
                # Check if there's content (title or article)
                has_content = example.get('Article_title') or example.get('Article')
                
                if has_content:
                    filtered_rows.append(example)
                    ticker_counts[ticker] += 1
                    total_filtered += 1
                    
                    # Save in batches if output path specified
                    if output_path and len(filtered_rows) >= save_batch_size:
                        df_batch = pd.DataFrame(filtered_rows)
                        # Append mode if file exists or we've saved batches before
                        mode = 'a' if (file_exists or batches_saved > 0) else 'w'
                        # Only write header if it's the first write
                        header = (not file_exists and batches_saved == 0)
                        df_batch.to_csv(output_path, mode=mode, header=header, index=False)
                        filtered_rows = []  # Clear batch
                        batches_saved += 1
                        file_exists = True  # Mark that file now exists
                        print(f"\n  ✓ Saved batch #{batches_saved}: {total_filtered:,} total rows saved")
                        
                        # Save checkpoint
                        if checkpoint_path:
                            checkpoint_data = {
                                'last_processed_index': i,
                                'total_filtered': total_filtered,
                                'batches_saved': batches_saved,
                                'ticker_counts': dict(ticker_counts)
                            }
                            with open(checkpoint_path, 'w') as f:
                                json.dump(checkpoint_data, f)
                            print(f"  ✓ Checkpoint saved at source row {i:,}")
        
        total_processed += 1
        
        # Progress update
        if (i + 1) % 100000 == 0:
            print(f"\n  Processed: {i+1:,} | Filtered: {total_filtered:,} | "
                  f"Unique S&P 500 tickers found: {len(ticker_counts)}")
    
    # Save remaining rows
    if filtered_rows:
        df_batch = pd.DataFrame(filtered_rows)
        mode = 'a' if (file_exists or batches_saved > 0) else 'w'
        header = (not file_exists and batches_saved == 0)
        if output_path:
            df_batch.to_csv(output_path, mode=mode, header=header, index=False)
            print(f"\n  ✓ Saved final batch: {total_filtered:,} total rows")
            
            # Save final checkpoint
            if checkpoint_path:
                checkpoint_data = {
                    'last_processed_index': i,
                    'total_filtered': total_filtered,
                    'batches_saved': batches_saved,
                    'ticker_counts': dict(ticker_counts),
                    'completed': True
                }
                with open(checkpoint_path, 'w') as f:
                    json.dump(checkpoint_data, f)
                print(f"  ✓ Final checkpoint saved")
    
    # Print completion summary (DRY - shared code)
    print("\n" + "="*60)
    print("FILTERING COMPLETE")
    print("="*60)
    print(f"Total processed: {total_processed:,}")
    print(f"Total filtered (S&P 500): {total_filtered:,}")
    print(f"Unique S&P 500 tickers found: {len(ticker_counts)}")
    
    # Add file-specific info if saving to file
    if output_path:
        file_size_mb = Path(output_path).stat().st_size / (1024**2) if Path(output_path).exists() else 0
        print(f"Output file size: {file_size_mb:.1f} MB")
        print(f"\n⚠ Note: Full dataset saved to file (not loaded into memory)")
        print(f"   Use pd.read_csv('{output_path}') to load when needed")
    
    # Print top companies (shared)
    print(f"\nTop 20 S&P 500 companies by article count:")
    print("-" * 60)
    for ticker, count in ticker_counts.most_common(20):
        print(f"  {ticker:<6} {count:>8,} articles")
    
    # Return appropriate result based on output mode
    if output_path:
        # File was saved in batches - don't load it back into memory!
        return None, ticker_counts
    else:
        # No output path - return DataFrame from memory
        df = pd.DataFrame(filtered_rows) if filtered_rows else pd.DataFrame()
        return df, ticker_counts


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Filter FNSPID dataset to S&P 500 companies")
    parser.add_argument('--output', type=str, default=None,
                      help='Output CSV file path (optional)')
    parser.add_argument('--ticker-file', type=str, default=None,
                      help='CSV file with S&P 500 tickers (optional)')
    parser.add_argument('--max-rows', type=int, default=None,
                      help='Maximum rows to process (for testing)')
    parser.add_argument('--batch-size', type=int, default=50000,
                      help='Batch size for saving CSV')
    parser.add_argument('--use-drive', action='store_true',
                      help='Save to Google Drive (mounts drive automatically in Colab)')
    parser.add_argument('--start-date', type=str, default='2015-01-01',
                      help='Start date for filtering (YYYY-MM-DD format, default: 2015-01-01)')
    parser.add_argument('--end-date', type=str, default='2022-12-31',
                      help='End date for filtering (YYYY-MM-DD format, default: 2022-12-31)')
    parser.add_argument('--reset-checkpoint', action='store_true',
                      help='Delete checkpoint and start from beginning')
    parser.add_argument('--start-from-row', type=int, default=None,
                      help='Start processing from a specific source row index (0-based). Overrides checkpoint resume.')
    
    args = parser.parse_args()
    
    # Mount Google Drive if requested
    if args.use_drive:
        try:
            from google.colab import drive
            # Check if drive is already mounted
            drive_path = Path('/content/drive/MyDrive')
            if drive_path.exists() and drive_path.is_dir():
                print("✓ Google Drive already mounted")
            else:
                drive.mount('/content/drive')
                print("✓ Google Drive mounted")
        except ImportError:
            print("⚠ Not running in Colab - --use-drive ignored")
            args.use_drive = False
        except Exception as e:
            print(f"⚠ Error mounting Google Drive: {e}")
            print("  Continuing without Drive mount...")
            args.use_drive = False
    
    # Load S&P 500 tickers
    sp500_tickers = load_sp500_tickers(args.ticker_file)
    
    # Load dataset
    print("\nLoading FNSPID dataset from Hugging Face...")
    dataset = load_dataset("Zihan1004/FNSPID", streaming=True)
    
    # Set output path
    if args.output:
        output_path = args.output
    elif args.use_drive:
        output_path = '/content/drive/MyDrive/fnspid_sp500_filtered.csv'
    else:
        output_path = '/content/fnspid_sp500_filtered.csv'
    
    # Handle checkpoint reset
    checkpoint_path = str(output_path) + ".checkpoint"
    if args.reset_checkpoint:
        if Path(checkpoint_path).exists():
            Path(checkpoint_path).unlink()
        if Path(output_path).exists():
            Path(output_path).unlink()
        # Also delete counts file if it exists
        counts_path = output_path.replace('.csv', '_counts.csv')
        if Path(counts_path).exists():
            Path(counts_path).unlink()
        print(f"✓ Checkpoint and output files reset - starting from scratch")
    
    print(f"\n💾 Saving to: {output_path}")
    if '/content/drive' in output_path:
        print("   ✓ Google Drive - will persist after session ends")
    elif '/content/' in output_path:
        print("   ⚠ WARNING: /content/ is temporary - will be lost when session ends!")
        print("   💡 Use --use-drive or --output /content/drive/MyDrive/... to save permanently")
    df, ticker_counts = filter_dataset_to_sp500(
        dataset, 
        sp500_tickers,
        output_path=output_path,
        max_rows=args.max_rows,
        save_batch_size=args.batch_size,
        start_date=args.start_date,
        end_date=args.end_date,
        start_from_row=args.start_from_row
    )
    
    # Save ticker counts
    counts_df = pd.DataFrame(ticker_counts.most_common(), 
                            columns=['ticker', 'article_count'])
    counts_path = output_path.replace('.csv', '_counts.csv')
    counts_df.to_csv(counts_path, index=False)
    print(f"\n✓ Saved article counts to: {counts_path}")
    
    print(f"\n{'='*60}")
    print("COMPLETE")
    print(f"{'='*60}")
    print(f"Filtered dataset saved to: {output_path}")
    
    if df is not None:
        print(f"Shape: {df.shape}")
        print(f"Total articles: {len(df):,}")
    else:
        # File was saved - get info without loading
        if Path(output_path).exists():
            file_size = Path(output_path).stat().st_size / (1024**2)
            print(f"File size: {file_size:.1f} MB")
            print(f"Total articles: {sum(ticker_counts.values()):,}")
            print(f"\n💡 To load the dataset later:")
            print(f"   df = pd.read_csv('{output_path}')")
    
    print(f"Unique S&P 500 companies: {len(ticker_counts):,}")


if __name__ == "__main__":
    main()

