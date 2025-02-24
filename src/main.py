# main.py

import click
from sec_edgar.utils import get_company_tickers, pull_all_history, save_history_to_csv
from sec_edgar.downloader import sync_download_all_forms
from sec_edgar.config import HEADERS
from sec_edgar.process_filings import parse_and_save_filings

@click.command()
@click.option('--top_n', default=10, help="Number of top records to download.")
@click.option('--form', default='10-K', help="Form type to download (e.g., '10-K').")
@click.option('--tickers', default=None, type=str, help="Comma-separated list of tickers (e.g., 'AAPL,MSFT')")
def main(top_n, form, tickers):
    """Main function to manage SEC form downloads in synchronous mode."""
    
    df_tickers = get_company_tickers()
    if tickers:
        tickers_list = [ticker.strip() for ticker in tickers.split(',')]
    else:
        tickers_list = []

    if tickers_list:
        df_tickers_filtered = df_tickers[df_tickers['ticker'].isin(tickers_list)]
    else:
        df_tickers_filtered = df_tickers

    df_history = pull_all_history(df_tickers_filtered, HEADERS)
    save_history_to_csv(df_history)
    # Filter the history DataFrame for the given form type
    df_filtered = df_history[df_history['form'] == form]
    
    if df_filtered.empty:
        print(f"No records found for form {form}.")
        return

    # Always use synchronous download
    sync_download_all_forms(df_filtered, form)

    parse_and_save_filings(df_filtered)

if __name__ == "__main__":
    main()
