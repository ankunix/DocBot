import os
import pandas as pd
from sec_edgar.config import HEADERS, DATA_DIR
from sec_edgar.utils import get_company_tickers, pull_all_history
from sec_edgar.downloader import sync_download_all_forms
from progiter.manager import ProgressManager  # added import

def extract_filings(form, tickers):
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
    df_filtered = df_history[df_history['form'] == form]
    if df_filtered.empty:
        print(f"No records found for form {form}.")
        return

    sync_download_all_forms(df_filtered, form)
   