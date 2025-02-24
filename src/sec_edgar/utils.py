import os
import pandas as pd
from progiter import ProgIter
import requests
from sec_edgar.config import BASE_URL, HEADERS, DATA_DIR

def get_company_tickers():
    """
    Loads company tickers from the JSON file.
    Assumes 'company_tickers.json' is located in DATA_DIR.
    """
    file_path = os.path.join(DATA_DIR, 'company_tickers.json')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Company tickers file not found at {file_path}")
    df = pd.read_json(file_path).T
    df.rename(columns={'cik_str': 'cik', 'title': 'name'}, inplace=True)
    return df

def get_current_filing_history(url, header):
    response = requests.get(url, headers=header)
    response.raise_for_status()
    company_filings = response.json()
    company_filings_df = pd.DataFrame(company_filings["filings"]["recent"])
    return company_filings_df

def pull_all_history(df, header):
    """
    For each company (row in df), pull its filing history from SEC.
    """
    df_all = pd.DataFrame()
    for index, row in ProgIter(df.iterrows(), total=df.shape[0], desc="Pulling filing history"):
        CIK = row['cik']
        url = f"https://data.sec.gov/submissions/CIK{str(CIK).zfill(10)}.json"
        company_filings_df = get_current_filing_history(url, header)
        company_filings_df['ticker'] = row['ticker']
        company_filings_df['cik'] = row['cik']
        df_all = pd.concat([company_filings_df, df_all], ignore_index=True)
    return df_all

def save_history_to_csv(df_history):
    """Saves the filing history DataFrame to CSV."""
    csv_path = os.path.join(DATA_DIR, 'filing_history.csv')
    df_history.to_csv(csv_path, index=False)
    print(f"Filing history saved to {csv_path}")