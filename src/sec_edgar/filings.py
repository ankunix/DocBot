import pandas as pd
from tqdm import tqdm
from sec_edgar.config import DATA_DIR
from sec_edgar.utils import fetch_data_with_retry

def get_current_filing_history(url):
    """Fetch the filing history for a company given its submissions URL."""
    response = fetch_data_with_retry(url)

    if response:
        if "filings" in response and "recent" in response["filings"]:
            filings_data = response["filings"]["recent"]
            if not filings_data:
                print("No filings found in the response.")
                return pd.DataFrame()

            filings_df = pd.DataFrame(filings_data)
            print(f"Fetched {len(filings_df)} filings.")  # Debugging output
            return filings_df
        else:
            print("Unexpected response structure:", response)
    else:
        print("Failed to fetch data.")

    return pd.DataFrame()

def pull_all_history(df, tickers=None):
    """
    Pull filing history for all companies in the provided DataFrame.
    Optionally, filter by a list of tickers.
    """
    if tickers:
        df = df[df['ticker'].isin(tickers)]
    
    df_all = pd.DataFrame()
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Pulling Filing History"):
        cik = row['cik']
        url = f"https://data.sec.gov/submissions/CIK{str(cik).zfill(10)}.json"
        company_filings_df = get_current_filing_history(url)
        company_filings_df['ticker'] = row['ticker']
        company_filings_df['cik'] = row['cik']
        df_all = pd.concat([company_filings_df, df_all])
    return df_all
