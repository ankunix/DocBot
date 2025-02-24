import os
import requests
from sec_edgar.config import FILINGS_DIR
from sec_edgar.utils import fetch_data_with_retry
from progiter.manager import ProgressManager

def get_filing_url(cik, accession_number, document):
    """Construct the URL for retrieving a filing document."""
    return f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_number}/{document}"

def download_filing(cik, accession_number, document):
    """Download a filing document and save it to disk."""
    url = get_filing_url(cik, accession_number, document)
    response = fetch_data_with_retry(url, cache=False, cache_dir=FILINGS_DIR)
    
    if response:
        file_path = os.path.join(FILINGS_DIR, f"{accession_number}_{document}")
        with open(file_path, "w") as f:
            f.write(response)
        print(f"Downloaded: {file_path}")

def download_all_forms(df, form, header):
    """Download all filings of a given form (e.g., 10-K)."""
    df_ = df[df.form == form]
    with ProgressManager() as pman:
        pman = pman or ProgressManager()  # fallback if None
        for _, row in pman.progiter(df_.iterrows(), total=df_.shape[0], desc="Downloading filings"):
            url = f"https://www.sec.gov/Archives/edgar/data/{row['cik']}/{row['accessionNumber'].replace('-', '')}/{row['primaryDocument']}"
            try:
                req_content = requests.get(url, headers=header, timeout=10).content.decode("utf-8")
            except Exception as ex:
                print(f"Failed downloading {url}: {ex}")
                continue
            # Save filings by ticker within FILINGS_DIR
            directory = os.path.join(FILINGS_DIR, row['ticker'])
            os.makedirs(directory, exist_ok=True)
            file_path = os.path.join(directory, row['primaryDocument'])
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(req_content)
                print(f"Downloaded: {file_path}")
            except Exception as ex:
                print(f"Failed saving {file_path}: {ex}")
