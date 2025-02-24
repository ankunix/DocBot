import os
import aiohttp
import asyncio
import time
import requests
import pandas as pd
from progiter.manager import ProgressManager  # updated import
from sec_edgar.config import BASE_URL, DATA_DIR, HEADERS
import logging

# Ensure the DATA_DIR exists
os.makedirs(DATA_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_sync_with_retries(url, retries=3, backoff_factor=2):
    """Synchronous fetch with retries."""
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=HEADERS, timeout=10)
            if response.status_code == 200:
                return response.text
            elif response.status_code in {429, 500, 502, 503, 504}:
                time.sleep(backoff_factor ** attempt)
        except requests.Timeout as e:
            logger.error(f"Timeout error for URL {url}: {e}")
            time.sleep(backoff_factor ** attempt)
    logger.error(f"Failed to fetch URL {url} after {retries} attempts")
    return None

def sync_download_form(row, form):
    """Synchronously downloads a single filing."""
    cik = str(row['cik']).zfill(10)
    accession_number = row['accessionNumber'].replace('-', '')
    primary_doc = row['primaryDocument']
    url = f"{BASE_URL}/Archives/edgar/data/{cik}/{accession_number}/{primary_doc}"
    folder = os.path.join(DATA_DIR, row['ticker'])
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, primary_doc)
    
    if os.path.exists(file_path):
        return
    
    content = fetch_sync_with_retries(url)
    if content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

def sync_download_all_forms(df_history, form):
    """Synchronously downloads filings for companies in df_history."""
    pman = ProgressManager()
    with pman:
        for _, row in pman.progiter(df_history.iterrows(), total=len(df_history), desc="Downloading filings (Sync)"):
            sync_download_form(row, form)
