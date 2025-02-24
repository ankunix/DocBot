import glob
import os
import pandas as pd
import logging
from sec_edgar.config import DATA_DIR
from sec_edgar.parser_lib import parse_10k_filing
from progiter.manager import ProgressManager  # added import

logger = logging.getLogger(__name__)

def parse_and_save_filings():
    """
    Locate filing files in DATA_DIR, parse each using parse_10k_filing, and save a consolidated CSV.
    The output DataFrame includes columns: Text, ticker, accessionNumber, cik, and filepath.
    """
    filing_files = glob.glob(os.path.join(DATA_DIR, "*", "*.htm"))
    if not filing_files:
        logger.error("No downloaded filing files found.")
        return
    processed_data = []
    with ProgressManager() as pman:
        pman = pman or ProgressManager()  # fallback if __enter__ returns None
        for file_path in pman.progiter(filing_files, desc="Parsing filings"):
            try:
                sections = parse_10k_filing(file_path, 0)  # returns list of sections
                # Extract ticker from the file's parent directory name.
                ticker = os.path.basename(os.path.dirname(file_path))
                for text in sections:
                    row_data = {
                        "Text": text,
                        "ticker": ticker,
                        "accessionNumber": "",
                        "cik": "",
                        "filepath": file_path
                    }
                    processed_data.append(row_data)
            except Exception as ex:
                logger.error("Error processing %s: %s", file_path, ex)
    processed_df = pd.DataFrame(processed_data)
    output_csv = os.path.join(DATA_DIR, "processed_filings.csv")
    processed_df.to_csv(output_csv, index=False)
    logger.info("Processed filings saved to %s", output_csv)
