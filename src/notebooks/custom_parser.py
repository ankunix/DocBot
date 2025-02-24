import glob
import os
import re
import pandas as pd
from progiter.manager import ProgressManager  # updated import
from sec_edgar.config import DATA_DIR
from sec_edgar.parser_lib import parse_10k_filing  # library function

def assign_category(df):
    df['Category'] = ''
    keywords = {
        'Item 1.': 'Business Overview',
        'Item 1A.': 'Risk Factors',
        'Item 7.': 'MD&A'
    }
    pman = ProgressManager()
    with pman:
        for index, row in pman.progiter(list(df.iterrows()), desc="Assigning categories"):
            text = row.get('CombinedText', '')
            for keyword, category in keywords.items():
                if keyword in text:
                    df.at[index, 'Category'] = category
                    break
    return df

def extract_metadata(file_path):
    ticker = os.path.basename(os.path.dirname(file_path))
    basename = os.path.splitext(os.path.basename(file_path))[0]
    m = re.search(r"a10[-_]?k(\d+)", basename, re.IGNORECASE)
    accession = m.group(1) if m else ""
    return ticker, accession

def parse_and_save_filings():
    filing_files = glob.glob(os.path.join(DATA_DIR, "*", "*.htm"))
    if not filing_files:
        print("No downloaded filing files found.")
        return

    processed_data = []
    pman = ProgressManager()
    with pman:
        for file_path in pman.progiter(filing_files, desc="Parsing filings"):
            sections = parse_10k_filing(file_path, 0)
            combined_text = " ".join(sections)
            ticker, accession = extract_metadata(file_path)
            row_data = {
                "File": file_path,
                "Ticker": ticker,
                "Accession": accession,
                "Business": sections[0] if len(sections) >= 1 else "",
                "Risk": sections[1] if len(sections) >= 2 else "",
                "MDA": sections[2] if len(sections) >= 3 else "",
                "CombinedText": combined_text
            }
            processed_data.append(row_data)

    df = pd.DataFrame(processed_data)
    df = assign_category(df)
    output_csv = os.path.join(DATA_DIR, "processed_filings.csv")
    df.to_csv(output_csv, index=False)
    print(f"Processed filings saved to {output_csv}")
