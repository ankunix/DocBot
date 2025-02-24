# SEC Extractor

This project downloads and extracts SEC filings from the EDGAR system.

## Project Structure

- **/data**: Directory for downloaded filings and CSV history.
- **/logs**: Directory for log files.
- **/src/sec_edgar/**
  - **config.py**: Configuration for SEC API, file paths, and headers.
  - **utils.py**: Utility functions for loading tickers and pulling filing history.
  - **logger.py**: Logging functions for download status.
  - **downloader.py**: Functions to download filings synchronously.
  - **extractor.py (or main.py)**: Main driver for triggering downloads.

## Usage

1. Ensure that the `company_tickers.json` file is placed in the `data` directory.
2. Run the main script from the project root:
   ```bash
   python src/sec_edgar/main.py --form "10-K" --tickers "AAPL,MSFT"
   ```
3. Downloaded filings, CSV history, and log files will be saved in the **/data** and **/logs** directories.

## Dependencies

- Python 3.x  
- `click`
- `requests`
- `aiohttp`
- `pandas`
- `progiter`

## License

This project is open source.