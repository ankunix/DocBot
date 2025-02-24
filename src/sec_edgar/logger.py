import logging
import os
import pandas as pd
from progiter import ProgIter
from sec_edgar.config import LOG_FILE

# Configure the logger
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def log_worker(log_queue):
    """
    Continuously reads log entries from log_queue and writes them to a CSV file.
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    csv_log_file = os.path.join(base_dir, 'data', 'download_history.csv')
    
    with ProgIter(desc="Logging progress") as progress_bar:  # Removed 'unit' argument
        while True:
            log_entry = log_queue.get()
            if log_entry is None:
                break
            df_log = pd.DataFrame([log_entry])
            df_log.to_csv(csv_log_file, mode="a", header=not os.path.exists(csv_log_file), index=False)
            log_queue.task_done()
            progress_bar.update(1)

if __name__ == "__main__":
    logger.info("Logger initialized successfully.")
