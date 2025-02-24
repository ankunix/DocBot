import os

# BASE_DIR: two level up from this file (i.e. the src directory's parent)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
# Directory where downloaded filings will be stored
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# Directory for logs
LOG_DIR = os.path.join(BASE_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, 'sec_edgar.log')

# SEC API configuration
BASE_URL = "https://www.sec.gov"

# SEC request headers
HEADERS = {
    "User-Agent": "harshit55 harshit.gola.off@gmail.com",
    "Accept-Encoding": "gzip, deflate"
}