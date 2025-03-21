"""
Wrapper script to run Streamlit with proper configuration
"""
import os
import sys
import subprocess

# Set environment variables to control Streamlit's behavior
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "poll"
os.environ["STREAMLIT_SERVER_HEADLESS"] = "false"
os.environ["STREAMLIT_SERVER_FILE_WATCHER_IGNORE_PATTERNS"] = "torch.*;transformers.*;*.pyc"

# Get path to the streamlit executable in the virtual environment
venv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv")
python_bin = os.path.join(venv_path, "bin", "python")

# Run streamlit using subprocess to ensure environment variables are applied
cmd = [python_bin, "-m", "streamlit", "run", "src/simple_app.py", 
       "--server.fileWatcherType=poll"]

print(f"Running command: {' '.join(cmd)}")
subprocess.run(cmd)
