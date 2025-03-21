#!/usr/bin/env python
import os
import sys
import warnings

# Set environment variables to improve compatibility
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Disable parallelism in tokenizers

# Fix for torch._classes.__path__._path errors in Streamlit's file watcher
# Monkey patch torch._classes to prevent the error
import torch

# Create a custom property to handle the problematic access pattern
def _patch_torch_classes():
    class PathFixerProperty:
        def __get__(self, obj, objtype=None):
            return []
    
    # Check if torch._classes exists before patching
    if hasattr(torch, '_classes'):
        # Create a class to handle __path__ attribute
        class PathFixer:
            _path = PathFixerProperty()
        
        # Patch torch._classes.__path__ if it doesn't have _path
        if not hasattr(torch._classes.__path__, '_path'):
            torch._classes.__path__ = PathFixer()

# Apply the patch
try:
    _patch_torch_classes()
    print("✅ Applied PyTorch compatibility patch")
except Exception as e:
    print(f"⚠️ Warning: Could not apply PyTorch patch: {str(e)}")

# Reduce memory usage for Streamlit
os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
os.environ['STREAMLIT_SERVER_ENABLE_TELEMETRY'] = 'false'

# Set memory efficiency options for PyTorch
os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'

# Configure Streamlit command line arguments
streamlit_args = [
    # Main app file
    "run",
    "src/app.py",
    # Additional streamlit options
    "--server.maxUploadSize=10",
    "--global.developmentMode=false",
    "--logger.level=error",
    "--server.enableXsrfProtection=false",
]

# Start Streamlit with the configured options
print("🚀 Starting DocBot RAG Application")
print("Press Ctrl+C to exit")

# Import and run streamlit
import streamlit.cli as stcli

# Execute Streamlit with our custom arguments
if __name__ == "__main__":
    sys.argv = ["streamlit"] + streamlit_args
    sys.exit(stcli.main())
