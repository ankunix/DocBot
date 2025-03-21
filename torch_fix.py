#!/usr/bin/env python
"""
PyTorch compatibility fix for Streamlit

This script patches torch._classes to prevent errors in Streamlit's file watcher.
Run this script directly to apply the patch and launch your Streamlit application.

Usage:
    python torch_fix.py src/app.py
"""

import os
import sys
import subprocess
import importlib

def patch_torch():
    """Patch the torch._classes module to avoid Streamlit errors"""
    try:
        import torch
        
        # Only patch if torch is installed
        if hasattr(torch, '_classes'):
            # Create a custom property for __path__._path
            class CustomPath:
                def __init__(self):
                    self._path = []
                
                @property
                def _path(self):
                    return []
                
                @_path.setter
                def _path(self, value):
                    pass
            
            # Apply the patch to torch._classes.__path__
            torch._classes.__path__ = CustomPath()
            
            print("✅ Successfully patched PyTorch for Streamlit compatibility")
            return True
    except ImportError:
        print("PyTorch not found, no patching needed")
    except Exception as e:
        print(f"⚠️ Warning: Failed to patch PyTorch: {e}")
    
    return False

def run_streamlit(app_path):
    """Run streamlit as a subprocess instead of importing it directly"""
    # Disable Streamlit's file watcher
    os.environ["STREAMLIT_DISABLE_FILE_WATCHER"] = "true"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

    # Use flags to disable development mode (stops watchers)
    cmd = [
        "streamlit", "run", app_path,
        "--global.developmentMode=false",   # disables developer mode
        "--server.runOnSave=false"          # stops auto reload
    ]
    print(f"🚀 Launching Streamlit with command: {' '.join(cmd)}")
    
    # Run subprocess with inherited stdout/stderr
    process = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
    
    try:
        # Wait for the process to complete
        process.wait()
        return process.returncode
    except KeyboardInterrupt:
        print("Stopping Streamlit...")
        process.terminate()
        return 1

def main():
    """Apply patch and launch Streamlit"""
    # Apply the PyTorch patch
    patch_torch()
    
    # Get the app path from command line or use default
    app_path = sys.argv[1] if len(sys.argv) > 1 else "src/app.py"
    
    # Run Streamlit as a subprocess
    return run_streamlit(app_path)

if __name__ == "__main__":
    sys.exit(main())
