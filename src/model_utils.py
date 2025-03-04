"""
Utilities for model caching and management
"""
import os
import torch
import psutil
import shutil
from pathlib import Path

# Define a cache directory
CACHE_DIR = Path(os.path.expanduser("~/.docbot_model_cache"))
CACHE_DIR.mkdir(exist_ok=True, parents=True)

def get_available_memory_gb():
    """Get available system memory in GB"""
    return psutil.virtual_memory().available / (1024 * 1024 * 1024)

def clear_model_cache():
    """Clear the downloaded model cache"""
    try:
        shutil.rmtree(CACHE_DIR)
        CACHE_DIR.mkdir(exist_ok=True)
        return True
    except Exception as e:
        print(f"Error clearing cache: {str(e)}")
        return False

def get_cache_size():
    """Get the total size of cached models in MB"""
    total_size = 0
    for path in CACHE_DIR.glob('**/*'):
        if path.is_file():
            total_size += path.stat().st_size
    return total_size / (1024 * 1024)  # Convert to MB

def is_model_cached(model_name):
    """Check if a model is already cached"""
    model_folder = CACHE_DIR / model_name.replace('/', '--')
    return model_folder.exists()

def get_recommended_model():
    """Recommend a model based on available system resources"""
    available_memory = get_available_memory_gb()
    
    if available_memory < 1.0:
        return "distilgpt2"  # Very small model
    elif available_memory < 2.0:
        return "facebook/opt-125m"  # Small model
    elif available_memory < 4.0:
        return "microsoft/phi-1_5"  # Medium model
    else:
        return "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Larger model
