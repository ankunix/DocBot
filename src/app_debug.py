"""
Debug utility for testing model loading and basic functionality
"""
import os
import time
import torch
import traceback
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_model_load(model_name="distilgpt2"):
    """Test basic model loading capabilities"""
    print(f"Testing model loading for: {model_name}")
    
    # Log system info
    print(f"\nSystem information:")
    print(f"Python version: {os.sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    try:
        print(f"\nTrying to load tokenizer for {model_name}...")
        start_time = time.time()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer_time = time.time() - start_time
        print(f"✓ Tokenizer loaded successfully in {tokenizer_time:.2f} seconds")
        
        print(f"\nTrying to load model for {model_name}...")
        start_time = time.time()
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model_time = time.time() - start_time
        print(f"✓ Model loaded successfully in {model_time:.2f} seconds")
        
        # Test basic inference
        print("\nTesting basic inference...")
        inputs = tokenizer("Hello, my name is", return_tensors="pt")
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=20)
        inference_time = time.time() - start_time
        print(f"✓ Inference completed in {inference_time:.2f} seconds")
        print(f"Output: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print(traceback.format_exc())
        return False

def verify_huggingface_cache():
    """Check if Hugging Face cache is accessible and working"""
    cache_dir = Path(os.path.expanduser('~/.cache/huggingface'))
    
    print(f"\nChecking Hugging Face cache at: {cache_dir}")
    
    if not cache_dir.exists():
        print(f"❌ Cache directory doesn't exist. Will be created on first model download.")
        return False
    
    # Check if cache is writable
    try:
        test_file = cache_dir / "test_write.txt"
        with open(test_file, 'w') as f:
            f.write("Test write access")
        test_file.unlink()  # Remove the test file
        print(f"✓ Cache directory is writable")
        
        # List cache contents
        print("\nCurrent cache contents:")
        models_dir = cache_dir / "models--"
        if models_dir.exists():
            for path in models_dir.glob("*"):
                if path.is_dir():
                    model_name = path.name.replace("--", "/")
                    print(f"  - {model_name}")
        else:
            print("  (No models cached yet)")
        
        return True
    except Exception as e:
        print(f"❌ Cache directory is not writable: {str(e)}")
        return False

if __name__ == "__main__":
    print("=== DocBot Model Debug Utility ===\n")
    
    # Check HF cache status
    verify_huggingface_cache()
    
    # Test lightweight model
    print("\n=== Testing distilgpt2 (smallest model) ===")
    test_model_load("distilgpt2")
    
    # Test Facebook OPT small model
    print("\n=== Testing facebook/opt-125m ===")
    test_model_load("facebook/opt-125m")
