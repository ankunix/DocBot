"""
Information about small LLM models suitable for POC development
"""

SMALL_MODELS = {
    "OPT Models": {
        "facebook/opt-125m": {
            "parameters": "125M",
            "description": "Facebook's smallest OPT model, very fast but limited capabilities",
            "memory_req": "~500MB",
            "speed": "Very Fast"
        },
        "facebook/opt-350m": {
            "parameters": "350M",
            "description": "Slightly larger OPT model with better capabilities",
            "memory_req": "~1GB",
            "speed": "Fast"
        }
    },
    "GPT-2 Variants": {
        "distilgpt2": {
            "parameters": "82M",
            "description": "Distilled version of GPT-2, extremely lightweight",
            "memory_req": "~330MB",
            "speed": "Very Fast"
        },
        "gpt2": {
            "parameters": "124M", 
            "description": "Original GPT-2 small model",
            "memory_req": "~500MB",
            "speed": "Fast"
        }
    },
    "TinyLlama": {
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0": {
            "parameters": "1.1B",
            "description": "Small but capable Llama model fine-tuned for chat",
            "memory_req": "~2.5GB",
            "speed": "Medium"
        }
    },
    "Microsoft Phi": {
        "microsoft/phi-1_5": {
            "parameters": "1.3B",
            "description": "Small but powerful model with good reasoning capabilities",
            "memory_req": "~3GB",
            "speed": "Medium"
        }
    },
    "Falcon": {
        "tiiuae/falcon-rw-1b": {
            "parameters": "1B",
            "description": "1B parameter Falcon model with decent performance",
            "memory_req": "~2GB",
            "speed": "Medium"
        }
    }
}

def get_recommended_model_by_resource_constraint(memory_limit_gb=1.0):
    """Return recommended model given memory constraints in GB"""
    if memory_limit_gb < 0.5:
        return "distilgpt2"
    elif memory_limit_gb < 1.0:
        return "facebook/opt-125m"
    elif memory_limit_gb < 2.0:
        return "facebook/opt-350m"
    elif memory_limit_gb < 3.0:
        return "tiiuae/falcon-rw-1b"
    else:
        return "microsoft/phi-1_5"
