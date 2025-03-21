"""
Wrapper for local language models
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os
from pathlib import Path
import concurrent.futures

# Global cache for models and tokenizers
MODEL_CACHE = {}
TOKENIZER_CACHE = {}
GENERATOR_CACHE = {}

# Define a cache directory
CACHE_DIR = Path(os.path.expanduser("~/.docbot_model_cache"))
CACHE_DIR.mkdir(exist_ok=True, parents=True)

class LocalLLMWrapper:
    def __init__(self, model_name="facebook/opt-125m", device="cpu"):
        """
        Initialize a local LLM model.
        
        Args:
            model_name: Hugging Face model name/path
            device: "cpu" or "cuda" for GPU support
        """
        self.model_name = model_name
        self.device = device
        self.tokenizer = None
        self.model = None
        self.generator = None
        self.is_initialized = False
        self.cache_key = f"{model_name}_{device}"
        
    def initialize(self, timeout=60):
        """Load model on first use from cache if available, with timeout"""
        if self.is_initialized:
            return
            
        try:
            # Check if model is already in cache
            if self.cache_key in MODEL_CACHE:
                print(f"Loading model {self.model_name} from memory cache...")
                self.model = MODEL_CACHE[self.cache_key]
                self.tokenizer = TOKENIZER_CACHE[self.cache_key]
                self.generator = GENERATOR_CACHE[self.cache_key]
                self.is_initialized = True
                print(f"Model loaded successfully from memory cache")
                return
                
            print(f"Loading model {self.model_name}...")
            
            # Use the cache directory for downloads
            def load_tokenizer():
                return AutoTokenizer.from_pretrained(
                    self.model_name,
                    cache_dir=CACHE_DIR
                )
                
            def load_model():
                return AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    low_cpu_mem_usage=True,
                    cache_dir=CACHE_DIR
                )
            
            # Load tokenizer and model with timeout
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_tokenizer = executor.submit(load_tokenizer)
                self.tokenizer = future_tokenizer.result(timeout=timeout)
                TOKENIZER_CACHE[self.cache_key] = self.tokenizer
                
                future_model = executor.submit(load_model)
                self.model = future_model.result(timeout=timeout)
                MODEL_CACHE[self.cache_key] = self.model
            
            if self.device == "cuda" and torch.cuda.is_available():
                self.model = self.model.to("cuda")
                
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" and torch.cuda.is_available() else -1
            )
            GENERATOR_CACHE[self.cache_key] = self.generator
            
            self.is_initialized = True
            print(f"Model loaded successfully on {self.device}")
            
        except concurrent.futures.TimeoutError:
            print(f"Timeout: Model loading took longer than {timeout} seconds")
            raise TimeoutError(f"Loading model {self.model_name} timed out after {timeout} seconds")
        except Exception as e:
            print(f"Error initializing model: {str(e)}")
            raise e
    
    def _format_prompt(self, messages):
        """Format messages according to model expectations"""
        # Default format for generic models
        prompt = ""
        
        # Check if it's a Llama-like model
        if "llama" in self.model_name.lower():
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "system":
                    prompt += f"<|system|>\n{content}\n"
                elif role == "user":
                    prompt += f"<|user|>\n{content}\n"
                elif role == "assistant":
                    prompt += f"<|assistant|>\n{content}\n"
            prompt += "<|assistant|>\n"
        
        # For OPT, Phi, and other basic models
        else:
            system_msg = ""
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "system":
                    system_msg = content
                elif role == "user":
                    # Add system message as context instruction
                    if system_msg:
                        prompt += f"Instructions: {system_msg}\n\n"
                    prompt += f"Question: {content}\n\nAnswer: "
                elif role == "assistant":
                    prompt += f"{content}\n"
        
        return prompt
    
    def chat_completion(self, messages):
        """Match the DeepSeek API interface"""
        self.initialize()
        
        # Convert messages to appropriate prompt format
        prompt = self._format_prompt(messages)
        
        # Generate response
        try:
            outputs = self.generator(
                prompt, 
                max_new_tokens=256,  # Reduced for smaller models
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            generated_text = outputs[0]["generated_text"]
            
            # Extract only the assistant's response
            if "llama" in self.model_name.lower():
                response_text = generated_text.split("<|assistant|>\n")[-1].strip()
            else:
                response_text = generated_text[len(prompt):].strip()
            
            # Format to match DeepSeek's response structure
            response = {
                "choices": [
                    {
                        "message": {
                            "content": response_text,
                            "role": "assistant"
                        }
                    }
                ]
            }
            
            return response
            
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return {"error": str(e)}
