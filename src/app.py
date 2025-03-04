import streamlit as st
import pandas as pd
import os
import threading
import time
import traceback
from sec_edgar.config import DATA_DIR
from local_model import LocalLLMWrapper
from fallback_model import simple_keyword_response

# Load the processed filings data
data_path = os.path.join(DATA_DIR, "processed_filings.csv")
df_filings = pd.read_csv(data_path)

# Thread-safe model tracking (don't use session_state in threads)
MODEL_STATUS = {}  # 'loading', 'ready', 'error'
MODEL_ERRORS = {}
model_instances = {}
MODEL_LOCK = threading.Lock()  # Thread lock for safe access

# Model options with their sizes
MODEL_OPTIONS = {
    "OPT-125M (125M parameters, fastest)": "facebook/opt-125m",
    "DistilGPT2 (82M parameters, very fast)": "distilgpt2",
    "Phi-1.5 (1.3B parameters, good balance)": "microsoft/phi-1_5",
    "TinyLlama (1.1B parameters, good quality)": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
}

# Get or create a model instance
def get_model(model_name):
    with MODEL_LOCK:
        if model_name not in model_instances:
            model_instances[model_name] = LocalLLMWrapper(model_name=model_name)
        return model_instances[model_name]

# Background initialization thread that doesn't use st.session_state
def initialize_model_in_background(model_name):
    try:
        with MODEL_LOCK:
            MODEL_STATUS[model_name] = 'loading'
        
        model = get_model(model_name)
        model.initialize()
        
        with MODEL_LOCK:
            MODEL_STATUS[model_name] = 'ready'
        
        print(f"Model {model_name} initialized successfully")
    except Exception as e:
        error_msg = f"Error initializing model: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        
        with MODEL_LOCK:
            MODEL_STATUS[model_name] = 'error'
            MODEL_ERRORS[model_name] = error_msg

# Track currently selected model
if 'active_model_key' not in st.session_state:
    st.session_state.active_model_key = list(MODEL_OPTIONS.values())[0]

# Start default model loading on first run
if 'app_initialized' not in st.session_state:
    st.session_state.app_initialized = True
    default_model = list(MODEL_OPTIONS.values())[0]
    threading.Thread(
        target=initialize_model_in_background,
        args=(default_model,),
        daemon=True
    ).start()

# Sidebar for model selection
with st.sidebar:
    st.title("Model Settings")
    selected_model = st.selectbox(
        "Select LLM Model:",
        list(MODEL_OPTIONS.keys()),
        index=0
    )
    model_key = MODEL_OPTIONS[selected_model]
    
    # Update active model when selection changes
    if st.session_state.active_model_key != model_key:
        st.session_state.active_model_key = model_key
    
    # Start model loading if not already started
    if model_key not in MODEL_STATUS:
        threading.Thread(
            target=initialize_model_in_background,
            args=(model_key,),
            daemon=True
        ).start()
        st.info(f"Starting initialization for {selected_model}...")
    
    # Display model status
    status = MODEL_STATUS.get(model_key, 'not started')
    if status == 'ready':
        st.success(f"Model {selected_model} is ready")
    elif status == 'loading':
        st.info(f"Model {selected_model} is loading...")
    elif status == 'error':
        st.error(f"Error loading model {selected_model}")
        with st.expander("Show Error Details"):
            st.code(MODEL_ERRORS.get(model_key, "Unknown error"))
            if st.button("Retry Loading"):
                threading.Thread(
                    target=initialize_model_in_background,
                    args=(model_key,),
                    daemon=True
                ).start()

def generate_response(query, df):
    model_key = st.session_state.active_model_key
    
    # Check if model is ready
    if MODEL_STATUS.get(model_key) != 'ready':
        if MODEL_STATUS.get(model_key) == 'error':
            return f"❌ Error: Model failed to initialize. Please check error details in sidebar or try another model."
        else:
            return f"⏳ Model is still initializing, please wait. Your query will be processed once the model is ready."
    
    model = get_model(model_key)
    
    # Retrieve relevant context from dataframe based on query
    context = " ".join(df['Text'].tolist())
    if len(context) > 4000:  # Limit context for smaller models
        context = context[:4000] + "..."
    
    try:
        # Generate response using the initialized model
        response = model.chat_completion(
            messages=[
                {"role": "system", "content": f"You are an assistant that answers questions based on the following context: {context}"},
                {"role": "user", "content": query}
            ]
        )
        
        if isinstance(response, dict):
            if 'choices' in response and len(response['choices']) > 0:
                return response['choices'][0].get('message', {}).get('content', str(response))
            else:
                return str(response)
        else:
            return str(response)
            
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error generating response: {str(e)}\n{error_details}")
        
        # Fall back to the simple keyword response
        fallback_response = simple_keyword_response(query, context)
        return f"""
        **Note: Error with the model, using fallback method**
        
        {fallback_response}
        
        *Error details: {str(e)}*
        """

# Streamlit app
st.title("RAG-based Application using Smaller LLM Models")
st.write(f"Currently using: **{selected_model}**")
st.write("Enter your query below to get responses based on the processed filings data.")

# Query input with a submit button for better control
query = st.text_input("Enter your query:")
submit_button = st.button("Submit")

# Response area with clear status
response_container = st.container()

if submit_button and query:
    with response_container:
        st.markdown("### Response:")
        
        # Check model status in real-time when generating response
        if MODEL_STATUS.get(st.session_state.active_model_key) != 'ready':
            with st.spinner("Model is initializing... Please wait."):
                # Brief polling to see if model becomes ready while user is waiting
                for _ in range(10):
                    if MODEL_STATUS.get(st.session_state.active_model_key) == 'ready':
                        break
                    time.sleep(1)
        
        # Generate and display the response
        response = generate_response(query, df_filings)
        st.markdown(response)

# Status display at the bottom
st.write("---")
st.write("### Model Status")

# Display current status of all models
for name, key in MODEL_OPTIONS.items():
    status = MODEL_STATUS.get(key, "not started")
    if status == 'ready':
        st.success(f"✓ {name}: Ready")
    elif status == 'loading':
        st.info(f"⏳ {name}: Loading...")
    elif status == 'error':
        st.error(f"❌ {name}: Error")
    else:
        st.write(f"○ {name}: Not loaded")
