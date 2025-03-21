import streamlit as st
import pandas as pd
import os
import threading
import time
import traceback
import warnings
import gc

# Suppress warnings to avoid console spam
warnings.filterwarnings('ignore')

# Check for memory leaks
try:
    import psutil
    def get_memory_usage():
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return f"{memory_info.rss / 1024 / 1024:.1f} MB"
except ImportError:
    def get_memory_usage():
        return "psutil not installed"

# Import the rest of the modules
try:
    from sec_edgar.config import DATA_DIR
    from local_model import LocalLLMWrapper
    from fallback_model import simple_keyword_response
    from simple_rag import SimpleRAG
except Exception as e:
    st.error(f"Error loading modules: {str(e)}")
    st.stop()

# Configure the page to optimize performance
st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load the processed filings data
try:
    data_path = os.path.join(DATA_DIR, "processed_filings.csv")
    df_filings = pd.read_csv(data_path)
    print(f"Loaded data with {len(df_filings)} rows")
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    df_filings = pd.DataFrame(columns=["Text", "Source"])

# Thread-safe model tracking (don't use session_state in threads)
MODEL_STATUS = {}  # 'loading', 'ready', 'error'
MODEL_ERRORS = {}
model_instances = {}
MODEL_LOCK = threading.Lock()  # Thread lock for safe access

# Model options with their sizes - limiting to smaller models to avoid memory issues
MODEL_OPTIONS = {
    "DistilGPT2 (82M parameters, very fast)": "distilgpt2",
    "OPT-125M (125M parameters, fastest)": "facebook/opt-125m",
    "Phi-1.5 (1.3B parameters, good balance)": "microsoft/phi-1_5",
}

# Initialize RAG component
rag_system = None
RAG_STATUS = "not_started"  # 'not_started', 'loading', 'ready', 'error'
RAG_ERROR = None

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

# Initialize RAG system in background
def initialize_rag_in_background():
    global rag_system, RAG_STATUS, RAG_ERROR
    try:
        RAG_STATUS = 'loading'
        
        # Create a new RAG system instance to avoid memory issues
        rag_system = SimpleRAG()
        
        # Make a copy of the dataframe to prevent memory leaks
        df_copy = df_filings.copy()
        
        # Load only required columns to reduce memory
        if 'Text' in df_copy.columns:
            if len(df_copy) > 1000:  # If the dataset is large, sample it
                print(f"Sampling data from {len(df_copy)} rows to 1000 rows for memory efficiency")
                df_copy = df_copy.sample(1000, random_state=42)
            
            # Clean up memory before intensive operations
            gc.collect()
            
            # Load data into RAG
            rag_system.load_data(df_copy)
            RAG_STATUS = 'ready'
            print("RAG system initialized successfully")
        else:
            RAG_ERROR = "DataFrame does not have 'Text' column"
            RAG_STATUS = 'error'
    except Exception as e:
        error_msg = f"Error initializing RAG: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        RAG_STATUS = 'error'
        RAG_ERROR = error_msg
        
        # Clean up on error
        del rag_system
        gc.collect()

# Track currently selected model
if 'active_model_key' not in st.session_state:
    st.session_state.active_model_key = list(MODEL_OPTIONS.values())[0]

# Start default model loading and RAG initialization on first run
if 'app_initialized' not in st.session_state:
    st.session_state.app_initialized = True
    default_model = list(MODEL_OPTIONS.values())[0]
    
    # Stagger thread initialization to prevent memory spikes
    threading.Thread(
        target=initialize_model_in_background,
        args=(default_model,),
        daemon=True
    ).start()
    
    # Start RAG initialization after a brief delay
    def delayed_rag_init():
        time.sleep(5)  # Wait 5 seconds to let model initialization start
        threading.Thread(
            target=initialize_rag_in_background,
            daemon=True
        ).start()
    
    threading.Thread(target=delayed_rag_init, daemon=True).start()

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
    
    # Display RAG status
    st.title("RAG System Status")
    if RAG_STATUS == 'ready':
        st.success(f"RAG system is ready")
    elif RAG_STATUS == 'loading':
        st.info(f"Building RAG system... This may take a moment.")
    elif RAG_STATUS == 'error':
        st.error(f"Error creating RAG system")
        with st.expander("Show Error Details"):
            st.code(RAG_ERROR)
            if st.button("Retry RAG"):
                threading.Thread(
                    target=initialize_rag_in_background,
                    daemon=True
                ).start()
    
    # RAG parameters
    st.title("RAG Settings")
    max_chunks = st.slider("Max Context Documents", 1, 5, 3)
    show_sources = st.checkbox("Show Sources", True)
    
    st.title("Memory Management")
    st.text(f"Current memory usage: {get_memory_usage()}")
    
    if st.button("Clear Memory Cache"):
        import gc
        st.session_state.clear()
        with MODEL_LOCK:
            for model_name in model_instances:
                try:
                    del model_instances[model_name]
                except:
                    pass
            model_instances = {}
            MODEL_STATUS = {}
            MODEL_ERRORS = {}
        gc.collect()
        st.success("Memory cache cleared")
        st.experimental_rerun()

def generate_response(query, df):
    model_key = st.session_state.active_model_key
    
    # Check if model is ready
    if MODEL_STATUS.get(model_key) != 'ready':
        if MODEL_STATUS.get(model_key) == 'error':
            return f"❌ Error: Model failed to initialize. Please check error details in sidebar or try another model."
        else:
            return f"⏳ Model is still initializing, please wait. Your query will be processed once the model is ready."
    
    # Check if RAG is ready
    if RAG_STATUS != 'ready':
        if RAG_STATUS == 'error':
            return f"❌ Error: RAG system failed to initialize. Please check error details in sidebar."
        else:
            return f"⏳ RAG system is still being created, please wait."
    
    model = get_model(model_key)
    
    try:
        # Retrieve relevant context using RAG
        context, sources = rag_system.get_relevant_context(
            query, 
            max_docs=st.session_state.get('max_chunks', 3)
        )
        
        # Generate response using the initialized model
        response = model.chat_completion(
            messages=[
                {"role": "system", "content": f"You are an assistant that answers questions based on the following context. Only use information from the provided context. If you don't know the answer, say so.\n\nCONTEXT:\n{context}"},
                {"role": "user", "content": query}
            ]
        )
        
        # Format the response
        if isinstance(response, dict):
            if 'choices' in response and len(response['choices']) > 0:
                result = response['choices'][0].get('message', {}).get('content', str(response))
            else:
                result = str(response)
        else:
            result = str(response)
        
        # Add sources if requested
        if st.session_state.get('show_sources', True) and sources:
            result += "\n\n**Sources:**\n"
            for i, source in enumerate(sources):
                result += f"- {source}\n"
                
        return result
            
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error generating response: {str(e)}\n{error_details}")
        
        # Fall back to the simple keyword response
        fallback_response = simple_keyword_response(query, " ".join(df['Text'].tolist()[:1000]))
        return f"""
        **Note: Error with the model, using fallback method**
        
        {fallback_response}
        
        *Error details: {str(e)}*
        """

# Store sidebar settings in session state for access during response generation
if 'max_chunks' not in st.session_state:
    st.session_state.max_chunks = max_chunks
else:
    st.session_state.max_chunks = max_chunks

if 'show_sources' not in st.session_state:
    st.session_state.show_sources = show_sources
else:
    st.session_state.show_sources = show_sources

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
        
        with st.spinner("Generating response..."):
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
