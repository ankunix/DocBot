"""
Simplified version of the app with minimal dependencies
"""
import streamlit as st
import pandas as pd
import os
import time
from local_model import LocalLLMWrapper

# Initialize session state variables
if "models" not in st.session_state:
    st.session_state.models = {}
if "model_status" not in st.session_state:
    st.session_state.model_status = {}

# Available model options
MODEL_OPTIONS = {
    "DistilGPT2 (smallest)": "distilgpt2",
    "OPT-125M (small)": "facebook/opt-125m",
}

# Initialize the app
st.title("Simple LLM Demo")
st.write("This is a simplified version to test model loading")

# Model selection
selected_model_name = st.selectbox("Select a model:", list(MODEL_OPTIONS.keys()))
selected_model_id = MODEL_OPTIONS[selected_model_name]

# Check if model exists in session state
if selected_model_id not in st.session_state.models:
    st.session_state.models[selected_model_id] = LocalLLMWrapper(model_name=selected_model_id)
    st.session_state.model_status[selected_model_id] = "not_loaded"

# Load model button
if st.button(f"Load {selected_model_name}"):
    try:
        with st.spinner(f"Loading {selected_model_name}..."):
            st.session_state.model_status[selected_model_id] = "loading"
            st.session_state.models[selected_model_id].initialize()
            st.session_state.model_status[selected_model_id] = "ready"
        st.success(f"Model {selected_model_name} loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.session_state.model_status[selected_model_id] = "error"

# Input for testing
query = st.text_input("Enter a test query:")
if st.button("Generate Response") and query:
    model = st.session_state.models[selected_model_id]
    
    # Check if model is loaded
    if not model.is_initialized:
        st.warning("Model is not loaded yet. Please load the model first.")
    else:
        with st.spinner("Generating response..."):
            try:
                response = model.chat_completion(
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": query}
                    ]
                )
                
                if 'choices' in response and len(response['choices']) > 0:
                    st.write(response['choices'][0]['message']['content'])
                else:
                    st.write(str(response))
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")

# Display model status
st.write("---")
st.write("### Model Status")
for name, model_id in MODEL_OPTIONS.items():
    status = st.session_state.model_status.get(model_id, "not_loaded")
    if status == "ready":
        st.success(f"{name}: Loaded ✓")
    elif status == "loading":
        st.info(f"{name}: Loading...")
    elif status == "error":
        st.error(f"{name}: Error loading ✗")
    else:
        st.write(f"{name}: Not loaded")

# System info
st.write("---")
st.write("### System Information")
st.code(f"""
Python Version: {os.sys.version.split()[0]}
Working Directory: {os.getcwd()}
""")
