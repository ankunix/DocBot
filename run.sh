#!/bin/bash

echo "Starting DocBot RAG application..."
echo "Setting up environment..."

# Setup Python environment
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Reduce warnings
export PYTHONWARNINGS=ignore
export TOKENIZERS_PARALLELISM=false

# Run the launcher
echo "Starting application..."
python launcher.py

# Handle clean exit
echo "Application closed."
