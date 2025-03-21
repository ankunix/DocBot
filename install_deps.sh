#!/bin/bash

echo "Installing DocBot dependencies..."

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "Activated virtual environment"
else
    echo "Creating virtual environment..."
    python -m venv .venv
    source .venv/bin/activate
    echo "Activated virtual environment"
fi

# Install base requirements
echo "Installing base requirements..."
pip install -r requirements.txt

# Install Streamlit with extras - properly escaped for zsh
echo "Installing Streamlit with extras..."
pip install "streamlit[extras]"

# Install psutil for memory management
pip install psutil

echo "All dependencies installed successfully!"
echo "You can now run the application with ./start_app.sh"
