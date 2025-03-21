#!/bin/bash

echo "Starting DocBot with PyTorch compatibility fix..."

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "Activated virtual environment"
fi

# Make sure torch_fix.py is executable
chmod +x torch_fix.py

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "⚠️ Streamlit not found! Installing dependencies first..."
    ./install_deps.sh
fi

# Run the app with our fix
echo "Starting application..."
python torch_fix.py src/app.py

# Capture exit code
exit_code=$?

if [ $exit_code -ne 0 ]; then
    echo "Application exited with error code $exit_code"
    echo "If you're seeing 'No module named streamlit.cli', try running:"
    echo "  pip install 'streamlit>=1.30.0'"
fi

exit $exit_code
