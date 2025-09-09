#!/bin/bash

# Upgrade pip and build tools to make sure we use prebuilt wheels
pip install --upgrade pip setuptools wheel

# Prefer binaries to avoid building from source
pip install --prefer-binary -r requirements.txt

# Start Streamlit (adjust filename if your main file is not main.py)
streamlit run main.py --server.port $PORT --server.address 0.0.0.0
