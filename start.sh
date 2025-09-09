#!/bin/bash

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Start Streamlit app
streamlit run main.py --server.port $PORT --server.address 0.0.0.0
