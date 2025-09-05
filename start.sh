#!/bin/bash
export STREAMLIT_CONFIG_DIR=/app/.streamlit
export STREAMLIT_HOME=/app
streamlit run app.py --server.port=8501 --server.address=0.0.0.0
