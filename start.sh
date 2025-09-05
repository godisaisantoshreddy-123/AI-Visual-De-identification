#!/bin/bash
# Export PORT for Render
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
