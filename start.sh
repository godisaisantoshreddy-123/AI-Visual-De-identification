#!/bin/bash
pip install --upgrade pip setuptools wheel
pip install --prefer-binary -r requirements.txt
python -m spacy download en_core_web_sm
streamlit run main.py --server.port $PORT --server.address 0.0.0.0
