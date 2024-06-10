#!/bin/bash
echo "Initializing fastapi and streamlit..."
fastapi run main.py &
streamlit run ui.py &
wait