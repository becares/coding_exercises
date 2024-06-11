#!/bin/bash
echo "Initializing fastapi and streamlit..."
conda activate coding-exercises
fastapi run src/api.py &
streamlit run src/ui.py &
wait