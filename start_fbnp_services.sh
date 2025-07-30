#!/bin/bash
source /home/smithkt/FBN_publishing/FBNP_env/bin/activate
cd /home/smithkt/FBN_publishing

# Start FastAPI server in background
/home/smithkt/FBN_publishing/FBNP_env/bin/uvicorn fbnp_api:app --host 0.0.0.0 --port 8000 --reload --log-level debug &

# Start worker pool (waits)
python start_workers.py
