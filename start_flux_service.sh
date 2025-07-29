#!/bin/bash
source /home/smithkt/FBN_Publishing/FBNP_env/bin/activate
cd /home/smithkt/FBN_Publishing

# Start FastAPI server in background
/home/smithkt/FBN_Publishing/FBNP_env/bin/uvicorn flux_api:app --host 0.0.0.0 --port 8000 &

# Start worker pool (waits)
python start_workers.py
