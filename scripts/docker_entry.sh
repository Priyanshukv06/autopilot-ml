#!/bin/sh
# Entry point of the Docker image (optional task 35): the API in the background, the dashboard in front.
# Both listen on 0.0.0.0 inside the container; publish the ports on 127.0.0.1 on the host.
set -e
python scripts/run_api.py --host 0.0.0.0 --port 8000 &
exec streamlit run app/main.py --server.address 0.0.0.0 --server.port 8501 --server.headless true \
    --browser.gatherUsageStats false
