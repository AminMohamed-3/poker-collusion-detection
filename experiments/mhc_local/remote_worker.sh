#!/bin/bash
. /opt/supervisor-scripts/utils/logging.sh ""
. /opt/supervisor-scripts/utils/environment.sh
set -eo pipefail
cd /workspace/mhc-20260910
pty /venv/main/bin/python -u remote_start.py 2>&1 | tee worker.log
