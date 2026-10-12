#!/bin/bash
. /opt/supervisor-scripts/utils/logging.sh ""
. /opt/supervisor-scripts/utils/environment.sh
set -eo pipefail
cd /workspace/data-expansion-20260910
pty /venv/main/bin/python -u run_pilot.py 2>&1 | tee generation.log
