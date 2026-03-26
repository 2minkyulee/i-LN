#!/bin/bash
set -euo pipefail

# Replace any preinstalled BasicSR with the local editable copy.
python -m pip uninstall -y basicsr || true
python -m pip install -r requirements.txt
python setup.py develop

# Optional system convenience package for remote sessions.
apt-get update
apt-get install -y htop
