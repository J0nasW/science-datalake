#!/usr/bin/env bash
# Regenerate DuckDB views after mounting on a new workstation.
# Run this from the data lake root directory.
#
# Usage:
#   cd /mnt/nvme03/science_datalake
#   ./remount.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${SCRIPT_DIR}/.venv"

if [ ! -d "$VENV" ]; then
    echo "Error: .venv not found. Run: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
    exit 1
fi

echo "Regenerating DuckDB views for: ${SCRIPT_DIR}"
"${VENV}/bin/python" "${SCRIPT_DIR}/scripts/create_unified_db.py"
echo "Done. Data lake is ready."
