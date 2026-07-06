#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" python "${SCRIPT_DIR}/calculate_one_rhodopsin.py" "$@"
