#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${1:-${SCRIPT_DIR}/config.json}"
PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" python -m opsigen preprocess --config "${CONFIG_PATH}"
