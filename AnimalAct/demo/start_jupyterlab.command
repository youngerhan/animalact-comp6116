#!/bin/zsh
set -euo pipefail
cd -- "${0:A:h}"

# Use the activated environment, or an explicitly selected Python executable.
if [[ -n "${ANIMALACT_PYTHON:-}" ]]; then
  python_bin="$ANIMALACT_PYTHON"
elif command -v python >/dev/null 2>&1; then
  python_bin="$(command -v python)"
elif command -v python3 >/dev/null 2>&1; then
  python_bin="$(command -v python3)"
else
  print -u2 "Python was not found. Activate your Python environment and try again."
  exit 1
fi

if ! "$python_bin" -c 'import jupyterlab' >/dev/null 2>&1; then
  print -u2 "JupyterLab is unavailable in the selected Python environment."
  print -u2 "Activate the intended environment and run: python -m pip install -r requirements.txt"
  exit 1
fi
exec "$python_bin" -m jupyterlab --ServerApp.ip=127.0.0.1 --ServerApp.root_dir="$PWD" --LabApp.default_url=/lab/tree/animal_action_baseline.ipynb "$@"
