#!/bin/sh
set -eu

cd -- "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
heading_python=${HEADING_PYTHON:-python3}
"$heading_python" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Use Python 3.12 for the recorded dependency pins"'
if [ ! -d .venv ]; then
    "$heading_python" -m venv .venv
fi
.venv/bin/python -m pip install --disable-pip-version-check -r requirements.txt
.venv/bin/python heading_probe.py
