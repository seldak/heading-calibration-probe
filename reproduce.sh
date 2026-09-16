#!/bin/sh
set -eu

cd -- "$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)"
heading_python=${HEADING_PYTHON:-python3}
requested_version=$("$heading_python" -c '
import sys

version = sys.version_info[:2]
if not (3, 12) <= version <= (3, 14):
    raise SystemExit(
        f"Python 3.12 through 3.14 is required; found {version[0]}.{version[1]}"
    )
print(f"{version[0]}.{version[1]}")
')
if [ ! -d .venv ]; then
    "$heading_python" -m venv .venv
elif [ ! -x .venv/bin/python ]; then
    echo "Existing .venv does not contain an executable Python interpreter" >&2
    exit 1
else
    venv_version=$(.venv/bin/python -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
    if [ "$venv_version" != "$requested_version" ]; then
        echo "Existing .venv uses Python $venv_version; selected interpreter uses Python $requested_version" >&2
        echo "Remove .venv to recreate it, or select its Python version with HEADING_PYTHON" >&2
        exit 1
    fi
fi
.venv/bin/python -m pip install --disable-pip-version-check -r requirements.txt
.venv/bin/python heading_probe.py
