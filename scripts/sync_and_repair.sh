#!/bin/sh
set -eu

uv sync --group dev
./.venv/bin/python scripts/repair_hidden_pth.py
