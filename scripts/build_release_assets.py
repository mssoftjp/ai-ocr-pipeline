from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(*args: str) -> None:
    print("+", " ".join(args))
    subprocess.run(args, cwd=ROOT, check=True)


def build_python_dist() -> None:
    _run(sys.executable, "-m", "build")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Python release assets.")
    parser.add_argument("target", choices=("python-dist",))
    args = parser.parse_args()

    if args.target == "python-dist":
        build_python_dist()


if __name__ == "__main__":
    main()
