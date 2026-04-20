#!/usr/bin/env python3
"""Repair the project virtualenv after `uv sync`.

On some macOS setups, `uv sync` leaves `.pth` files in `.venv` with the
`hidden` file flag. CPython's `site.py` skips hidden `.pth` files, which breaks
editable installs and other site-packages path injections.
"""

from __future__ import annotations

import os
import stat
import sys
import textwrap
from pathlib import Path


def iter_pth_files(venv_path: Path) -> list[Path]:
    return sorted(venv_path.glob("lib/python*/site-packages/*.pth"))


def find_site_packages(venv_path: Path) -> Path:
    matches = sorted(venv_path.glob("lib/python*/site-packages"))
    if not matches:
        raise FileNotFoundError(f"No site-packages directory found under {venv_path}")
    return matches[0]


def clear_hidden_flag(path: Path) -> bool:
    flags = getattr(path.stat(), "st_flags", 0)
    hidden_flag = getattr(stat, "UF_HIDDEN", 0)
    if not hidden_flag or not (flags & hidden_flag):
        return False
    os.chflags(path, flags & ~hidden_flag)
    return True


def patch_entrypoint(repo_root: Path) -> bool:
    entrypoint = repo_root / ".venv/bin/ai-ocr-pipeline"
    if not entrypoint.exists():
        return False

    text = entrypoint.read_text(encoding="utf-8")
    marker = 'src_root = repo_root / "src"'
    if marker in text:
        return False

    old = "import sys\nfrom ai_ocr_pipeline.cli import app\n"
    new = (
        "import sys\n"
        "from pathlib import Path\n"
        "repo_root = Path(__file__).resolve().parents[2]\n"
        'src_root = repo_root / "src"\n'
        "if str(src_root) not in sys.path:\n"
        "    sys.path.insert(0, str(src_root))\n"
        "from ai_ocr_pipeline.cli import app\n"
    )
    if old not in text:
        return False

    entrypoint.write_text(text.replace(old, new), encoding="utf-8")
    return True


def ensure_sitecustomize(repo_root: Path) -> bool:
    sitecustomize_path = find_site_packages(repo_root / ".venv") / "sitecustomize.py"
    content = textwrap.dedent(
        """
        from __future__ import annotations

        import sys
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[4]
        src_root = repo_root / "src"
        if src_root.exists() and str(src_root) not in sys.path:
            sys.path.insert(0, str(src_root))
        """
    ).lstrip()
    current = sitecustomize_path.read_text(encoding="utf-8") if sitecustomize_path.exists() else None
    if current == content:
        return False
    sitecustomize_path.write_text(content, encoding="utf-8")
    return True


def ensure_reading_order_utils_modules(repo_root: Path) -> list[str]:
    utils_dir = find_site_packages(repo_root / ".venv") / "reading_order/utils"
    utils_dir.mkdir(parents=True, exist_ok=True)

    modules = {
        "logger.py": textwrap.dedent(
            """
            from __future__ import annotations

            import logging


            def get_logger(name: str) -> logging.Logger:
                logger = logging.getLogger(name)
                if not logging.getLogger().handlers:
                    logging.basicConfig(level=logging.INFO)
                return logger
            """
        ).lstrip(),
        "time.py": textwrap.dedent(
            """
            from __future__ import annotations

            import time
            from contextlib import contextmanager


            class TimeKeeper:
                def __init__(self) -> None:
                    self.elapsed: dict[str, float] = {}

                @contextmanager
                def measure_time(self, label: str):
                    start = time.perf_counter()
                    try:
                        yield
                    finally:
                        self.elapsed[label] = self.elapsed.get(label, 0.0) + (
                            time.perf_counter() - start
                        )
            """
        ).lstrip(),
        "xml.py": textwrap.dedent(
            """
            from __future__ import annotations

            from collections import Counter


            def _count_tags(root) -> Counter[str]:
                counts: Counter[str] = Counter()
                for element in root.iter():
                    counts[element.tag] += 1
                return counts


            def insert_before(parent, new_child, reference_child) -> None:
                children = list(parent)
                try:
                    index = children.index(reference_child)
                except ValueError:
                    parent.append(new_child)
                    return
                parent.insert(index, new_child)


            class IndexedTags:
                def __init__(self, root, key: str = "__tmp_idx") -> None:
                    self.root = root
                    self.key = key
                    self._indexed = []

                def __enter__(self):
                    for index, element in enumerate(self.root.iter()):
                        element.set(self.key, str(index))
                        self._indexed.append(element)
                    return self

                def __exit__(self, exc_type, exc, tb) -> bool:
                    for element in self._indexed:
                        element.attrib.pop(self.key, None)
                    return False


            class ConstantNumberOfTags:
                \"\"\"Compatibility shim for the packaged `reading_order` module.\"\"\"

                def __init__(self, root) -> None:
                    self.root = root
                    self._before: Counter[str] | None = None

                def __enter__(self):
                    self._before = _count_tags(self.root)
                    return self

                def __exit__(self, exc_type, exc, tb) -> bool:
                    if exc_type is not None or self._before is None:
                        return False
                    after = _count_tags(self.root)
                    if after != self._before:
                        raise ValueError("The number of XML tags changed unexpectedly.")
                    return False
            """
        ).lstrip(),
    }

    updated = []
    for name, content in modules.items():
        path = utils_dir / name
        current = path.read_text(encoding="utf-8") if path.exists() else None
        if current == content:
            continue
        path.write_text(content, encoding="utf-8")
        updated.append(f"reading_order/utils/{name}")
    return updated


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    venv_path = repo_root / ".venv"
    if not venv_path.exists():
        print(f"No virtualenv found at {venv_path}", file=sys.stderr)
        return 1

    cleared = []
    for path in iter_pth_files(venv_path):
        if clear_hidden_flag(path):
            cleared.append(path)

    if cleared:
        print("Cleared hidden flags from:")
        for path in cleared:
            print(f"  {path.relative_to(repo_root)}")
    else:
        print("No hidden `.pth` files found.")

    if patch_entrypoint(repo_root):
        print("Patched `.venv/bin/ai-ocr-pipeline` with a `src/` fallback import path.")
    else:
        print("No entrypoint patch needed.")
    if ensure_sitecustomize(repo_root):
        print("Installed `.venv/.../sitecustomize.py` to add `src/` without relying on `.pth`.")
    else:
        print("`sitecustomize.py` already present.")
    added = ensure_reading_order_utils_modules(repo_root)
    if added:
        print("Added compatibility modules:")
        for path in added:
            print(f"  {path}")
    else:
        print("`reading_order.utils` compatibility modules already present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
