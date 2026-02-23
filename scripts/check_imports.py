"""
Lightweight import validation for core project modules.

Usage:
    python scripts/check_imports.py
"""

from importlib import import_module
from pathlib import Path
import sys


MODULES = [
    "src.lens_models",
    "src.optics",
    "src.ml",
    "src.time_delay",
    "src.validation",
    "api.main",
    "database",
    "app.utils",
    "benchmarks",
]


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root))

    failed = []
    for module_name in MODULES:
        try:
            import_module(module_name)
            print(f"[OK] {module_name}")
        except Exception as exc:  # pragma: no cover - script-level diagnostic
            failed.append((module_name, exc))
            print(f"[FAIL] {module_name}: {type(exc).__name__}: {exc}")

    if failed:
        print(f"\nImport check failed: {len(failed)} module(s) not importable.")
        return 1

    print("\nImport check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
