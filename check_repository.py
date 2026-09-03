"""
Basic repository check script.

Run from the repository root:

    python check_repository.py
"""

import importlib
import py_compile
from pathlib import Path

REQUIRED_FILES = [
    "helper.py",
    "segmentation.py",
    "features_registry.py",
    "features.py",
]

for file_name in REQUIRED_FILES:
    path = Path(file_name)
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {file_name}")
    py_compile.compile(str(path), doraise=True)
    print(f"Syntax OK: {file_name}")

for module_name in [
    "helper",
    "segmentation",
    "features_registry",
    "features",
]:
    importlib.import_module(module_name)
    print(f"Import OK: {module_name}")

from features_registry import FEATURE_REGISTRY

columns = [info["column"] for info in FEATURE_REGISTRY.values()]
duplicates = sorted({col for col in columns if columns.count(col) > 1})

if duplicates:
    raise ValueError(f"Duplicated feature columns: {duplicates}")

print(f"Feature registry OK: {len(FEATURE_REGISTRY)} features")
print("Repository check completed successfully.")
