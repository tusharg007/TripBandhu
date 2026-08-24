"""
release_check.py — Automated local release readiness verification.
Orchestrates fast test suite, evaluation benchmarks, syntax compilation,
dependency validation, forbidden string checks, and secret hygiene.
"""

from __future__ import annotations

import os
import pathlib
import py_compile
import re
import subprocess
import sys

# Release checks must not export traces from a developer's local .env file.
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

_FORBIDDEN_TARGET = bytes.fromhex("61746c616e").decode("ascii")

_IGNORED_DIRECTORIES = {
    ".git",
    ".pytest_cache",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
}

_SENSITIVE_PATTERNS = [
    re.compile(r"gsk_[a-zA-Z0-9_\-]{28,}", re.IGNORECASE),
    re.compile(r"tvly-[a-zA-Z0-9_\-]{28,}", re.IGNORECASE),
]


def is_ignored_path(path: pathlib.Path, root: pathlib.Path) -> bool:
    """Return True for generated or third-party directories outside release scope."""
    relative_parts = path.relative_to(root).parts
    return any(part.lower() in _IGNORED_DIRECTORIES for part in relative_parts)


def run_step(step_name: str, fn) -> bool:
    print(f"[*] Running: {step_name}...", end=" ", flush=True)
    try:
        fn()
        print("[PASS]")
        return True
    except Exception as exc:
        print(f"[FAIL]\n    Error: {exc}")
        return False


def check_compilation():
    root = pathlib.Path(__file__).parent.parent
    for current_dir, directory_names, file_names in os.walk(root):
        directory_names[:] = [
            name for name in directory_names
            if name.lower() not in _IGNORED_DIRECTORIES
        ]
        for file_name in file_names:
            if file_name.endswith(".py"):
                py_compile.compile(
                    str(pathlib.Path(current_dir) / file_name),
                    doraise=True,
                )


def check_dependencies():
    res = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"pip check failed:\n{res.stdout}\n{res.stderr}")


def check_forbidden_name():
    root = pathlib.Path(__file__).parent.parent
    scanned_exts = {".py", ".md", ".json", ".html", ".css", ".js", ".yml", ".yaml", ".txt"}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in scanned_exts:
            if is_ignored_path(path, root) or path.name == "release_check.py":
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if _FORBIDDEN_TARGET in text.lower():
                raise RuntimeError(f"Forbidden target detected in {path}")


def check_secret_hygiene():
    root = pathlib.Path(__file__).parent.parent
    scanned_exts = {".py", ".md", ".json", ".html", ".css", ".js", ".yml", ".yaml"}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in scanned_exts:
            if is_ignored_path(path, root) or "eval_dataset.py" in path.parts or path.name.startswith("test_"):
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for pattern in _SENSITIVE_PATTERNS:
                # Ignore mock/dummy strings like gsk_test..., your_...
                matches = pattern.findall(text)
                real_matches = [m for m in matches if not any(dummy in m.lower() for dummy in ["test", "dummy", "example", "mock", "placeholder", "your_"])]
                if real_matches:
                    raise RuntimeError(f"Possible sensitive credential detected in {path}: {real_matches}")


def check_fast_tests():
    res = subprocess.run([sys.executable, "-m", "pytest", "-m", "not integration", "-q"], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Pytest fast suite failed:\n{res.stdout}\n{res.stderr}")


def check_fast_evaluator():
    res = subprocess.run([sys.executable, "-m", "eval.evaluator"], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"FAST evaluation benchmark failed:\n{res.stdout}\n{res.stderr}")


def main():
    print("==================================================")
    print("TRIPBANDHU RELEASE READINESS VERIFICATION")
    print("==================================================")
    steps = [
        ("Syntax & Bytecode Compilation", check_compilation),
        ("Python Dependency Health (pip check)", check_dependencies),
        ("Forbidden Name Scan", check_forbidden_name),
        ("Secret & Credential Hygiene", check_secret_hygiene),
        ("Fast Pytest Regression Suite", check_fast_tests),
        ("Deterministic Benchmark Evaluator", check_fast_evaluator),
    ]

    all_passed = True
    for name, fn in steps:
        if not run_step(name, fn):
            all_passed = False

    print("==================================================")
    if all_passed:
        print("[SUCCESS] All local release checks PASSED cleanly!")
        sys.exit(0)
    else:
        print("[FAILURE] One or more release checks failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
