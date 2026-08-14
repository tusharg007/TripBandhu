"""
release_check.py — Automated local release readiness verification.
Orchestrates fast test suite, evaluation benchmarks, syntax compilation,
dependency validation, forbidden string checks, and secret hygiene.
"""

from __future__ import annotations

import compileall
import os
import pathlib
import re
import subprocess
import sys

_FORBIDDEN_TARGET = bytes.fromhex("61746c616e").decode("ascii")

_SENSITIVE_PATTERNS = [
    re.compile(r"gsk_[a-zA-Z0-9_\-]{28,}", re.IGNORECASE),
    re.compile(r"tvly-[a-zA-Z0-9_\-]{28,}", re.IGNORECASE),
]


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
    success = compileall.compile_dir(str(root), quiet=1, force=False)
    if not success:
        raise RuntimeError("Bytecode compilation failed on some files.")


def check_dependencies():
    res = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"pip check failed:\n{res.stdout}\n{res.stderr}")


def check_forbidden_name():
    root = pathlib.Path(__file__).parent.parent
    scanned_exts = {".py", ".md", ".json", ".html", ".css", ".js", ".yml", ".yaml", ".txt"}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in scanned_exts:
            if ".git" in path.parts or ".pytest_cache" in path.parts or "node_modules" in path.parts or path.name == "release_check.py":
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if _FORBIDDEN_TARGET in text.lower():
                raise RuntimeError(f"Forbidden target detected in {path}")


def check_secret_hygiene():
    root = pathlib.Path(__file__).parent.parent
    scanned_exts = {".py", ".md", ".json", ".html", ".css", ".js", ".yml", ".yaml"}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in scanned_exts:
            if ".git" in path.parts or ".pytest_cache" in path.parts or "node_modules" in path.parts or "eval_dataset.py" in path.parts or path.name.startswith("test_"):
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
