"""Production WSGI entrypoint for Ghent RE Intelligence (Issue #24).

Imports `torch` FIRST before any LightGBM/CatBoost module to guarantee OpenMP safety
on Apple Silicon, serves both `/api/*` and the built React SPA (`frontend/dist`) from
one process via Waitress (single-process multi-threaded — MPS-safe, zero fork crashes).

Usage:
    .venv-api/bin/python -m api.wsgi
    GHENT_HOST=0.0.0.0 GHENT_PORT=5001 GHENT_THREADS=4 .venv-api/bin/python -m api.wsgi
"""
from __future__ import annotations

import os
import sys

# CRITICAL on macOS Apple Silicon: import torch BEFORE lightgbm/catboost!
try:
    import torch  # noqa: F401
except Exception:
    pass

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from api.app import create_app  # noqa: E402

application = create_app()


def main() -> None:
    host = os.environ.get("GHENT_HOST", "127.0.0.1")
    port = int(os.environ.get("GHENT_PORT", "5001"))
    threads = int(os.environ.get("GHENT_THREADS", "4"))

    try:
        from waitress import serve
        print(f"Serving Ghent RE Intelligence (Waitress WSGI, {threads} threads) on http://{host}:{port}", flush=True)
        serve(application, host=host, port=port, threads=threads)
    except ImportError:
        print("waitress not installed; falling back to Flask threaded server", flush=True)
        application.run(host=host, port=port, debug=False, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
