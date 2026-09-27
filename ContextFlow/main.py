# main.py
"""Backward-compatible demo entry point.

The full 16-scenario demonstration now lives in the installable package
as ``contextflow.demo`` (also available via the ``contextflow-demo``
console script after ``pip install``). Running this file preserves the
old behavior: ``python main.py``.
"""

from contextflow.demo import main

if __name__ == "__main__":
    main()
