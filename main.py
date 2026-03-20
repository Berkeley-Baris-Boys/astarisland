#!/usr/bin/env python3
"""
Compatibility shim.

The canonical orchestrator now lives in solver.py.
This module re-exports key entrypoints so existing imports/tests keep working.
"""

from solver import load_active_round, main, run_query_phase


if __name__ == "__main__":
    main()
