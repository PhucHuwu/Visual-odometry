#!/usr/bin/env python3
"""Run Script - Entry point wrapper

Chạy Visual Odometry system với proper imports.
"""

from main import main
import sys
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).resolve().parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Import và chạy main

if __name__ == '__main__':
    sys.exit(main())
