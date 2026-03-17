"""
conftest.py — Mock unavailable third-party modules (ta, yfinance)
so that tests can import project modules without install failures.
"""

import sys
from unittest.mock import MagicMock

# Mock 'ta' and its submodules before any project code imports them
for mod_name in ["ta", "ta.momentum", "ta.trend", "yfinance"]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()
