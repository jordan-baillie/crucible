"""Shared test config: auto-skip network-marked tests in CI.

GitHub's US runners get HTTP 451 (geo-block) from Binance and flaky responses from
CFTC/CBOE. The network adapter tests run locally, where the real data work happens.
"""
import os

import pytest


def pytest_collection_modifyitems(config, items):
    if not os.environ.get("CI"):
        return
    skip = pytest.mark.skip(reason="network tests skipped in CI (geo-blocks / flaky public endpoints)")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)
