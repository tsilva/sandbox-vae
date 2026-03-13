from __future__ import annotations

import os


def pytest_configure() -> None:
    os.environ.setdefault("SANDBOX_AUTOENCODERS_TEST_MODE", "1")
