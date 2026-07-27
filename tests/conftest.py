"""Shared fixtures for armv7m-decoder tests."""

import pytest

from armv7m_decoder import Context


@pytest.fixture
def ctx() -> Context:
    return Context()
