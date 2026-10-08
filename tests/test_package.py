from __future__ import annotations

import importlib.metadata

import pytest

import cditools as m


@pytest.mark.skip
def test_version():
    assert importlib.metadata.version("cditools") == m.__version__
