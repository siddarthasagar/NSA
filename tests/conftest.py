"""Shared pytest fixtures and configuration for NSA test suite."""

import os
import shutil
from pathlib import Path
from typing import Generator

import pytest


@pytest.fixture(scope="session")
def test_cache_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create temporary cache directory for tests.
    
    This fixture creates a session-scoped temporary directory that mimics
    the production cache structure for testing purposes.
    
    Args:
        tmp_path_factory: pytest's temporary path factory
        
    Returns:
        Path to temporary cache directory
    """
    cache_dir = tmp_path_factory.mktemp("test_cache")
    
    # Create cache subdirectories
    (cache_dir / "checkpoints").mkdir(exist_ok=True)
    (cache_dir / "data").mkdir(exist_ok=True)
    (cache_dir / "logs").mkdir(exist_ok=True)
    (cache_dir / "tta").mkdir(exist_ok=True)
    
    return cache_dir


@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create temporary data directory for test datasets.
    
    This fixture creates a session-scoped temporary directory for storing
    test datasets and generated data files.
    
    Args:
        tmp_path_factory: pytest's temporary path factory
        
    Returns:
        Path to temporary data directory
    """
    data_dir = tmp_path_factory.mktemp("test_data")
    return data_dir


@pytest.fixture(autouse=True)
def setup_test_environment(
    monkeypatch: pytest.MonkeyPatch, test_cache_dir: Path
) -> Generator[None, None, None]:
    """Configure environment variables for test execution.
    
    This fixture automatically runs for every test and sets up the necessary
    environment variables to redirect cache operations to test directories.
    
    Args:
        monkeypatch: pytest's monkeypatch fixture for modifying environment
        test_cache_dir: temporary cache directory fixture
        
    Yields:
        None during test execution
    """
    # Set cache directory to test location
    monkeypatch.setenv("NSA_CACHE_DIR", str(test_cache_dir))
    
    # Ensure we're using test mode
    monkeypatch.setenv("NSA_TEST_MODE", "1")
    
    yield
    
    # Cleanup is handled automatically by tmp_path_factory


@pytest.fixture
def sample_arc_task() -> dict:
    """Provide a minimal ARC task for testing.
    
    Returns a simple ARC task with one training example and one test example.
    This can be used for quick integration tests without loading real datasets.
    
    Returns:
        Dictionary containing train and test examples
    """
    return {
        "train": [
            {
                "input": [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                "output": [[0, 0, 0], [0, 2, 0], [0, 0, 0]],
            }
        ],
        "test": [{"input": [[0, 0, 0], [0, 1, 0], [0, 0, 0]]}],
    }


@pytest.fixture
def minimal_vocab() -> dict[str, int]:
    """Provide a minimal vocabulary for testing tokenization.
    
    Returns:
        Dictionary mapping tokens to integer IDs
    """
    return {
        "<pad>": 0,
        "<sos>": 1,
        "<eos>": 2,
        "extract": 3,
        "move_node": 4,
        "rotate_node": 5,
    }
