"""Tests for gaia_input.py

Tests cover CSV loading, filtering logic, and input validation.
Gaia ADQL queries and interactive input functions are excluded
from unit tests as they require network access or user input.
"""

import pytest
import pandas as pd
from unittest.mock import patch
from gaiadr3_analysis.gaia_input import load_csv, apply_filter

# Fixtures
@pytest.fixture
def sample_df():
    """Returns a minimal Gaia-like DataFrame for testing.

    Returns:
        pd.DataFrame: DataFrame with ra, dec, parallax, and pmra columns.
    """
    return pd.DataFrame({
        "ra": [10.0, 20.0, 30.0],
        "dec": [-5.0, 15.0, 25.0],
        "parallax": [0.5, 1.5, 2.5],
        "pmra": [0.1, 0.2, 0.3],
    })

# load_csv
def test_load_csv_returns_dataframe(tmp_path):
    """Checks that load_csv returns a pandas DataFrame.

    Args:
        tmp_path: pytest built-in fixture providing a temporary directory.
    """
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("ra,dec\n10.0,-5.0\n20.0,15.0\n")

    result = load_csv(str(csv_file))

    assert isinstance(result, pd.DataFrame)

def test_load_csv_correct_columns(tmp_path):
    """Checks that load_csv preserves column names from the CSV file.

    Args:
        tmp_path: pytest built-in fixture providing a temporary directory.
    """
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("ra,dec,parallax\n10.0,-5.0,1.0\n")

    result = load_csv(str(csv_file))

    assert list(result.columns) == ["ra", "dec", "parallax"]

def test_load_csv_correct_row_count(tmp_path):
    """Checks that load_csv loads the correct number of rows.

    Args:
        tmp_path: pytest built-in fixture providing a temporary directory.
    """
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("ra,dec\n10.0,-5.0\n20.0,15.0\n30.0,25.0\n")

    result = load_csv(str(csv_file))

    assert len(result) == 3

def test_load_csv_file_not_found():
    """Checks that load_csv raises FileNotFoundError for a missing file."""
    with pytest.raises(FileNotFoundError):
        load_csv("nonexistent_file.csv")

# apply_filter
def test_apply_filter_valid_expression(sample_df):
    """Checks that apply_filter correctly filters rows using a valid expression.

    Args:
        sample_df (pd.DataFrame): Sample Gaia DataFrame fixture.
    """
    with patch("builtins.input", return_value="parallax > 1.0"):
        result = apply_filter(sample_df)

    assert len(result) == 2
    assert all(result["parallax"] > 1.0)

def test_apply_filter_empty_expression_returns_original(sample_df):
    """Checks that apply_filter returns the original DataFrame when skipped.

    Args:
        sample_df (pd.DataFrame): Sample Gaia-like DataFrame fixture.
    """
    with patch("builtins.input", return_value=""):
        result = apply_filter(sample_df)

    assert len(result) == len(sample_df)

def test_apply_filter_invalid_expression_returns_original(sample_df):
    """Checks that apply_filter returns the original DataFrame on invalid input.

    Args:
        sample_df (pd.DataFrame): Sample Gaia-like DataFrame fixture.
    """
    with patch("builtins.input", return_value="not_a_column > 1"):
        result = apply_filter(sample_df)

    assert len(result) == len(sample_df)

def test_apply_filter_exact_match(sample_df):
    """Checks that apply_filter returns only rows matching an exact condition.

    Args:
        sample_df (pd.DataFrame): Sample Gaia-like DataFrame fixture.
    """
    with patch("builtins.input", return_value="parallax == 0.5"):
        result = apply_filter(sample_df)

    assert len(result) == 1
    assert result["parallax"].iloc[0] == 0.5