"""Tests for mean_photometry.py

Tests cover input validation for ra_vs_dec and pmra_vs_pmdec.
Matplotlib rendering is patched out so tests run without a display.
"""

import pytest
import pandas as pd
from unittest.mock import patch
from gaiadr3_analysis.mean_photometry import ra_vs_dec, pmra_vs_pmdec

# Fixtures
@pytest.fixture
def sky_df():
    """Returns a DataFrame with ra and dec columns for position plot tests.

    Returns:
        pd.DataFrame: DataFrame with ra and dec columns.
    """
    return pd.DataFrame({
        "ra": [10.0, 20.0, 30.0],
        "dec": [-5.0, 15.0, 25.0],
    })

@pytest.fixture
def proper_motion_df():
    """Returns a DataFrame with pmra and pmdec columns for proper motion tests.

    Returns:
        pd.DataFrame: DataFrame with pmra and pmdec columns.
    """
    return pd.DataFrame({
        "pmra": [0.1, 0.2, 0.3],
        "pmdec": [-0.1, 0.0, 0.1],
    })

# ra_vs_dec
def test_ra_vs_dec_raises_type_error_for_non_dataframe():
    """Checks that ra_vs_dec raises TypeError when given a non-DataFrame input."""
    with pytest.raises(TypeError):
        ra_vs_dec([1, 2, 3])

def test_ra_vs_dec_raises_key_error_for_missing_ra(sky_df):
    """Checks that ra_vs_dec raises KeyError when the ra column is missing.

    Args:
        sky_df (pd.DataFrame): Sample sky position DataFrame fixture.
    """
    df = sky_df.drop(columns=["ra"])
    with pytest.raises(KeyError):
        ra_vs_dec(df)

def test_ra_vs_dec_raises_key_error_for_missing_dec(sky_df):
    """Checks that ra_vs_dec raises KeyError when the dec column is missing.

    Args:
        sky_df (pd.DataFrame): Sample sky position DataFrame fixture.
    """
    df = sky_df.drop(columns=["dec"])
    with pytest.raises(KeyError):
        ra_vs_dec(df)

def test_ra_vs_dec_runs_without_error(sky_df):
    """Checks that ra_vs_dec completes without error on valid input.

    Args:
        sky_df (pd.DataFrame): Sample sky position DataFrame fixture.
    """
    with patch("matplotlib.pyplot.show"):
        ra_vs_dec(sky_df)

def test_ra_vs_dec_accepts_custom_title(sky_df):
    """Checks that ra_vs_dec accepts a custom title without raising an error.

    Args:
        sky_df (pd.DataFrame): Sample sky position DataFrame fixture.
    """
    with patch("matplotlib.pyplot.show"):
        ra_vs_dec(sky_df, title="My Custom Title")

def test_ra_vs_dec_accepts_xlim_and_ylim(sky_df):
    """Checks that ra_vs_dec accepts xlim and ylim without raising an error.

    Args:
        sky_df (pd.DataFrame): Sample sky position DataFrame fixture.
    """
    with patch("matplotlib.pyplot.show"):
        ra_vs_dec(sky_df, xlim=50.0, ylim=50.0)

# pmra_vs_pmdec
def test_pmra_vs_pmdec_raises_type_error_for_non_dataframe():
    """Checks that pmra_vs_pmdec raises TypeError when given a non-DataFrame input."""
    with pytest.raises(TypeError):
        pmra_vs_pmdec("not a dataframe")

def test_pmra_vs_pmdec_raises_key_error_for_missing_pmra(proper_motion_df):
    """Checks that pmra_vs_pmdec raises KeyError when pmra column is missing.

    Args:
        proper_motion_df (pd.DataFrame): Sample proper motion DataFrame fixture.
    """
    df = proper_motion_df.drop(columns=["pmra"])
    with pytest.raises(KeyError):
        pmra_vs_pmdec(df)

def test_pmra_vs_pmdec_raises_key_error_for_missing_pmdec(proper_motion_df):
    """Checks that pmra_vs_pmdec raises KeyError when pmdec column is missing.

    Args:
        proper_motion_df (pd.DataFrame): Sample proper motion DataFrame fixture.
    """
    df = proper_motion_df.drop(columns=["pmdec"])
    with pytest.raises(KeyError):
        pmra_vs_pmdec(df)

def test_pmra_vs_pmdec_runs_without_error(proper_motion_df):
    """Checks that pmra_vs_pmdec completes without error on valid input.

    Args:
        proper_motion_df (pd.DataFrame): Sample proper motion DataFrame fixture.
    """
    with patch("matplotlib.pyplot.show"):
        pmra_vs_pmdec(proper_motion_df)

def test_pmra_vs_pmdec_accepts_custom_title(proper_motion_df):
    """Checks that pmra_vs_pmdec accepts a custom title without raising an error.

    Args:
        proper_motion_df (pd.DataFrame): Sample proper motion DataFrame fixture.
    """
    with patch("matplotlib.pyplot.show"):
        pmra_vs_pmdec(proper_motion_df, title="Proper Motion Plot")