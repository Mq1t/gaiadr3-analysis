"""Tests for epoch_photometry.py

Tests cover the phase calculation function and input validation
for lightcurve and lomb_scargle. Matplotlib rendering is patched
out so tests run headlessly without a display.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch
from gaiadr3_analysis.epoch_photometry import phase, lightcurve, lomb_scargle

# Fixtures
@pytest.fixture
def photometry_df():
    """Returns a minimal epoch photometry DataFrame for lightcurve tests.

    Returns:
        pd.DataFrame: DataFrame with required lightcurve columns.
    """
    n = 10
    return pd.DataFrame({
        "g_transit_time": np.linspace(0, 100, n),
        "g_transit_mag": np.random.uniform(12, 13, n),
        "bp_obs_time": np.linspace(0, 100, n),
        "bp_mag": np.random.uniform(12.5, 13.5, n),
        "rp_obs_time": np.linspace(0, 100, n),
        "rp_mag": np.random.uniform(11.5, 12.5, n),
        "variability_flag_g_reject": [False] * n,
        "variability_flag_bp_reject": [False] * n,
        "variability_flag_rp_reject": [False] * n,
    })

@pytest.fixture
def time_series():
    """Returns a simple time and magnitude array for lomb_scargle tests.

    Returns:
        tuple: (t, mag) as pandas Series with 50 evenly spaced points.
    """
    t = pd.Series(np.linspace(0, 50, 50))
    mag = pd.Series(np.sin(2 * np.pi * t / 5.0))
    return t, mag

# phase
def test_phase_output_in_range():
    """Checks that phase values are always in the interval [0, 1)."""
    t = np.array([0.0, 1.0, 2.5, 10.0, 100.0])
    result = phase(t, T_0=0.0, P=3.0)

    assert np.all(result >= 0.0)
    assert np.all(result < 1.0)

def test_phase_at_reference_epoch_is_zero():
    """Checks that phase is 0.0 when t equals T_0."""
    result = phase(np.array([5.0]), T_0=5.0, P=3.0)

    assert result[0] == pytest.approx(0.0)

def test_phase_one_full_period_is_zero():
    """Checks that t = T_0 + P wraps back to phase 0.0."""
    result = phase(np.array([8.0]), T_0=5.0, P=3.0)

    assert result[0] == pytest.approx(0.0)

def test_phase_half_period_is_point_five():
    """Checks that t = T_0 + P/2 gives phase 0.5."""
    result = phase(np.array([6.5]), T_0=5.0, P=3.0)

    assert result[0] == pytest.approx(0.5)

def test_phase_returns_numpy_array():
    """Checks that phase always returns a numpy array."""
    t = np.array([1.0, 2.0, 3.0])
    result = phase(t, T_0=0.0, P=2.0)

    assert isinstance(result, np.ndarray)

def test_phase_handles_array_input():
    """Checks that phase correctly handles a multi-element array."""
    t = np.array([0.0, 1.0, 2.0, 3.0])
    result = phase(t, T_0=0.0, P=2.0)

    assert len(result) == 4

# lightcurve
def test_lightcurve_raises_type_error_for_non_dataframe():
    """Checks that lightcurve raises TypeError when given a non-DataFrame input."""
    with pytest.raises(TypeError):
        lightcurve([1, 2, 3])

def test_lightcurve_raises_key_error_for_missing_columns():
    """Checks that lightcurve raises KeyError when required columns are missing."""
    df = pd.DataFrame({"ra": [1, 2], "dec": [3, 4]})
    with pytest.raises(KeyError):
        lightcurve(df)

def test_lightcurve_runs_overplot_mode(photometry_df):
    """Checks that lightcurve runs without error in overplot mode.

    Args:
        photometry_df (pd.DataFrame): Sample photometry DataFrame fixture.
    """
    with patch("matplotlib.pyplot.show"):
        lightcurve(photometry_df, overplot=True)

def test_lightcurve_runs_subplot_mode(photometry_df):
    """Checks that lightcurve runs without error in subplot mode.

    Args:
        photometry_df (pd.DataFrame): Sample photometry DataFrame fixture.
    """
    with patch("matplotlib.pyplot.show"):
        lightcurve(photometry_df, overplot=False)

def test_lightcurve_runs_with_period(photometry_df):
    """Checks that lightcurve runs without error when period is provided.

    Args:
        photometry_df (pd.DataFrame): Sample photometry DataFrame fixture.
    """
    with patch("matplotlib.pyplot.show"):
        lightcurve(photometry_df, period=5.0)

def test_lightcurve_runs_with_rejectflags(photometry_df):
    """Checks that lightcurve runs without error when rejectflags is True.

    Args:
        photometry_df (pd.DataFrame): Sample photometry DataFrame fixture.
    """
    with patch("matplotlib.pyplot.show"):
        lightcurve(photometry_df, rejectflags=True)

def test_lightcurve_runs_with_xlims_and_ylims(photometry_df):
    """Checks that lightcurve accepts xlims and ylims without error.

    Args:
        photometry_df (pd.DataFrame): Sample photometry DataFrame fixture.
    """
    with patch("matplotlib.pyplot.show"):
        lightcurve(photometry_df, xlims=(0, 100), ylims=(11, 14))

# lomb_scargle
def test_lomb_scargle_returns_float(time_series):
    """Checks that lomb_scargle returns a float value.

    Args:
        time_series (tuple): Sample (t, mag) time series fixture.
    """
    t, mag = time_series
    result = lomb_scargle(t, mag)

    assert isinstance(result, float)

def test_lomb_scargle_returns_positive_period(time_series):
    """Checks that lomb_scargle returns a positive period value.

    Args:
        time_series (tuple): Sample (t, mag) time series fixture.
    """
    t, mag = time_series
    result = lomb_scargle(t, mag)

    assert result > 0.0

def test_lomb_scargle_default_example_returns_float():
    """Checks that lomb_scargle returns a float when called with no arguments."""
    result = lomb_scargle()

    assert isinstance(result, float)

def test_lomb_scargle_with_period_range(time_series):
    """Checks that lomb_scargle respects a custom period_range.

    Args:
        time_series (tuple): Sample (t, mag) time series fixture.
    """
    t, mag = time_series
    result = lomb_scargle(t, mag, period_range=[1.0, 20.0])

    assert 1.0 <= result <= 20.0

def test_lomb_scargle_plot_runs_without_error(time_series):
    """Checks that lomb_scargle runs without error when plot is True.

    Args:
        time_series (tuple): Sample (t, mag) time series fixture.
    """
    t, mag = time_series
    with patch("matplotlib.pyplot.show"):
        lomb_scargle(t, mag, plot=True)