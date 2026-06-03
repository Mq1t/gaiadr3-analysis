"""hr_diagram.py

Plot an HR diagram from a Gaia pandas datafame

Usage:
    Import and call plot_hr_diagram(df) with a dataframe from gaia_input.py
"""

import numpy as np
import matplotlib.pyplot as plt


def get_distance(parallax):
    """Convert parallax (mas) to distance (pc)

    Args:
        parallax (float): Parallax in milliarcseconds

    Returns:
        float: Distance in parsecs
    """
    return 1 / (parallax / 1000)


def get_magnitude(apparent_mag, distance):
    """Convert apparent magnitude to absolute magnitude

    Args:
        apparent_mag (float): Apparent G magnitude
        distance (float): Distance in parsecs

    Returns:
        float: Absolute magnitude
    """
    return apparent_mag - 5 * np.log10(distance / 10)


def get_bprp(bp, rp):
    """Calculate BP-RP colour index

    Args:
        bp (float): BP magnitude
        rp (float): RP magnitude

    Returns:
        float: BP-RP colour index
    """
    return bp - rp


def plot_hr_diagram(df):
    """Plot an HR diagram from a Gaia dataframe.

    Args:
        df (pandas.dataframe): Gaia data containing parallax,
            phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag.
    """
    df = df.dropna(subset=["parallax", "phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag"])

    magnitude = [get_magnitude(get_distance(row["parallax"]), row["phot_g_mean_mag"]) for _, row in df.iterrows()]
    bprp = [get_bprp(row["phot_bp_mean_mag"], row["phot_rp_mean_mag"]) for _, row in df.iterrows()]

    plt.style.use("dark_background")
    plt.scatter(bprp, magnitude, c="white", s=1)
    plt.xlabel("BP - RP")
    plt.ylabel("Absolute Magnitude")
    plt.title("HR Diagram")
    plt.gca().invert_yaxis()
    plt.show()