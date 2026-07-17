#Creating a constant for which I can import for the Julian Date conversion for Gaia DR3 analysis.
JD_offset = 2455197.5

SPTYPE_TEFF_RANGES = {
    "O": (30000, 60000),
    "B": (10000, 30000),
    "A": (7500, 10000),
    "F": (6000, 7500),
    "G": (5200, 6000),
    "K": (3700, 5200),
    "M": (2400, 3700),
}