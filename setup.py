from setuptools import setup, find_packages

setup(
    name="gaiadr3-analysis",
    version="0.1.0",
    description="Tools for Gaia DR3 astrophysical data analysis",
    author="Matthew McGuire, Jennifer Farrell, Nakia McKay",
    packages=find_packages(),
    python_requires=">=3.9",
    license="Apache License 2.0",
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
    install_requires=[
        "astropy>=5.0",
        "numpy>=1.22",
        "pandas>=1.4",
        "matplotlib>=3.5",
        "scipy>=1.8",
        "astroquery>=0.4",
    ],
    include_package_data=True,
)
