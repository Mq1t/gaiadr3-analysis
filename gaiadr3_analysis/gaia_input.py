"""gaia_input.py

Load GAIA data into a pandas DataFrame via two input methods:
    1. ADQL query 
    2. CSV file upload

Usage:
    $ python gaia_input.py
"""

import pandas as pd
from astroquery.gaia import Gaia

def query_by_adql(adql_query):
    """Query Gaia with an ADQL query

    Args:
        adql_query (str): ADQL query targeting a Gaia table

    Returns:
        pandas.DataFrame: Query results
    """
    job = Gaia.launch_job(adql_query)
    return job.get_results().to_pandas()

def query_by_datalink(gaia_ids:[int], release:str ='Gaia DR3', retrieval:str = 'EPOCH_PHOTOMETRY', structure:str = 'INDIVIDUAL'):
    """Query Gaia with an Datalink query

    Args:
        gaia_id ([int]): Gaia ID of the star targetted by the query.
        release (str): Data release version. Default is 'Gaia DR3'.
        retrieval (str): Retrieval type. Default is 'EPOCH_PHOTOMETRY'.
        structture (str): Data structure. Default is 'INDIVIDUAL'.

    Returns:
        pandas.DataFrame: Query results
    """
    
    dl_query = Gaia.load_data(ids=gaia_ids, data_release=release, retrieval_type=retrieval, data_structure=structure)
    dl_query = list(dl_query.values())[0][0].to_table().to_pandas()
    return dl_query
    
def load_csv(file_path):
    """Load a Gaia CSV file

    Args:
        file_path (str): Path to the CSV file

    Returns:
        pandas.dataframe: Loaded data
    """
    return pd.read_csv(file_path)


def apply_filter(df):
    """Filter a dataframe using a pandas query expression

    Args:
        df (pandas.dataframe): dataframe to filter

    Returns:
        pandas.dataframe: Filtered dataframe, or original if skipped/invalid
    """
    print("Columns:", df.columns.tolist())
    expression = input("Filter expression (e.g. parallax > 1.0), or Enter to skip: ").strip()

    if not expression:
        return df
    try:
        return df.query(expression)
    except Exception as e:
        print(f"Invalid filter: {e}")
        return df


def get_dataframe():
    """Run the input menu and return a Gaia dataframe.

    Returns:
        pandas.DataFrame: Final dataframe for downstream use.
    """
    print("\n1. ADQL query\n2. CSV file")
    choice = input("Choose input method (1/2): ").strip()

    if choice == "1":
        adql_query = input("ADQL query: ").strip()
        df = query_by_adql(adql_query)

    elif choice == "2":
        filepath = input("CSV path: ").strip()
        df = load_csv(filepath)

    else:
        print("Invalid choice.")
        return None

    if input("Apply a filter? (y/n): ").strip().lower() == "y":
        df = apply_filter(df)

    return df

if __name__ == "__main__":
    df = get_dataframe()
    if df is not None:
        print(df.head())