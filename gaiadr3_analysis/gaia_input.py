"""gaia_input.py

Load GAIA data into a pandas DataFrame via three input methods:
    1. ADQL query
    2. CSV file upload
    3. Datalink (Epoch Photometry)

Usage:
    $ python gaia_input.py
"""

import pandas as pd
from astroquery.gaia import Gaia

def gaia_login_prompt():
    """Prompt user to optionally log in to Gaia archive."""
    if input("Log in to Gaia archive? (y/n): ").strip().lower() == "y":
        user = input("Gaia username: ").strip()
        password = input("Gaia password: ").strip()
        try:
            Gaia.login(user=user, password=password)
            print("Logged in to Gaia archive.")
        except Exception as e:
            print(f"Login failed: {e}")


def get_dataframe():
    """Run the input menu and return a Gaia dataframe.

    Returns:
        pandas.DataFrame: Final dataframe for downstream use.
    """
    
    gaia_login_prompt()

    print("\n1. ADQL query\n2. CSV file\n3. Datalink (Epoch Photometry)")
    choice = input("Choose input method (1/2/3): ").strip()

    if choice == "1":
        adql_query = input("ADQL query: ").strip()
        if input("Save file to CSV? (y/n): ").strip().lower() == "y":
            f_name = input("Enter file name or path. (Press enter for default): "):
                if(f_name is not None):
                    df = query_by_adql(adql_query, save_file = True, file_name = f_name)
                else:
                    df = query_by_adql(adql_query, save_file = True)
        else:
            df = query_by_adql(adql_query)
        
    elif choice == "2":
        filepath = input("CSV path: ").strip()
        df = load_csv(filepath)

    elif choice == "3":
        raw = input("Gaia source ID(s), comma-separated: ").strip()
        gaia_ids = [int(i.strip()) for i in raw.split(",")]
        results = query_by_datalink(gaia_ids)

        if input("Save file to CSV? (y/n): ").strip().lower() == "y":
            f_name = input("Enter folder name or path. (Press enter for default): "):
                if(f_name is not None):
                    results = query_by_datalink(gaia_ids, save_file = True, folder_name = f_name)
                else:
                    results = query_by_datalink(gaia_ids, save_file = True)
        else:
            results = query_by_datalink(gaia_ids)

        #Note, I think we should remove this and just return results.
        if results:
            first_id = next(iter(results))
            df = results[first_id]
            print(f"Returning epoch photometry for ID {first_id}. Full dictionary available via query_by_datalink().")
        else:
            return None

    else:
        print("Invalid choice.")
        return None

    if input("Apply a filter? (y/n): ").strip().lower() == "y":
        df = apply_filter(df)

    return df


def query_by_adql(adql_query, save_file:bool = False, file_name: str = "ADQL_query.csv"):
    """Query Gaia with an ADQL query.

    Args:
        adql_query (str): ADQL query targeting a Gaia table
        save_file (bool, optional): If True, save file to CSV.
        file_name (str, optional): Name of the file, only applies if save_file is True. 

    Returns:
        pandas.DataFrame: Query results
    """
    job = Gaia.launch_job(adql_query)

    if(save_file == True):
        df.to_csv(file_name, index=False)
    
    return job.get_results().to_pandas()


def query_by_datalink(
    gaia_ids: int | list[int], 
    release: str = 'Gaia DR3', 
    retrieval: str = 'EPOCH_PHOTOMETRY', 
    structure: str = 'INDIVIDUAL',
    save_file:bool = False, 
    fodler_name: str = "DL_query"
):
    """Query Gaia with a Datalink query.

    Args:
        gaia_ids ([int] | int): Gaia ID(s) of the star(s) targeted by the query.
        release (str): Data release version. Default is 'Gaia DR3'.
        retrieval (str): Retrieval type. Default is 'EPOCH_PHOTOMETRY'.
        structure (str): Data structure. Default is 'INDIVIDUAL'.
        save_file (bool, optional): If True, save file to CSV.
        folder_name (str, optional): Name of the folder, only applies if save_file is True. 

    Returns:
        dict: Query results. Keys are Gaia IDs, values are epoch photometry DataFrames.
    """
    if isinstance(gaia_ids, int):
        gaia_ids = [gaia_ids]

    dl_query = Gaia.load_data(ids=gaia_ids, data_release=release, retrieval_type=retrieval, data_structure=structure)
    df_dict = {}
    retrieved_ids = set()

    for key, value in dl_query.items():
        retrieved_ids.add(key)
        gaia_id = int(key[26:-4])
        df = value[0].to_table().to_pandas()
        df_dict[gaia_id] = df

        #Creates a folder named {folder_name} that has all the CSV files of the data where each 
        if(save_file == True):
            file_path = Path(f"{folder_name}/{str(gaia_id)}.csv")
            df.to_csv(folder_path, index=False)

    if len(retrieved_ids) <= 0:
        print("No IDs could be retrieved. (no Epoch Photometry data).")
    else:
        not_retrieved = set(gaia_ids) - retrieved_ids
        if not_retrieved:
            print(f"IDs not retrieved (no Epoch Photometry data):\n{not_retrieved}")


    return df_dict
    
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

if __name__ == "__main__":
    df = get_dataframe()
    if df is not None:
        print(df.head())