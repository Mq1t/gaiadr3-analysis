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
from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
import astropy.units as u

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

def resolve_id(identifier):
    """Convert any star identifier to the Gaia DR3 source ID.
 
    Numeric input is returned as-is. Anything else is looked up via SIMBAD.
 
    Args:
        identifier (int | str): Gaia source ID or any SIMBAD-recognised star name/identifier.
 
    Returns:
        int | None: Gaia DR3 source ID, or None if it can't be resolved.
    """
    if str(identifier).strip().isdigit():
        return int(identifier)
 
    result = Simbad.query_objectids(identifier)
    if result is None:
        print(f"Could not resolve '{identifier}'.")
        return None
 
    for row in result["id"]:
        if row.startswith("Gaia DR3 "):
            return int(row.replace("Gaia DR3 ", ""))
 
    print(f"'{identifier}' has no Gaia DR3 cross-match.")
    return None

def find_cluster(ra, dec, distance):
    """Find the cluster nearest in distance to a given sky position, using SIMBAD.

    Args:
        ra (float): Right ascension in degrees.
        dec (float): Declination in degrees.
        distance (float): Distance to the star in parsecs.

    Returns:
        str | None: Name of the best matching cluster, or None if no match is found.
    """
    Simbad.add_votable_fields('otype', 'parallax')
    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    result = Simbad.query_region(coord, radius=0.5 * u.deg)

    if result is None:
        return None

    best_name, best_diff = None, None
    for row in result:
        if row["otype"] not in ("Cl*", "OpC", "GlC", "As*"):
            continue
        plx = row["plx_value"]
        if not plx or plx <= 0:
            continue

        diff = abs((1000 / plx) - distance)
        if best_diff is None or diff < best_diff:
            best_diff, best_name = diff, row["main_id"]

    return best_name

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
        if input("Save to csv file? (y/n): ").strip().lower() == "y":
            path = input("Enter name for file OR path to file (press Enter for default name): ").strip()
            if path != "":
                df = query_by_adql(adql_query, save_file = True, file_name = path)
            else:
                df = query_by_adql(adql_query, save_file = True)
        else:
            df = query_by_adql(adql_query)
 
    elif choice == "2":
        filepath = input("CSV path: ").strip()
        df = load_csv(filepath)
 
    elif choice == "3":
        raw = input("Gaia source ID(s) or star name(s), comma-separated: ").strip()
        gaia_ids = [i.strip() for i in raw.split(",")]
        
        if input("Save to csv file? (y/n): ").strip().lower() == "y":
            path = input("Enter folder name OR path to folder (folder must exist): ").strip()
            if path != "":
                df = query_by_datalink(gaia_ids, folder_name = path)
        else:
            df = query_by_datalink(gaia_ids)


        """ I think we should remove this and only return the full dictionary.
        if results:
            first_id = next(iter(results))
            df = results[first_id]
            print(f"Returning epoch photometry for ID {first_id}. Full dictionary available via query_by_datalink().")
        else:
            return None
        """
    else:
        print("Invalid choice.")
        return None
 
    if input("Apply a filter? (y/n): ").strip().lower() == "y":
        df = apply_filter(df)
 
    return df


def query_by_adql(adql_query, save_file: bool = False, file_name: str = None):
    """Query Gaia with an ADQL query.
 
    If 'identifier' is given, it is resolved to a Gaia DR3 source ID (via resolve_id) and substituted into 'adql_query' wherever the
    placeholder '{source_id}' appears.
 
    Args:
        adql_query (str): ADQL query targeting a Gaia table. May contain a '{source_id}' placeholder to be filled in from 'identifier'.
        identifier (int | str, optional): Gaia source ID or other star identifier to resolve and insert into the query in place of 
            '{source_id}'. Defaults to None.
 
    Returns:
        pandas.DataFrame: Query results
    """
                        
    if identifier is not None:
        gaia_id = resolve_id(identifier)
            
        if gaia_id is None:
            print(f"Could not resolve '{identifier}' to a Gaia DR3 source ID; query not run.")
            return None
        elif file_name == None:
            file_name = str(gaia_id)
        adql_query = adql_query.format(source_id=gaia_id)
 
    job = Gaia.launch_job(adql_query)
    df = job.get_results().to_pandas()
    
    if save_file:
            formatted_name = file_name.replace(" ", "_")
            formatted_name = f"{formatted_name}.csv"
            df.to_csv(formatted_name)
    
    return df

def query_by_datalink(
    gaia_ids: int | list[int], 
    release: str = 'Gaia DR3', 
    retrieval: str = 'EPOCH_PHOTOMETRY', 
    structure: str = 'INDIVIDUAL',
    save_file: bool = False,
    folder_name: str = None
):
    """Query Gaia with a Datalink query.
 
    Any non-Gaia identifiers are automatically resolved to Gaia DR3 source IDs via SIMBAD (see resolve_id) before the query is run.
 
    Args:
        gaia_ids (int | str | list[int | str]): Gaia source ID(s) or other
            star identifier(s) (resolved automatically) of the star(s) targeted by the query.
        release (str): Data release version. Default is 'Gaia DR3'.
        retrieval (str): Retrieval type. Default is 'EPOCH_PHOTOMETRY'.
        structure (str): Data structure. Default is 'INDIVIDUAL'.
 
    Returns:
        dict: Query results. Keys are Gaia IDs, values are epoch photometry DataFrames.
    """

     #Make sure folder_name exists if save_file is True
    if save_file == True and type(folder_name) != str:
        raise TypeError(f"Expected string data for folder_name, got {type(folder_name)}")
    
    if isinstance(gaia_ids, (int, str)):
        gaia_ids = [gaia_ids]
 
    resolved_ids = [r for r in (resolve_id(gid) for gid in gaia_ids) if r is not None]
 
    if not resolved_ids:
        print("No IDs could be resolved.")
        return {}
 
    dl_query = Gaia.load_data(ids=resolved_ids, data_release=release, retrieval_type=retrieval, data_structure=structure)
    df_dict = {}
    retrieved_ids = set()
    
    for key, value in dl_query.items():
        retrieved_ids.add(key)
        gaia_id = int(key[26:-4])
        df = value[0].to_table().to_pandas()
        df_dict[gaia_id] = df

        #Add file to folder if save_file is True
        if save_file: 
            file_name = folder_name+"/"+str(gaia_id)+".csv" 
            df.to_csv(file_name)
            
 
    if len(retrieved_ids) <= 0:
        print("No IDs could be retrieved. (no Epoch Photometry data).")
    else:
        not_retrieved = set(resolved_ids) - retrieved_ids
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