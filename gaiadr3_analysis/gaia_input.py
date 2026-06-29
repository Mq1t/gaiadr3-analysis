"""gaia_input.py

Load GAIA data into a pandas DataFrame via three input methods:
    1. ADQL query
    2. CSV file upload
    3. Datalink

Usage:
    $ python gaia_input.py
"""

import pandas as pd
from astroquery.gaia import Gaia
from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
import astropy.units as u

def gaia_login_prompt():
    """Prompt the user to optionally log in to the Gaia archive."""
    if input("Log in to Gaia archive? (y/n): ").strip().lower() == "y":
        user = input("Gaia username: ").strip()
        password = input("Gaia password: ").strip()
        try:
            Gaia.login(user=user, password=password)
            print("Logged in to Gaia archive.")
        except Exception as e:
            print(f"Login failed: {e}")

def resolve_id(identifier):
    """Convert a star or cluster identifier to its Gaia DR3 source ID(s).

    Numeric input is returned as-is and is assumed to be a valid Gaia DR3 source ID. 
    Anything else is looked up via SIMBAD. If the identifier resolves directly to an 
    object with a Gaia DR3 cross-match, that single ID is returned. If no direct cross-match is
    found, SIMBAD's hierarchy is checked for children of the identifier, and the Gaia DR3 IDs 
    of every child that has one are returned as a list.

    Args:
        identifier (int or str): Gaia source ID, or any SIMBAD recognized star or cluster name/identifier.

    Returns:
        int or list[int] or None: A single Gaia DR3 source ID if the identifier resolves directly, 
        a list of Gaia DR3 source IDs if it resolves to multiple children, or None if nothing
        could be resolved.
    """
    if str(identifier).strip().isdigit():
        return int(identifier)

    result = Simbad.query_objectids(identifier)
    if result is not None:
        for row in result["id"]:
            if row.startswith("Gaia DR3 "):
                return int(row.replace("Gaia DR3 ", ""))

    # No direct cross-match on the identifier itself
    children = Simbad.query_hierarchy(identifier, hierarchy="children")
    if children is None or len(children) == 0:
        print(f"Could not resolve '{identifier}'.")
        return None

    resolved_ids = []
    for child_name in children["main_id"]:
        child_ids = Simbad.query_objectids(child_name)
        if child_ids is None:
            continue
        for row in child_ids["id"]:
            if row.startswith("Gaia DR3 "):
                resolved_ids.append(int(row.replace("Gaia DR3 ", "")))
                break

    if not resolved_ids:
        print(f"'{identifier}' has no Gaia DR3 cross-match, and none of its children do either.")
        return None

    return resolved_ids

def find_cluster(ra, dec, distance):
    """Find the cluster nearest in distance to a given sky position, using SIMBAD.

    Args:
        ra (float): Right ascension in degrees.
        dec (float): Declination in degrees.
        distance (float): Distance to the star in parsecs.

    Returns:
        str or None: Name of the best matching cluster, or None if no match is found.
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

def query_by_adql(adql_query: str = None, save_file: bool = False, file_name: str = None, identifier: int | str = None):
    """Query Gaia with an ADQL query.

    If 'identifier' is given, it is resolved to a Gaia DR3 source ID (via resolve_id) and 
    substituted into 'adql_query' wherever the placeholder '{source_id}' appears. If the 
    identifier resolves to multiple children, all resolved IDs are substituted in as a comma-separated list.

    If 'adql_query' is not given but 'identifier' is, a default query of 
    'SELECT * FROM gaiadr3.gaia_source WHERE source_id IN ({source_id})' is
    used automatically, so a bare identifier is enough to look up a star (or every member of a cluster) 
    without writing any ADQL.

    Note:
        Because an identifier can resolve to more than one Gaia ID, any custom 'adql_query' you write should 
        use an 'IN ({source_id})' clause rather than '= {source_id}'. With '=', a query will work fine for a 
        single resolved ID, but will fail with an ADQL syntax error if 'identifier' resolves to multiple IDs.

    Args:
        adql_query (str, optional): ADQL query targeting a Gaia table. May contain a '{source_id}' placeholder 
            to be filled in from 'identifier'. If omitted, 'identifier' must be given, and a default 'SELECT *' 
            query is built automatically. Defaults to None.
        save_file (bool, optional): Whether to save the result to a CSV file. Defaults to False.
        file_name (str, optional): Name or path for the saved CSV file. Defaults to the resolved Gaia ID(s) 
            if an identifier was given, or to "gaia_query" otherwise.
        identifier (int or str, optional): Gaia source ID or other star/cluster identifier to resolve and 
            insert into the query in place of '{source_id}'. Defaults to None.

    Returns:
        pandas.DataFrame: Query results.

    Raises:
        ValueError: If neither 'adql_query' nor 'identifier' is given.
    """
    if adql_query is None and identifier is None:
        raise ValueError("Provide either adql_query, identifier, or both.")

    if identifier is not None:
        gaia_id = resolve_id(identifier)

        if gaia_id is None:
            print(f"Could not resolve '{identifier}' to a Gaia DR3 source ID; query not run.")
            return None

        if isinstance(gaia_id, list):
            source_id_str = ",".join(str(i) for i in gaia_id)
            default_name = "_".join(str(i) for i in gaia_id)
        else:
            source_id_str = str(gaia_id)
            default_name = str(gaia_id)

        if file_name is None:
            file_name = default_name

        if adql_query is None:
            adql_query = "SELECT * FROM gaiadr3.gaia_source WHERE source_id IN ({source_id})"
        adql_query = adql_query.format(source_id=source_id_str)

    job = Gaia.launch_job(adql_query)
    df = job.get_results().to_pandas()

    if save_file:
        if file_name is None:
            file_name = "gaia_query"
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

    Any non-Gaia identifiers are automatically resolved to Gaia DR3 source IDs via SIMBAD (see resolve_id) before 
    the query is run. Identifiers that resolve to multiple children are expanded so every resolved star is included.

    Args:
        gaia_ids (int or str or list[int or str]): Gaia source ID(s) or other star/cluster identifier(s) 
            (resolved automatically) of the target(s) of the query.
        release (str, optional): Data release version. Defaults to 'Gaia DR3'.
        retrieval (str, optional): Retrieval type. Defaults to 'EPOCH_PHOTOMETRY'.
        structure (str, optional): Data structure. Defaults to 'INDIVIDUAL'.
        save_file (bool, optional): Whether to save each result to a CSV file. Defaults to False.
        folder_name (str, optional): Folder to save CSV files into. Required if save_file is True. Defaults to None.

    Returns:
        dict: Query results. Keys are Gaia IDs, values are epoch photometry DataFrames.

    Raises:
        TypeError: If save_file is True and folder_name is not a string.
    """
    if save_file == True and type(folder_name) != str:
        raise TypeError(f"Expected string data for folder_name, got {type(folder_name)}")

    if isinstance(gaia_ids, (int, str)):
        gaia_ids = [gaia_ids]

    # resolve_id may return a single int (direct match) or a list[int]
    resolved_ids = []
    for gid in gaia_ids:
        resolved = resolve_id(gid)
        if resolved is None:
            continue
        if isinstance(resolved, list):
            resolved_ids.extend(resolved)
        else:
            resolved_ids.append(resolved)

    if not resolved_ids:
        print("No IDs could be resolved.")
        return {}

    dl_query = Gaia.load_data(ids=resolved_ids, data_release=release, retrieval_type=retrieval, data_structure=structure)
    df_dict = {}
    retrieved_ids = set()

    for key, value in dl_query.items():
        gaia_id = int(key[26:-4])
        retrieved_ids.add(gaia_id)
        df = value[0].to_table().to_pandas()
        df_dict[gaia_id] = df

        # Add file to folder if save_file is True
        if save_file:
            file_name = folder_name + "/" + str(gaia_id) + ".csv"
            df.to_csv(file_name)

    if len(retrieved_ids) <= 0:
        print("No IDs could be retrieved. (no Epoch Photometry data).")
    else:
        not_retrieved = set(resolved_ids) - retrieved_ids
        if not_retrieved:
            print(f"IDs not retrieved (no Epoch Photometry data):\n{not_retrieved}")

    return df_dict

def find_star(
    ra: str | float = None,
    dec: str | float = None,
    coordinates: SkyCoord = None,
    columns: str = '*',
    save_file: bool = False,
    file_name: str = "star_query",
    degree_range: float = 0.0001
):
    """Query Gaia for sources within a small radius of a sky position.

    The position can be supplied either as sexagesimal RA/Dec strings, or as an astropy SkyCoord. Exactly one 
    of these input forms must resolve to a usable position, or a ValueError is raised.

    Args:
        ra (str or float, optional): Right ascension, as a sexagesimal string (e.g. "10h21m00s") when paired 
            with a string `dec`. Defaults to None.
        dec (str or float, optional): Declination, as a sexagesimal string (e.g. "+41d05m00s") when paired with 
            a string `ra`. Defaults to None.
        coordinates (astropy.coordinates.SkyCoord, optional): Sky position to search around, used if `ra`/`dec` are 
            not given. Defaults to None.
        columns (str, optional): Comma-separated columns to select from gaiadr3.gaia_source. Defaults to '*' (all columns).
        save_file (bool, optional): Whether to save the result to a CSV file. Defaults to False.
        file_name (str, optional): Name for the saved CSV file. Defaults to "star_query".
        degree_range (float, optional): Search radius in degrees. Defaults to 0.0001.

    Returns:
        pandas.DataFrame: Query results.

    Raises:
        ValueError: If neither a valid ra/dec pair nor coordinates is given.
    """
    if type(ra) == str and type(dec) == str:
        deg_coord = SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
        ra = deg_coord.ra.deg
        dec = deg_coord.dec.deg
    elif (ra is None and dec is None) and coordinates is not None:
        ra = coordinates.ra.deg
        dec = coordinates.dec.deg

    if ra is None or dec is None:
        raise ValueError("Provide either (ra and dec) as strings, or coordinates as a SkyCoord.")

    query = f"""
    SELECT TOP 10 {columns}
    FROM gaiadr3.gaia_source
    WHERE CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', {ra}, {dec}, {degree_range})
    ) = 1
    """
    if save_file:
        df = query_by_adql(query, save_file=True, file_name=file_name)
    else:
        df = query_by_adql(query)

    return df


def load_csv(file_path):
    """Load a Gaia CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pandas.DataFrame: Loaded data.
    """
    return pd.read_csv(file_path)


def apply_filter(df):
    """Filter a dataframe, or each dataframe in a dict of dataframes, using a pandas query expression.

    If 'df' is a dict of DataFrames (query_by_datalink), the same filter expression is applied to every DataFrame 
    in the dict, and a dict of filtered results is returned. The column list shown to the user is taken from the 
    first entry in the dict, since epoch photometry tables share the same schema.

    Args:
        df (pandas.DataFrame or dict[int, pandas.DataFrame]): Dataframe, or dict of dataframes, to filter.

    Returns:
        pandas.DataFrame or dict[int, pandas.DataFrame]: Filtered dataframe(s), or original if skipped/invalid.
    """
    if isinstance(df, dict):
        if not df:
            return df

        sample = next(iter(df.values()))
        print("Columns:", sample.columns.tolist())
        expression = input("Filter expression (e.g. parallax > 1.0), or Enter to skip: ").strip()

        if not expression:
            return df

        filtered = {}
        for key, value in df.items():
            try:
                filtered[key] = value.query(expression)
            except Exception as e:
                print(f"Invalid filter for ID {key}: {e}")
                filtered[key] = value
        return filtered

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
    gaia_login_prompt()

    print("\n1. ADQL query\n2. CSV file\n3. Datalink (Epoch Photometry)")
    choice = input("Choose input method (1/2/3): ").strip()

    if choice == "1":
        use_identifier = input(
            "Search by identifier (Gaia ID, star name, cluster, etc.) instead of writing a full ADQL query? (y/n): "
        ).strip().lower() == "y"

        if use_identifier:
            identifier = input("Identifier: ").strip()
            adql_query = None
        else:
            adql_query = input("ADQL query (use {source_id} as a placeholder to resolve an identifier into it): ").strip()
            id_input = input("Identifier to resolve into {source_id}, or Enter to skip: ").strip()
            identifier = id_input or None

        if input("Save to csv file? (y/n): ").strip().lower() == "y":
            path = input("Enter name for file OR path to file (press Enter for default name): ").strip()
            if path != "":
                df = query_by_adql(adql_query, save_file=True, file_name=path, identifier=identifier)
            else:
                df = query_by_adql(adql_query, save_file=True, identifier=identifier)
        else:
            df = query_by_adql(adql_query, identifier=identifier)

    elif choice == "2":
        filepath = input("CSV path: ").strip()
        df = load_csv(filepath)

    elif choice == "3":
        raw = input("Gaia source ID(s) or star name(s), comma-separated: ").strip()
        gaia_ids = [i.strip() for i in raw.split(",")]

        if input("Save to csv file? (y/n): ").strip().lower() == "y":
            path = input("Enter folder name OR path to folder (folder must exist): ").strip()
            if path != "":
                df = query_by_datalink(gaia_ids, save_file=True, folder_name=path)
            else:
                df = query_by_datalink(gaia_ids, save_file=True, folder_name=".")
        else:
            df = query_by_datalink(gaia_ids)
    else:
        print("Invalid choice.")
        return None

    return df

if __name__ == "__main__":
    df = get_dataframe()
    if df is not None:
        print(df.head())