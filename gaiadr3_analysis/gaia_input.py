"""gaia_input.py

Load GAIA data into a pandas DataFrame via three input methods:
    1. ADQL query
    2. CSV file upload
    3. Datalink

Usage:
    $ python gaia_input.py
"""

import pandas as pd
import matplotlib.pyplot as plt
from astroquery.gaia import Gaia
from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
import astropy.units as u
from .constants import SPTYPE_TEFF_RANGES

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

def find_name_and_binaries(id_result):
    """Pull a common name and any binary/multiple star designations from a SIMBAD ID result.

    Args:
        id_result (astropy.table.Table): Result of Simbad.query_objectids().

    Returns:
        tuple[str or None, list[str]]: Common name (or None if not found), and a list of any
        binary/multiple star designations found.
    """
    common_name = None
    binaries = []
    for row in id_result["id"]:
        if row.startswith("NAME ") and common_name is None:
            common_name = row.replace("NAME ", "")
        if row.startswith("** "):
            binaries.append(row)
    return common_name, binaries

def fetch_coordinates(gaia_ids):
    """Look up RA/Dec (in degrees) for one or more Gaia DR3 source IDs via SIMBAD, for plotting.

    Args:
        gaia_ids (int or list[int]): Gaia DR3 source ID(s).

    Returns:
        list[tuple[str, float, float]]: (label, ra_deg, dec_deg) for each ID successfully resolved
        to a position. Label is the SIMBAD name if available, else the Gaia ID itself.
    """
    if isinstance(gaia_ids, int):
        gaia_ids = [gaia_ids]

    points = []
    if not gaia_ids:
        return points

    try:
        query_names = [f"Gaia DR3 {gid}" for gid in gaia_ids]
        result = Simbad.query_objects(query_names)
    except Exception as e:
        print(f"  (coordinate lookup for plotting failed: {e})")
        return points

    if result is None:
        return points

    for i, row in enumerate(result):
        gid = gaia_ids[i] if i < len(gaia_ids) else None
        try:
            label = str(row["main_id"]) if "main_id" in result.colnames and row["main_id"] else str(gid)

            if "ra" in result.colnames and "dec" in result.colnames:
                ra_val, dec_val = row["ra"], row["dec"]
            else:
                ra_val, dec_val = row["RA"], row["DEC"]

            if isinstance(ra_val, str):
                coord = SkyCoord(ra=ra_val, dec=dec_val, unit=(u.hourangle, u.deg))
            else:
                coord = SkyCoord(ra=ra_val * u.deg, dec=dec_val * u.deg)

            points.append((label, coord.ra.deg, coord.dec.deg))
        except Exception:
            continue

    return points

def plot_positions(points, title, save_plot=False, plot_file_name=None, center=None, radius_deg=None):
    """Plot RA/Dec positions on a simple matplotlib scatter plot.

    Args:
        points (list[tuple[str, float, float]]): (label, ra_deg, dec_deg) for each point to plot.
        title (str): Plot title.
        save_plot (bool, optional): Whether to save the plot to an image file. Defaults to False.
        plot_file_name (str, optional): Name or path for the saved plot image. Defaults to "plot"
            if save_plot is True and no name is given.
        center (tuple[float, float], optional): (ra_deg, dec_deg) center of a search area to mark. Defaults to None.
        radius_deg (float, optional): Search radius in degrees, drawn as a dashed circle around center. Defaults to None.
    """
    if not points:
        print("  (nothing to plot - no coordinates available)")
        return

    try:
        fig, ax = plt.subplots(figsize=(7, 6))

        ras = [p[1] for p in points]
        decs = [p[2] for p in points]
        ax.scatter(ras, decs, c="tab:blue", s=40, zorder=3)

        for label, ra, dec in points:
            ax.annotate(label, (ra, dec), fontsize=8, xytext=(4, 4), textcoords="offset points")

        if center is not None and radius_deg is not None:
            circle = plt.Circle(center, radius_deg, fill=False, edgecolor="tab:red", linestyle="--", zorder=2)
            ax.add_patch(circle)
            ax.scatter([center[0]], [center[1]], marker="x", c="tab:red", s=60, zorder=4, label="search center")
            ax.legend(loc="upper right", fontsize=8)

        ax.set_xlabel("RA (deg)")
        ax.set_ylabel("Dec (deg)")
        ax.set_title(title)
        ax.invert_xaxis()
        plt.tight_layout()

        if save_plot:
            formatted_name = (plot_file_name or "plot").replace(" ", "_")
            if not formatted_name.lower().endswith((".png", ".jpg", ".jpeg", ".pdf", ".svg")):
                formatted_name = f"{formatted_name}.png"
            fig.savefig(formatted_name)
            print(f"  Plot saved to {formatted_name}")

        plt.show()
    except Exception as e:
        print(f"  (plotting skipped: {e})")

def resolve_search_position(ra=None, dec=None, coordinates=None):
    """Resolve an RA/Dec position from either sexagesimal strings or a SkyCoord.

    Args:
        ra (str or float, optional): Right ascension, as a sexagesimal string (e.g. "10h21m00s") when
            paired with a string `dec`. Defaults to None.
        dec (str or float, optional): Declination, as a sexagesimal string (e.g. "+41d05m00s") when
            paired with a string `ra`. Defaults to None.
        coordinates (astropy.coordinates.SkyCoord, optional): Sky position to use if `ra`/`dec` are
            not given. Defaults to None.

    Returns:
        tuple[float, float]: (ra_deg, dec_deg).

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

    return ra, dec

def fetch_common_names(gaia_ids):
    """Look up SIMBAD common names for one or more Gaia DR3 source IDs, in a single batched call.

    Args:
        gaia_ids (list[int]): Gaia DR3 source ID(s).

    Returns:
        dict[int, str]: Maps each Gaia ID that SIMBAD has a record for to its SIMBAD main_id.
        IDs with no SIMBAD record are simply absent from the returned dict.
    """
    names = {}
    if not gaia_ids:
        return names

    try:
        query_names = [f"Gaia DR3 {gid}" for gid in gaia_ids]
        result = Simbad.query_objects(query_names)
    except Exception as e:
        print(f"  (common name lookup failed: {e})")
        return names

    if result is None:
        return names

    for i, row in enumerate(result):
        gid = gaia_ids[i] if i < len(gaia_ids) else None
        if gid is None:
            continue
        try:
            if "main_id" in result.colnames and row["main_id"]:
                names[gid] = str(row["main_id"])
        except Exception:
            continue

    return names

def sanity_check_star(gaia_ids, teff_tolerance=2000):
    """Cross-check Gaia's own data against SIMBAD's spectral type, to flag possible mismatches.

    Compares Gaia's photometric temperature (teff_gspphot) against the temperature range implied
    by SIMBAD's spectral type. A large disagreement can be a sign of a wrong cross-match or 
    another data issue worth double-checking.

    Args:
        gaia_ids (int or list[int]): Gaia DR3 source ID(s) to check.
        teff_tolerance (float, optional): How many Kelvin Gaia's teff_gspphot can fall outside the
            spectral-type-implied range before being flagged. Defaults to 2000 K.

    Returns:
        pandas.DataFrame: One row per ID, with a 'common_name' column plus Gaia and SIMBAD fields
        side by side, and a 'flag' column (empty string if no mismatch was detected).
    """
    if isinstance(gaia_ids, int):
        gaia_ids = [gaia_ids]

    columns_out = ["gaia_id", "common_name", "gaia_g_mag", "gaia_teff", "simbad_sp_type", "simbad_v_mag", "flag"]
    if not gaia_ids:
        return pd.DataFrame(columns=columns_out)

    # One batched Gaia query for photometry + photometric temperature.
    gaia_data = {}
    try:
        id_list = ",".join(str(g) for g in gaia_ids)
        gaia_query = f"""
        SELECT g.source_id, g.phot_g_mean_mag, ap.teff_gspphot
        FROM gaiadr3.gaia_source AS g
        JOIN gaiadr3.astrophysical_parameters AS ap ON g.source_id = ap.source_id
        WHERE g.source_id IN ({id_list})
        """
        job = Gaia.launch_job(gaia_query)
        gaia_df = job.get_results().to_pandas()
        for _, row in gaia_df.iterrows():
            gaia_data[int(row["source_id"])] = (row.get("phot_g_mean_mag"), row.get("teff_gspphot"))
    except Exception as e:
        print(f"  (Gaia lookup for sanity check failed: {e})")

    # One batched SIMBAD query for common name, spectral type, and V magnitude.
    common_names, sp_types, v_mags = {}, {}, {}
    try:
        Simbad.add_votable_fields('sp_type', 'flux(V)')
    except Exception as e:
        print(f"  (could not request SIMBAD sp_type/V-mag fields: {e})")

    try:
        query_names = [f"Gaia DR3 {gid}" for gid in gaia_ids]
        result = Simbad.query_objects(query_names)
    except Exception as e:
        result = None
        print(f"  (SIMBAD lookup for sanity check failed: {e})")

    if result is not None:
        for i, row in enumerate(result):
            gid = gaia_ids[i] if i < len(gaia_ids) else None
            if gid is None:
                continue
            try:
                if "main_id" in result.colnames and row["main_id"]:
                    common_names[gid] = str(row["main_id"])
                if "sp_type" in result.colnames and row["sp_type"]:
                    sp_types[gid] = str(row["sp_type"])
                for v_col in ("flux_V", "FLUX_V"):
                    if v_col in result.colnames:
                        v_val = row[v_col]
                        if v_val is not None and not (hasattr(v_val, "mask") and v_val.mask):
                            v_mags[gid] = float(v_val)
                            break
            except Exception:
                continue

    rows = []
    n_flagged = 0
    for gid in gaia_ids:
        gaia_g_mag, gaia_teff = gaia_data.get(gid, (None, None))
        sp_type = sp_types.get(gid)

        flag = ""
        teff_range = SPTYPE_TEFF_RANGES.get(str(sp_type).strip()[0:1].upper()) if sp_type else None
        if teff_range is not None and gaia_teff is not None and not pd.isna(gaia_teff):
            low, high = teff_range
            if gaia_teff < low - teff_tolerance or gaia_teff > high + teff_tolerance:
                flag = f"Possible mismatch: SIMBAD type '{sp_type}' implies ~{low}-{high} K, Gaia teff_gspphot={gaia_teff:.0f} K"
                n_flagged += 1

        rows.append({
            "gaia_id": gid,
            "common_name": common_names.get(gid),
            "gaia_g_mag": gaia_g_mag,
            "gaia_teff": gaia_teff,
            "simbad_sp_type": sp_type,
            "simbad_v_mag": v_mags.get(gid),
            "flag": flag,
        })

    df = pd.DataFrame(rows, columns=columns_out)
    print(f"Sanity check: {len(df)} source(s) checked, {n_flagged} flagged as possible mismatch(es).")
    if n_flagged > 0:
        for _, r in df[df["flag"] != ""].iterrows():
            print(f"  [{r['gaia_id']}] {r['flag']}")

    return df

def resolve_id(identifier, plot=True, save_plot=False, plot_file_name=None, sanity_check=False):
    """Convert a star or cluster identifier to its Gaia DR3 source ID(s).

    Numeric input is returned as-is and is assumed to be a valid Gaia DR3 source ID. 
    Anything else is looked up via SIMBAD. If the identifier resolves directly to an 
    object with a Gaia DR3 cross-match, that single ID is returned. If no direct cross-match is
    found, SIMBAD's hierarchy is checked for children of the identifier, and the Gaia DR3 IDs 
    of every child that has one are returned as a list.

    Args:
        identifier (int or str): Gaia source ID, or any SIMBAD recognized star or cluster name/identifier.
        plot (bool, optional): Whether to plot the resolved position(s). Defaults to True.
        save_plot (bool, optional): Whether to save the plot to an image file. Defaults to False.
        plot_file_name (str, optional): Name or path for the saved plot image. Defaults to None.
        sanity_check (bool, optional): Whether to cross-check the resolved star(s) against SIMBAD's
            spectral type via sanity_check_star, and print any flagged mismatches. Adds 2 extra
            network calls when True. Defaults to False.

    Returns:
        int or list[int] or None: A single Gaia DR3 source ID if the identifier resolves directly, 
        a list of Gaia DR3 source IDs if it resolves to multiple children, or None if nothing
        could be resolved.
    """
    if str(identifier).strip().isdigit():
        gaia_id = int(identifier)
        id_result = Simbad.query_objectids(f"Gaia DR3 {gaia_id}")
        if id_result is not None:
            common_name, binaries = find_name_and_binaries(id_result)
            if common_name:
                print(f"Resolved: {gaia_id} (numeric ID, used as-is) -> common name: {common_name}")
            else:
                print(f"Resolved: {gaia_id} (numeric ID, used as-is)")
            if binaries:
                print(f"  Binary/multiple star designation(s) found: {binaries}")
        else:
            print(f"Resolved: {gaia_id} (numeric ID, used as-is)")

        if plot:
            points = fetch_coordinates(gaia_id)
            plot_positions(points, title=f"Cross-match position for {gaia_id}", save_plot=save_plot, plot_file_name=plot_file_name)

        if sanity_check:
            sanity_check_star(gaia_id)

        return gaia_id

    result = Simbad.query_objectids(identifier)
    if result is not None:
        common_name, binaries = find_name_and_binaries(result)
        for row in result["id"]:
            if row.startswith("Gaia DR3 "):
                gaia_id = int(row.replace("Gaia DR3 ", ""))
                if common_name and common_name != identifier:
                    print(f"Resolved: '{identifier}' ({common_name}) -> Gaia DR3 {gaia_id}")
                else:
                    print(f"Resolved: '{identifier}' -> Gaia DR3 {gaia_id}")
                if binaries:
                    print(f"  Binary/multiple star designation(s) found: {binaries}")

                if plot:
                    points = fetch_coordinates(gaia_id)
                    plot_positions(points, title=f"Cross-match position for '{identifier}'", save_plot=save_plot, plot_file_name=plot_file_name)

                if sanity_check:
                    sanity_check_star(gaia_id)

                return gaia_id

    children = Simbad.query_hierarchy(identifier, hierarchy="children")
    if children is None or len(children) == 0:
        print(f"Could not resolve '{identifier}'.")
        return None

    resolved_ids = []
    resolved_lines = []
    unresolved_children = []
    for child_name in children["main_id"]:
        child_ids = Simbad.query_objectids(child_name)
        if child_ids is None:
            unresolved_children.append(str(child_name))
            continue

        gaia_id = None
        for row in child_ids["id"]:
            if row.startswith("Gaia DR3 "):
                gaia_id = int(row.replace("Gaia DR3 ", ""))
                break

        if gaia_id is None:
            unresolved_children.append(str(child_name))
            continue

        resolved_ids.append(gaia_id)
        common_name, binaries = find_name_and_binaries(child_ids)
        if common_name and common_name != str(child_name):
            line = f"{child_name} ({common_name}) -> Gaia DR3 {gaia_id}"
        else:
            line = f"{child_name} -> Gaia DR3 {gaia_id}"
        if binaries:
            line += f"  [binary/multiple star designation(s): {binaries}]"
        resolved_lines.append(line)

    if not resolved_ids:
        print(f"'{identifier}' has no Gaia DR3 cross-match, and none of its children do either.")
        return None

    print(f"Resolved {len(resolved_ids)}/{len(children['main_id'])} child(ren) of '{identifier}':")
    for line in resolved_lines:
        print(f"  {line}")
    if unresolved_children:
        print(f"Could not resolve {len(unresolved_children)} child(ren) of '{identifier}': {unresolved_children}")

    if plot:
        points = fetch_coordinates(resolved_ids)
        plot_positions(points, title=f"Cross-match positions for children of '{identifier}'", save_plot=save_plot, plot_file_name=plot_file_name)

    if sanity_check:
        sanity_check_star(resolved_ids)

    return resolved_ids

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

def find_star(
    ra: str | float = None,
    dec: str | float = None,
    coordinates: SkyCoord = None,
    columns: str = '*',
    save_file: bool = False,
    file_name: str = "star_query",
    degree_range: float = 0.0001,
    mag_min: float = None,
    mag_max: float = None,
    mag_band: str = "phot_g_mean_mag",
    dist_min: float = None,
    dist_max: float = None,
    plot: bool = True,
    save_plot: bool = False,
    plot_file_name: str = None
):
    """Query Gaia for sources within a small radius of a sky position.

    The position can be supplied either as sexagesimal RA/Dec strings, or as an astropy SkyCoord. Exactly one 
    of these input forms must resolve to a usable position, or a ValueError is raised.

    Optionally, results can be filtered by magnitude and/or distance. Both are hard filters applied directly
    in the ADQL query, regardless of whether the filtered column is included in 'columns'.

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
        mag_min (float, optional): Faintest magnitude to include. Defaults to None (no lower cutoff).
        mag_max (float, optional): Brightest magnitude to include. Defaults to None (no upper cutoff).
        mag_band (str, optional): Gaia magnitude column to filter on. Defaults to "phot_g_mean_mag".
        dist_min (float, optional): Nearest distance to include, in parsecs. Defaults to None (no lower cutoff).
        dist_max (float, optional): Farthest distance to include, in parsecs. Defaults to None (no upper cutoff).
        plot (bool, optional): Whether to plot the found source(s). Defaults to True.
        save_plot (bool, optional): Whether to save the plot to an image file. Defaults to False.
        plot_file_name (str, optional): Name or path for the saved plot image. Defaults to None.

    Returns:
        pandas.DataFrame: Query results, with an added 'common_name' column from SIMBAD (blank for
        sources SIMBAD has no record of).

    Raises:
        ValueError: If neither a valid ra/dec pair nor coordinates is given.
    """
    ra, dec = resolve_search_position(ra, dec, coordinates)

    conditions = []
    if mag_min is not None:
        conditions.append(f"{mag_band} >= {mag_min}")
    if mag_max is not None:
        conditions.append(f"{mag_band} <= {mag_max}")
    if dist_max is not None and dist_max > 0:
        conditions.append(f"parallax >= {1000 / dist_max}")
    if dist_min is not None and dist_min > 0:
        conditions.append(f"parallax <= {1000 / dist_min}")

    extra_filter = ""
    if conditions:
        extra_filter = " AND " + " AND ".join(conditions)
        print(f"Filter: {' AND '.join(conditions)}")

    query = f"""
    SELECT TOP 10 {columns}
    FROM gaiadr3.gaia_source
    WHERE CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', {ra}, {dec}, {degree_range})
    ) = 1{extra_filter}
    """
    if save_file:
        df = query_by_adql(query, save_file=True, file_name=file_name)
    else:
        df = query_by_adql(query)

    if df is not None and "source_id" in df.columns:
        common_names = fetch_common_names(df["source_id"].tolist())
        df["common_name"] = df["source_id"].map(common_names)

    if df is not None:
        print(f"Found {len(df)} source(s) within {degree_range} deg.")
        if "non_single_star" in df.columns and "source_id" in df.columns:
            flagged = df[df["non_single_star"] > 0]
            if len(flagged) > 0:
                print(f"  {len(flagged)} source(s) flagged as non-single-star (binary/multiple) system(s):")
                for _, row in flagged.iterrows():
                    flags = int(row["non_single_star"])
                    types = []
                    if flags & 1:
                        types.append("astrometric")
                    if flags & 2:
                        types.append("spectroscopic")
                    if flags & 4:
                        types.append("eclipsing")
                    print(f"    source_id {row['source_id']}: {', '.join(types) if types else f'flag={flags}'}")

    if plot and df is not None and "ra" in df.columns and "dec" in df.columns:
        points = [(str(row.get("source_id", i)), row["ra"], row["dec"]) for i, row in df.iterrows()]
        plot_positions(
            points,
            title=f"Cone search near (ra={ra:.4f}, dec={dec:.4f})",
            save_plot=save_plot,
            plot_file_name=plot_file_name,
            center=(ra, dec),
            radius_deg=degree_range,
        )

    return df

def find_cluster(
    ra: str | float = None,
    dec: str | float = None,
    coordinates: SkyCoord = None,
    distance: float = None,
    mag_min: float = None,
    mag_max: float = None,
    mag_band: str = "V",
    search_radius_deg: float = 2.0,
    top_n: int = 10,
    save_file: bool = False,
    file_name: str = "cluster_query",
    plot: bool = True,
    save_plot: bool = False,
    plot_file_name: str = None
):
    """Find cluster-type objects near a sky position, using SIMBAD.

    The position can be supplied either as sexagesimal RA/Dec strings, or as an astropy SkyCoord, same as
    find_star. Candidates are ranked by angular separation from the given position, not by distance,
    since many SIMBAD cluster records have no parallax on file. If a candidate does have a parallax, its
    implied distance is shown for context and compared against 'distance' if given, but this is not used
    to include/exclude/rank a candidate.

    Magnitude filtering is a soft filter: a candidate is only excluded if it has magnitude data on file AND
    that data is outside the given range. Candidates with no magnitude data are kept, since SIMBAD's cluster
    records are inconsistently populated for this field.

    Args:
        ra (str or float, optional): Right ascension, as a sexagesimal string (e.g. "10h21m00s") when paired
            with a string `dec`. Defaults to None.
        dec (str or float, optional): Declination, as a sexagesimal string (e.g. "+41d05m00s") when paired
            with a string `ra`. Defaults to None.
        coordinates (astropy.coordinates.SkyCoord, optional): Sky position to search around, used if `ra`/`dec`
            are not given. Defaults to None.
        distance (float, optional): Distance to compare against, in parsecs, shown for informational
            comparison only. Defaults to None (no comparison shown).
        mag_min (float, optional): Faintest magnitude to include. Defaults to None (no lower cutoff).
        mag_max (float, optional): Brightest magnitude to include. Defaults to None (no upper cutoff).
        mag_band (str, optional): SIMBAD photometric band to filter on. Defaults to "V".
        search_radius_deg (float, optional): Search radius in degrees. Defaults to 2.0.
        top_n (int, optional): Maximum number of candidates to return, ranked by angular separation.
            Defaults to 10.
        save_file (bool, optional): Whether to save the result to a CSV file. Defaults to False.
        file_name (str, optional): Name for the saved CSV file. Defaults to "cluster_query".
        plot (bool, optional): Whether to plot the returned candidate(s). Defaults to True.
        save_plot (bool, optional): Whether to save the plot to an image file. Defaults to False.
        plot_file_name (str, optional): Name or path for the saved plot image. Defaults to None.

    Returns:
        pandas.DataFrame: Up to 'top_n' candidate clusters, sorted by angular separation, with columns
        'name', 'ra', 'dec', 'otype', 'separation_deg', 'plx_value', 'implied_distance_pc',
        'distance_diff_pc', and 'mag'. Empty (but same columns) if nothing is found.

    Raises:
        ValueError: If neither a valid ra/dec pair nor coordinates is given.
    """
    ra, dec = resolve_search_position(ra, dec, coordinates)

    columns_out = [
        "name", "ra", "dec", "otype", "separation_deg",
        "plx_value", "implied_distance_pc", "distance_diff_pc", "mag",
    ]

    Simbad.add_votable_fields('otype', 'parallax')
    if mag_min is not None or mag_max is not None:
        try:
            Simbad.add_votable_fields(f'flux({mag_band})')
        except Exception as e:
            print(f"  (could not request magnitude field '{mag_band}', filter will be skipped: {e})")

    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    result = Simbad.query_region(coord, radius=search_radius_deg * u.deg)

    if result is None:
        print(f"Cluster match: no SIMBAD objects found within {search_radius_deg} deg of (ra={ra}, dec={dec}).")
        return pd.DataFrame(columns=columns_out)

    candidates = []
    checked = 0
    for row in result:
        if not str(row["otype"]).startswith(("Cl*", "OpC", "GlC", "As*")):
            continue

        try:
            if "ra" in result.colnames and "dec" in result.colnames:
                row_ra, row_dec = row["ra"], row["dec"]
            else:
                row_ra, row_dec = row["RA"], row["DEC"]
            if isinstance(row_ra, str):
                row_coord = SkyCoord(ra=row_ra, dec=row_dec, unit=(u.hourangle, u.deg))
            else:
                row_coord = SkyCoord(ra=row_ra * u.deg, dec=row_dec * u.deg)
        except Exception:
            continue

        mag_val = None
        if mag_min is not None or mag_max is not None:
            for mag_col in (f"flux_{mag_band}", f"FLUX_{mag_band}", mag_band.lower(), mag_band.upper()):
                if mag_col in result.colnames:
                    try:
                        candidate_val = row[mag_col]
                        if candidate_val is not None and not (hasattr(candidate_val, "mask") and candidate_val.mask):
                            mag_val = float(candidate_val)
                            break
                    except Exception:
                        continue
            # Soft filter: only exclude when magnitude data IS present and out of range.
            if mag_val is not None:
                if mag_min is not None and mag_val < mag_min:
                    continue
                if mag_max is not None and mag_val > mag_max:
                    continue

        checked += 1
        sep = coord.separation(row_coord).deg

        plx = row["plx_value"]
        plx_val = float(plx) if plx and plx > 0 else None
        implied_distance = 1000 / plx_val if plx_val else None
        distance_diff = (
            abs(implied_distance - distance) if (implied_distance is not None and distance is not None) else None
        )

        candidates.append({
            "name": str(row["main_id"]),
            "ra": row_coord.ra.deg,
            "dec": row_coord.dec.deg,
            "otype": str(row["otype"]),
            "separation_deg": sep,
            "plx_value": plx_val,
            "implied_distance_pc": implied_distance,
            "distance_diff_pc": distance_diff,
            "mag": mag_val,
        })

    if not candidates:
        print(f"Cluster match: checked {checked} candidate(s) near (ra={ra}, dec={dec}); no match found.")
        return pd.DataFrame(columns=columns_out)

    df = pd.DataFrame(candidates).sort_values("separation_deg").head(top_n).reset_index(drop=True)

    print(
        f"Cluster match: checked {checked} candidate(s) near (ra={ra}, dec={dec}); "
        f"returning top {len(df)} by angular separation."
    )

    if save_file:
        formatted_name = file_name.replace(" ", "_")
        formatted_name = f"{formatted_name}.csv"
        df.to_csv(formatted_name)

    if plot and "ra" in df.columns and "dec" in df.columns:
        points = [(row["name"], row["ra"], row["dec"]) for _, row in df.iterrows()]
        plot_positions(
            points,
            title=f"Candidate clusters near (ra={ra}, dec={dec})",
            save_plot=save_plot,
            plot_file_name=plot_file_name,
            center=(ra, dec),
            radius_deg=search_radius_deg,
        )

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