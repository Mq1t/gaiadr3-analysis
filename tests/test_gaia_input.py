"""Tests for gaia_input.py

Tests cover CSV loading, filtering logic, identifier resolution via SIMBAD, cluster lookup, ADQL/Datalink querying, 
coordinate-based star search, login handling, and the interactive menu's routing logic.

All calls to SIMBAD and the Gaia archive (via astroquery) are mocked, so this runs fully offline and does not require 
network access or live credentials. Interactive input prompts are also mocked with scripted answers so the menu 
logic can be tested.
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from gaiadr3_analysis.gaia_input import (
    gaia_login_prompt,
    resolve_id,
    find_cluster,
    query_by_adql,
    query_by_datalink,
    find_star,
    load_csv,
    apply_filter,
    get_dataframe,
)

# A 26-character filler used to build fake Datalink keys for tests.
KEY_PREFIX = "A" * 26

# Fixtures
@pytest.fixture
def sample_df():
    """Returns a minimal Gaia DataFrame for testing.

    Returns:
        pd.DataFrame: DataFrame with ra, dec, parallax, and pmra columns.
    """
    return pd.DataFrame({
        "ra": [10.0, 20.0, 30.0],
        "dec": [-5.0, 15.0, 25.0],
        "parallax": [0.5, 1.5, 2.5],
        "pmra": [0.1, 0.2, 0.3],
    })

@pytest.fixture
def sample_dict_of_df():
    """Returns a dict of DataFrames mimicking query_by_datalink output.

    Returns:
        dict[int, pd.DataFrame]: Mapping of fake Gaia IDs to epoch
        photometry DataFrames.
    """
    return {
        111: pd.DataFrame({"flux": [1.0, 5.0, 10.0]}),
        222: pd.DataFrame({"flux": [2.0, 6.0, 11.0]}),
    }

# load_csv
def test_load_csv_returns_dataframe(tmp_path):
    """Checks that load_csv returns a pandas DataFrame.

    Args:
        tmp_path: pytest built-in fixture providing a temporary directory.
    """
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("ra,dec\n10.0,-5.0\n20.0,15.0\n")
    result = load_csv(str(csv_file))
    assert isinstance(result, pd.DataFrame)

def test_load_csv_correct_columns(tmp_path):
    """Checks that load_csv preserves column names from the CSV file.

    Args:
        tmp_path: pytest built-in fixture providing a temporary directory.
    """
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("ra,dec,parallax\n10.0,-5.0,1.0\n")
    result = load_csv(str(csv_file))
    assert list(result.columns) == ["ra", "dec", "parallax"]

def test_load_csv_correct_row_count(tmp_path):
    """Checks that load_csv loads the correct number of rows.

    Args:
        tmp_path: pytest built-in fixture providing a temporary directory.
    """
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("ra,dec\n10.0,-5.0\n20.0,15.0\n30.0,25.0\n")
    result = load_csv(str(csv_file))
    assert len(result) == 3

def test_load_csv_file_not_found():
    """Checks that load_csv raises FileNotFoundError for a missing file."""
    with pytest.raises(FileNotFoundError):
        load_csv("nonexistent_file.csv")

# apply_filter (single DataFrame)
def test_apply_filter_valid_expression(sample_df):
    """Checks that apply_filter correctly filters rows using a valid expression.

    Args:
        sample_df (pd.DataFrame): Sample Gaia DataFrame fixture.
    """
    with patch("builtins.print"), patch("builtins.input", return_value="parallax > 1.0"):
        result = apply_filter(sample_df)
    assert len(result) == 2
    assert all(result["parallax"] > 1.0)

def test_apply_filter_empty_expression_returns_original(sample_df):
    """Checks that apply_filter returns the original DataFrame when skipped.

    Args:
        sample_df (pd.DataFrame): Sample Gaia DataFrame fixture.
    """
    with patch("builtins.print"), patch("builtins.input", return_value=""):
        result = apply_filter(sample_df)
    assert len(result) == len(sample_df)

def test_apply_filter_invalid_expression_returns_original(sample_df):
    """Checks that apply_filter returns the original DataFrame on invalid input.

    Args:
        sample_df (pd.DataFrame): Sample Gaia DataFrame fixture.
    """
    with patch("builtins.print"), patch("builtins.input", return_value="not_a_column > 1"):
        result = apply_filter(sample_df)
    assert len(result) == len(sample_df)

def test_apply_filter_exact_match(sample_df):
    """Checks that apply_filter returns only rows matching an exact condition.

    Args:
        sample_df (pd.DataFrame): Sample Gaia DataFrame fixture.
    """
    with patch("builtins.print"), patch("builtins.input", return_value="parallax == 0.5"):
        result = apply_filter(sample_df)
    assert len(result) == 1
    assert result["parallax"].iloc[0] == 0.5

# apply_filter (dict of DataFrames)
def test_apply_filter_dict_applies_to_every_entry(sample_dict_of_df):
    """Checks that apply_filter applies the same expression to every DataFrame in a dict.

    Args:
        sample_dict_of_df (dict[int, pd.DataFrame]): Sample dict-of-DataFrames fixture.
    """
    with patch("builtins.print"), patch("builtins.input", return_value="flux > 4.0"):
        result = apply_filter(sample_dict_of_df)
    assert set(result.keys()) == {111, 222}
    assert len(result[111]) == 2
    assert len(result[222]) == 2
    assert (result[111]["flux"] > 4.0).all()

def test_apply_filter_dict_empty_expression_returns_original(sample_dict_of_df):
    """Checks that apply_filter returns the original dict unchanged when the filter is skipped.

    Args:
        sample_dict_of_df (dict[int, pd.DataFrame]): Sample dict-of-DataFrames fixture.
    """
    with patch("builtins.print"), patch("builtins.input", return_value=""):
        result = apply_filter(sample_dict_of_df)
    assert result is sample_dict_of_df

def test_apply_filter_dict_invalid_expression_keeps_original_entry(sample_dict_of_df):
    """Checks that apply_filter keeps the original DataFrame for an entry when its expression is invalid.

    Args:
        sample_dict_of_df (dict[int, pd.DataFrame]): Sample dict-of-DataFrames fixture.
    """
    with patch("builtins.print"), patch("builtins.input", return_value="not_a_column > 1"):
        result = apply_filter(sample_dict_of_df)
    assert len(result[111]) == len(sample_dict_of_df[111])

def test_apply_filter_empty_dict_returns_empty_dict():
    """Checks that apply_filter returns an empty dict unchanged without prompting for input."""
    result = apply_filter({})
    assert result == {}

# resolve_id
def test_resolve_id_numeric_string_returns_int():
    """Checks that a purely numeric string identifier is returned as an int without querying SIMBAD."""
    result = resolve_id("123456789")
    assert result == 123456789

def test_resolve_id_int_input_returns_int():
    """Checks that an integer identifier is returned as-is."""
    result = resolve_id(987654321)
    assert result == 987654321

@patch("gaiadr3_analysis.gaia_input.Simbad")
def test_resolve_id_direct_cross_match(mock_simbad):
    """Checks that a name with a direct Gaia DR3 cross-match resolves to a single int.

    Args:
        mock_simbad: Mocked Simbad class from astroquery.
    """
    mock_simbad.query_objectids.return_value = {"id": ["HD 209458", "Gaia DR3 1234567890123456789"]}
    result = resolve_id("HD 209458")
    assert result == 1234567890123456789

@patch("gaiadr3_analysis.gaia_input.Simbad")
def test_resolve_id_no_match_at_all_returns_none(mock_simbad):
    """Checks that resolve_id returns None when neither a direct match nor children are found.

    Args:
        mock_simbad: Mocked Simbad class from astroquery.
    """
    mock_simbad.query_objectids.return_value = None
    mock_simbad.query_hierarchy.return_value = None
    with patch("builtins.print"):
        result = resolve_id("Nonexistent Object")
    assert result is None

@patch("gaiadr3_analysis.gaia_input.Simbad")
def test_resolve_id_resolves_children_to_list(mock_simbad):
    """Checks that an identifier with no direct match but resolvable children returns a list of IDs.

    Args:
        mock_simbad: Mocked Simbad class from astroquery.
    """
    mock_simbad.query_objectids.side_effect = [
        None,
        {"id": ["Gaia DR3 111"]},
        {"id": ["Gaia DR3 222"]},
    ]
    mock_simbad.query_hierarchy.return_value = {"main_id": ["Star A", "Star B"]}

    result = resolve_id("NGC 188")

    assert result == [111, 222]

@patch("gaiadr3_analysis.gaia_input.Simbad")
def test_resolve_id_children_found_but_none_have_gaia_id(mock_simbad):
    """Checks that resolve_id returns None when children exist but none have a Gaia DR3 cross-match.

    Args:
        mock_simbad: Mocked Simbad class from astroquery.
    """
    mock_simbad.query_objectids.side_effect = [None, {"id": ["HD 1"]}, {"id": ["HD 2"]}]
    mock_simbad.query_hierarchy.return_value = {"main_id": ["Star A", "Star B"]}

    with patch("builtins.print"):
        result = resolve_id("Cluster")

    assert result is None

@patch("gaiadr3_analysis.gaia_input.Simbad")
def test_resolve_id_empty_children_table_returns_none(mock_simbad):
    """Checks that resolve_id returns None when query_hierarchy returns an empty table.

    Args:
        mock_simbad: Mocked Simbad class from astroquery.
    """
    mock_simbad.query_objectids.return_value = None
    mock_simbad.query_hierarchy.return_value = []

    with patch("builtins.print"):
        result = resolve_id("Empty Object")

    assert result is None

# find_cluster
@patch("gaiadr3_analysis.gaia_input.Simbad")
def test_find_cluster_returns_none_when_no_region_results(mock_simbad):
    """Checks that find_cluster returns None when query_region finds nothing nearby.

    Args:
        mock_simbad: Mocked Simbad class from astroquery.
    """
    mock_simbad.query_region.return_value = None
    result = find_cluster(ra=10.0, dec=20.0, distance=100.0)
    assert result is None

@patch("gaiadr3_analysis.gaia_input.Simbad")
def test_find_cluster_picks_closest_distance_match(mock_simbad):
    """Checks that find_cluster returns the cluster whose distance best matches the target.

    Args:
        mock_simbad: Mocked Simbad class from astroquery.
    """
    mock_simbad.query_region.return_value = [
        {"otype": "OpC", "plx_value": 10.0, "main_id": "Cluster A"},  # 100 pc
        {"otype": "OpC", "plx_value": 5.0, "main_id": "Cluster B"},   # 200 pc
    ]

    result = find_cluster(ra=10.0, dec=20.0, distance=190.0)
    assert result == "Cluster B"

@patch("gaiadr3_analysis.gaia_input.Simbad")
def test_find_cluster_skips_non_cluster_otypes(mock_simbad):
    """Checks that find_cluster ignores rows whose otype isn't a recognized cluster type.

    Args:
        mock_simbad: Mocked Simbad class from astroquery.
    """
    mock_simbad.query_region.return_value = [
        {"otype": "Star", "plx_value": 10.0, "main_id": "Not A Cluster"},
    ]

    result = find_cluster(ra=10.0, dec=20.0, distance=100.0)
    assert result is None

@patch("gaiadr3_analysis.gaia_input.Simbad")
def test_find_cluster_skips_nonpositive_parallax(mock_simbad):
    """Checks that find_cluster ignores rows with zero or negative parallax.

    Args:
        mock_simbad: Mocked Simbad class from astroquery.
    """
    mock_simbad.query_region.return_value = [
        {"otype": "GlC", "plx_value": -1.0, "main_id": "Bad Cluster"},
        {"otype": "GlC", "plx_value": 0.0, "main_id": "Zero Cluster"},
    ]

    result = find_cluster(ra=10.0, dec=20.0, distance=100.0)
    assert result is None

# query_by_adql
def test_query_by_adql_raises_value_error_without_query_or_identifier():
    """Checks that query_by_adql raises ValueError when neither adql_query nor identifier is given."""
    with pytest.raises(ValueError):
        query_by_adql()

@patch("gaiadr3_analysis.gaia_input.Gaia")
def test_query_by_adql_runs_raw_query(mock_gaia):
    """Checks that query_by_adql runs a provided ADQL query and returns a DataFrame.

    Args:
        mock_gaia: Mocked Gaia class from astroquery.
    """
    mock_job = MagicMock()
    mock_job.get_results.return_value.to_pandas.return_value = pd.DataFrame({"source_id": [1, 2]})
    mock_gaia.launch_job.return_value = mock_job

    result = query_by_adql("SELECT TOP 2 source_id FROM gaiadr3.gaia_source")
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2

@patch("gaiadr3_analysis.gaia_input.resolve_id")
@patch("gaiadr3_analysis.gaia_input.Gaia")
def test_query_by_adql_builds_default_query_from_identifier(mock_gaia, mock_resolve):
    """Checks that query_by_adql builds a default SELECT * query when only an identifier is given.

    Args:
        mock_gaia: Mocked Gaia class from astroquery.
        mock_resolve: Mocked resolve_id function.
    """
    mock_resolve.return_value = 123456789
    mock_job = MagicMock()
    mock_job.get_results.return_value.to_pandas.return_value = pd.DataFrame({"source_id": [123456789]})
    mock_gaia.launch_job.return_value = mock_job

    query_by_adql(identifier="HD 209458")
    called_query = mock_gaia.launch_job.call_args[0][0]
    assert "IN (123456789)" in called_query

@patch("gaiadr3_analysis.gaia_input.resolve_id")
def test_query_by_adql_returns_none_for_unresolved_identifier(mock_resolve):
    """Checks that query_by_adql returns None and doesn't run a query if the identifier can't be resolved.

    Args:
        mock_resolve: Mocked resolve_id function.
    """
    mock_resolve.return_value = None
    with patch("builtins.print"):
        result = query_by_adql(identifier="Unresolvable Star")
    assert result is None

@patch("gaiadr3_analysis.gaia_input.resolve_id")
@patch("gaiadr3_analysis.gaia_input.Gaia")
def test_query_by_adql_multiple_ids_joined_with_in_clause(mock_gaia, mock_resolve):
    """Checks that a list of resolved IDs is substituted as a comma-separated IN clause.

    Args:
        mock_gaia: Mocked Gaia class from astroquery.
        mock_resolve: Mocked resolve_id function.
    """
    mock_resolve.return_value = [111, 222, 333]
    mock_job = MagicMock()
    mock_job.get_results.return_value.to_pandas.return_value = pd.DataFrame({"source_id": [111, 222, 333]})
    mock_gaia.launch_job.return_value = mock_job

    query_by_adql(identifier="Some Cluster")

    called_query = mock_gaia.launch_job.call_args[0][0]
    assert "IN (111,222,333)" in called_query

@patch("gaiadr3_analysis.gaia_input.Gaia")
def test_query_by_adql_saves_to_default_filename_when_none_given(mock_gaia):
    """Checks that query_by_adql falls back to a default filename when saving without an identifier.

    Args:
        mock_gaia: Mocked Gaia class from astroquery.
    """
    mock_job = MagicMock()
    mock_job.get_results.return_value.to_pandas.return_value = pd.DataFrame({"source_id": [1]})
    mock_gaia.launch_job.return_value = mock_job

    with patch("pandas.DataFrame.to_csv") as mock_to_csv:
        query_by_adql("SELECT TOP 1 source_id FROM gaiadr3.gaia_source", save_file=True)

    mock_to_csv.assert_called_once_with("gaia_query.csv")

# query_by_datalink
def test_query_by_datalink_raises_type_error_for_bad_folder_name():
    """Checks that query_by_datalink raises TypeError when save_file is True and folder_name isn't a string."""
    with pytest.raises(TypeError):
        query_by_datalink(123456789, save_file=True, folder_name=None)

@patch("gaiadr3_analysis.gaia_input.resolve_id")
def test_query_by_datalink_returns_empty_dict_when_nothing_resolves(mock_resolve):
    """Checks that query_by_datalink returns an empty dict if no IDs could be resolved.

    Args:
        mock_resolve: Mocked resolve_id function.
    """
    mock_resolve.return_value = None

    with patch("builtins.print"):
        result = query_by_datalink(["Unresolvable Star"])

    assert result == {}

@patch("gaiadr3_analysis.gaia_input.resolve_id")
@patch("gaiadr3_analysis.gaia_input.Gaia")
def test_query_by_datalink_flattens_list_results_from_resolve_id(mock_gaia, mock_resolve):
    """Checks that a resolve_id call returning a list of children IDs is flattened into the query.

    Args:
        mock_gaia: Mocked Gaia class from astroquery.
        mock_resolve: Mocked resolve_id function.
    """
    mock_resolve.return_value = [111, 222]
    mock_table = MagicMock()
    mock_table.to_table.return_value.to_pandas.return_value = pd.DataFrame({"flux": [1.0]})
    mock_gaia.load_data.return_value = {
        f"{KEY_PREFIX}111.xml": [mock_table],
        f"{KEY_PREFIX}222.xml": [mock_table],
    }

    result = query_by_datalink("Some Cluster")

    called_ids = mock_gaia.load_data.call_args.kwargs["ids"]
    assert called_ids == [111, 222]
    assert set(result.keys()) == {111, 222}

@patch("gaiadr3_analysis.gaia_input.resolve_id")
@patch("gaiadr3_analysis.gaia_input.Gaia")
def test_query_by_datalink_reports_unretrieved_ids(mock_gaia, mock_resolve):
    """Checks that IDs with no epoch photometry data are reported as not retrieved.

    Args:
        mock_gaia: Mocked Gaia class from astroquery.
        mock_resolve: Mocked resolve_id function.
    """
    mock_resolve.side_effect = [111, 222]
    mock_table = MagicMock()
    mock_table.to_table.return_value.to_pandas.return_value = pd.DataFrame({"flux": [1.0]})
    mock_gaia.load_data.return_value = {f"{KEY_PREFIX}111.xml": [mock_table]}

    with patch("builtins.print") as mock_print:
        result = query_by_datalink([111, 222])

    assert 111 in result
    assert 222 not in result
    printed = " ".join(str(call.args) for call in mock_print.call_args_list)
    assert "222" in printed

@patch("gaiadr3_analysis.gaia_input.resolve_id")
@patch("gaiadr3_analysis.gaia_input.Gaia")
def test_query_by_datalink_saves_csv_per_star(mock_gaia, mock_resolve):
    """Checks that query_by_datalink writes one CSV file per resolved star when save_file is True.

    Args:
        mock_gaia: Mocked Gaia class from astroquery.
        mock_resolve: Mocked resolve_id function.
    """
    mock_resolve.return_value = 111
    mock_table = MagicMock()
    mock_table.to_table.return_value.to_pandas.return_value = pd.DataFrame({"flux": [1.0]})
    mock_gaia.load_data.return_value = {f"{KEY_PREFIX}111.xml": [mock_table]}

    with patch("pandas.DataFrame.to_csv") as mock_to_csv:
        query_by_datalink(111, save_file=True, folder_name="output")

    mock_to_csv.assert_called_once_with("output/111.csv")

# find_star
def test_find_star_raises_value_error_without_position():
    """Checks that find_star raises ValueError when no position is given at all."""
    with pytest.raises(ValueError):
        find_star()

@patch("gaiadr3_analysis.gaia_input.query_by_adql")
def test_find_star_runs_with_string_ra_dec(mock_query):
    """Checks that find_star converts sexagesimal ra/dec strings and runs a query.

    Args:
        mock_query: Mocked query_by_adql function.
    """
    mock_query.return_value = pd.DataFrame({"source_id": [1]})

    result = find_star(ra="06h45m08.9s", dec="-16d42m58s")

    assert mock_query.called
    assert isinstance(result, pd.DataFrame)

@patch("gaiadr3_analysis.gaia_input.query_by_adql")
def test_find_star_runs_with_skycoord(mock_query):
    """Checks that find_star accepts a SkyCoord object directly.

    Args:
        mock_query: Mocked query_by_adql function.
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    mock_query.return_value = pd.DataFrame({"source_id": [1]})
    coord = SkyCoord(ra=101.287 * u.deg, dec=-16.716 * u.deg)

    result = find_star(coordinates=coord)

    assert mock_query.called
    assert isinstance(result, pd.DataFrame)

@patch("gaiadr3_analysis.gaia_input.query_by_adql")
def test_find_star_passes_save_file_through(mock_query):
    """Checks that find_star forwards save_file and file_name to query_by_adql.

    Args:
        mock_query: Mocked query_by_adql function.
    """
    mock_query.return_value = pd.DataFrame({"source_id": [1]})

    find_star(ra="06h45m08.9s", dec="-16d42m58s", save_file=True, file_name="my_star")

    _, kwargs = mock_query.call_args
    assert kwargs.get("save_file") is True
    assert kwargs.get("file_name") == "my_star"

# gaia_login_prompt
def test_gaia_login_prompt_skips_login_on_no():
    """Checks that gaia_login_prompt does not attempt login when the user declines."""
    with patch("builtins.input", return_value="n"), patch("gaiadr3_analysis.gaia_input.Gaia") as mock_gaia:
        gaia_login_prompt()
    mock_gaia.login.assert_not_called()

@patch("gaiadr3_analysis.gaia_input.Gaia")
def test_gaia_login_prompt_logs_in_on_yes(mock_gaia):
    """Checks that gaia_login_prompt calls Gaia.login with the entered credentials.

    Args:
        mock_gaia: Mocked Gaia class from astroquery.
    """
    with patch("builtins.input", side_effect=["y", "test_user", "test_pass"]), patch("builtins.print"):
        gaia_login_prompt()

    mock_gaia.login.assert_called_once_with(user="test_user", password="test_pass")

@patch("gaiadr3_analysis.gaia_input.Gaia")
def test_gaia_login_prompt_handles_login_failure(mock_gaia):
    """Checks that gaia_login_prompt catches and reports a failed login attempt.

    Args:
        mock_gaia: Mocked Gaia class from astroquery.
    """
    mock_gaia.login.side_effect = Exception("bad credentials")

    with patch("builtins.input", side_effect=["y", "test_user", "wrong_pass"]), patch("builtins.print") as mock_print:
        gaia_login_prompt()

    assert any("Login failed" in str(call.args) for call in mock_print.call_args_list)

# get_dataframe
@patch("gaiadr3_analysis.gaia_input.query_by_adql")
@patch("gaiadr3_analysis.gaia_input.gaia_login_prompt")
def test_get_dataframe_adql_identifier_path(mock_login, mock_query):
    """Checks that choosing the ADQL/identifier menu path calls query_by_adql with the right args.

    Args:
        mock_login: Mocked gaia_login_prompt function.
        mock_query: Mocked query_by_adql function.
    """
    mock_query.return_value = pd.DataFrame({"source_id": [1]})
    inputs = iter(["1", "y", "HD 209458", "n"])

    with patch("builtins.input", lambda *_: next(inputs)):
        result = get_dataframe()

    mock_query.assert_called_once_with(None, identifier="HD 209458")
    assert isinstance(result, pd.DataFrame)

@patch("gaiadr3_analysis.gaia_input.load_csv")
@patch("gaiadr3_analysis.gaia_input.gaia_login_prompt")
def test_get_dataframe_csv_path(mock_login, mock_load_csv):
    """Checks that choosing the CSV menu option calls load_csv with the given path.

    Args:
        mock_login: Mocked gaia_login_prompt function.
        mock_load_csv: Mocked load_csv function.
    """
    mock_load_csv.return_value = pd.DataFrame({"source_id": [1]})
    inputs = iter(["2", "data/my_stars.csv"])

    with patch("builtins.input", lambda *_: next(inputs)):
        result = get_dataframe()

    mock_load_csv.assert_called_once_with("data/my_stars.csv")
    assert isinstance(result, pd.DataFrame)

@patch("gaiadr3_analysis.gaia_input.query_by_datalink")
@patch("gaiadr3_analysis.gaia_input.gaia_login_prompt")
def test_get_dataframe_datalink_path(mock_login, mock_query_dl):
    """Checks that choosing the Datalink menu option calls query_by_datalink with parsed IDs.

    Args:
        mock_login: Mocked gaia_login_prompt function.
        mock_query_dl: Mocked query_by_datalink function.
    """
    mock_query_dl.return_value = {111: pd.DataFrame({"flux": [1.0]})}
    inputs = iter(["3", "111, 222", "n"])

    with patch("builtins.input", lambda *_: next(inputs)):
        result = get_dataframe()

    mock_query_dl.assert_called_once_with(["111", "222"])
    assert result is mock_query_dl.return_value

@patch("gaiadr3_analysis.gaia_input.gaia_login_prompt")
def test_get_dataframe_invalid_choice_returns_none(mock_login):
    """Checks that an invalid menu choice prints an error and returns None.

    Args:
        mock_login: Mocked gaia_login_prompt function.
    """
    with patch("builtins.input", return_value="9"), patch("builtins.print"):
        result = get_dataframe()

    assert result is None