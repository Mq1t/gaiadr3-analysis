import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import os

default_folder = None 
style = 'seaborn-v0_8-darkgrid'
plt.style.use(style)

#Create a Ra vs Dec diagram.
def ra_vs_dec(
        df: pd.DataFrame, 
        xlim: int|float = None, 
        ylim: int|float = None, 
        color: str ='red', 
        size: int|float = 0.5, 
        title: str = 'Right Ascension Vs. Declination', 
        save_plot: bool = False, 
        file_name: str | None = 'ra_vs_dec', 
        save_folder: str = default_folder):
    """
    Plot Right Ascension (RA) vs Declination (Dec) from a pandas DataFrame.

    Args:
        df (pd.DataFrame): A pandas DataFrame containing at least two columns, 'ra' for Right Ascension
                          and 'dec' for Declination.
        xlim (int|float, optional): The x-axis upper limit. If None, the default limits are used. Default is None.
        ylim (int|float, optional): The y-axis upper limit. If None, the default limits are used. Default is None.
        color (str, optional): Color of the plotted points. Default is 'red'.
        size (int|float, optional): Size of the plotted points. Default is 0.5.
        title (str, optional): Title of the plot. Default is 'Right Ascension Vs. Declination'.
        save_plot (bool, optional): If true, saves plot as a PDF file. Defaults to False. 
        file_name (str, optional): File name of the resulting plot. Default is 'ra_vs_dec'. File identifier is added automatically.
        save_folder (str, optional): Optional folder destination. A destination folder could also be set using the file name.
    
    Returns:
        None

    Raises:
        TypeError: If the input data is not a pandas DataFrame.
        KeyError: If the required columns are missing ('ra', 'dec')
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Data must be of type pandas.DataFrame')
    # Ensure required columns exist
    required_cols = {'ra', 'dec'}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"DataFrame is missing required columns: {', '.join(sorted(missing))}")

    # RA, X-Value
    x = df['ra']
    # Declination, Y-Values
    y = df['dec']


    plt.scatter(x, y, c = color, s = size)
    plt.title(title)
    plt.xlabel("RA")
    plt.ylabel("Dec")
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    if save_plot:
        safe_name = file_name.replace(" ", "_")
        safe_name = f"{safe_name}.pdf"
        if save_folder is not None:
            os.makedirs(save_folder, exist_ok=True)  
            filepath = os.path.join(save_folder, safe_name)
        else:
            filepath = safe_name
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {filepath}")
    plt.show()

#Proper motion
def pmra_vs_pmdec(
        df: pd.DataFrame, 
        xlim:float=None, 
        ylim:float=None, 
        color: str ='red', 
        size: int|float = 0.5, 
        title: str = 'Right Ascension Vs. Declination in Proper Motion Space', 
        save_plot: bool = False, 
        file_name: str | None = 'pmra_vs_pmdec', 
        save_folder: str = default_folder):
    """
    Plot Right Ascension (RA) vs Declination (Dec) in proper motion space from a pandas DataFrame.

    Args:
        df (pd.DataFrame): A pandas DataFrame containing at least two columns, 'pmra' for Proper Motion in RA
                          and 'pmdec' for Proper Motion in Dec.
        xlim (int|float, optional): The x-axis upper limit. If None, the default limits are used. Default is None.
        ylim (int|float, optional): The y-axis upper limit. If None, the default limits are used. Default is None.
        color (str, optional): Color of the plotted points. Default is 'red'.
        size (int|float, optional): Size of the plotted points. Default is 0.5.
        title (str, optional): Title of the plot. Default is 'Right Ascension Vs. Declination in Proper Motion Space'.
        save_plot (bool, optional): If true, saves plot as a PDF file. Defaults to False. 
        file_name (str, optional): File name of the resulting plot. Default is 'pmra_vs_pmdec'. File identifier is added automatically.
        save_folder (str, optional): Optional folder destination. A destination folder could also be set using the file name.
    
    Returns:
        None

    Raises:
        TypeError: If the input data is not a pandas DataFrame.
        KeyError: If the required columns are missing ('pmra', 'pmdec')
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Data must be of type pandas.DataFrame')
    # Ensure required columns exist
    required_cols = {'pmra', 'pmdec'}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"DataFrame is missing required columns: {', '.join(sorted(missing))}")

    # Proper motion RA, X-Value; Dec, Y-Values
    x = df['pmra']
    y = df['pmdec']

    plt.scatter(x, y, c = color, s = size)

    #Titles and Show graph
    plt.title(title)
    plt.xlabel("PM RA")
    plt.ylabel("PM Dec")
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    if save_plot:
        safe_name = file_name.replace(" ", "_")
        safe_name = f"{safe_name}.pdf"
        if default_folder is not None:
            os.makedirs(save_folder, exist_ok=True)
            filepath = os.path.join(save_folder, safe_name)
        else:
            filepath = safe_name
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {filepath}")
    plt.show()

def get_distance(parallax):
    """Convert parallax (mas) to distance (pc)

    Args:
        parallax (float): Parallax in milliarcseconds

    Returns:
        float: Distance in parsecs
    """
    return 1 / (parallax / 1000)


def get_magnitude(phot_g_mean_mag, distance):
    """Convert apparent magnitude to absolute magnitude

    Args:
        phot_g_mean_mag (float): G-band apparent magnitude
        distance (float): Distance in parsecs

    Returns:
        float: Absolute magnitude
    """
    return phot_g_mean_mag - 5 * np.log10(distance / 10)


def get_bprp(phot_bp_mean_mag, phot_rp_mean_mag):
    """Calculate BP-RP colour index

    Args:
        phot_bp_mean_mag (float): BP-band apparent magnitude
        phot_rp_mean_mag (float): RP-band apparent magnitude

    Returns:
        float: BP-RP colour index
    """
    return phot_bp_mean_mag - phot_rp_mean_mag


def plot_hr_diagram(
        df, 
        title: str = "Hertzsprung-Russell Diagram", 
        save_plot: bool = False, 
        file_name: str = "hr_diagram", 
        save_folder: str = default_folder
    ):
    """Plot an HR diagram from a Gaia dataframe.

    Args:
        df (pandas.dataframe): Gaia data containing parallax,
            phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag.
        title (str, optional): Title of the plot. Default is 'Hertzsprung-Russell Diagram'.
        save_plot (bool, optional): If true, saves plot as a PDF file. Defaults to False. 
        file_name (str, optional): File name of the resulting plot. Default is 'hr_diagram'. File identifier is added automatically.
        save_folder (str, optional): Optional folder destination. A destination folder could also be set using the file name.

    Returns: 
        None
    """

    df = df.dropna(subset=["parallax", "phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag"])

    magnitude = [get_magnitude(row["phot_g_mean_mag"], get_distance(row["parallax"])) for _, row in df.iterrows()]
    bprp = [get_bprp(row["phot_bp_mean_mag"], row["phot_rp_mean_mag"]) for _, row in df.iterrows()]

    plt.figure()
    plt.scatter(bprp, magnitude, c="purple", s=1)
    plt.xlabel("BP - RP")
    plt.ylabel("Absolute Magnitude")
    plt.title(title)
    plt.gca().invert_yaxis()

    if save_plot:
        safe_name = file_name.replace(" ", "_")
        safe_name = f"{safe_name}.pdf"
        if default_folder is not None:
            os.makedirs(save_folder, exist_ok=True)
            filepath = os.path.join(save_folder, safe_name)
        else:
            filepath = safe_name
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {filepath}")
    plt.show()

def hist(
        dists, 
        bin_num:int = 50, 
        parallax:bool =False, 
        title:str = "Distances histogram", 
        save_plot: bool = False, 
        file_name: str = "distance_hist", 
        save_folder: str = default_folder):

    """Plot an HR diagram from a Gaia dataframe.

    Args:
        dists (array-like): Distances to star(s).
        bin_num (int, optional): Number of bins, defaults to 50.
        parallax (bool, optinal): Set this to be true if using parallax data. If true converts the data into parsecs. 
        title (str, optional): Title of the plot. Default is 'Distances histogramm'.
        save_plot (bool, optional): If true, saves plot as a PDF file. Defaults to False. 
        file_name (str, optional): File name of the resulting plot. Default is 'distance_hist'. File identifier is added automatically.
        save_folder (str, optional): Optional folder destination. A destination folder could also be set using the file name.

    Returns: 
        None
    """

    #Adjust if dist given in parallax
    if parallax:
        dists = (1000/dists)
    
    plt.title(title)
    plt.hist(dists, bins=bin_num)
    plt.xlabel('Distance (pc)')
    plt.ylabel('Stars per bin')
    if save_plot:
        safe_name = file_name.replace(" ", "_")
        safe_name = f"{safe_name}.pdf"
        if default_folder is not None:
            os.makedirs(save_folder, exist_ok=True)
            filepath = os.path.join(save_folder, safe_name)
        else:
            filepath = safe_name
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {filepath}")
    plt.show()

def gaussian(x, A, sigma, mu):
    return A*(1/(sigma * np.sqrt(2*np.pi)) * np.exp(-1*(x - mu)**2 / (2*sigma**2)))

def fittedHist(
        dists, 
        bin_num:int =50, 
        range:list[int] =[-500,500],
        parallax:bool =False, 
        title:str = "Distances histogram", 
        save_plot: bool = False, 
        file_name: str = "fitted_dist_hist", 
        save_folder: str = default_folder):
    #Magnitude, Y-Values
    if parallax:
        dists = (1000/dists)

    median = dists.median()
    std = dists.std()

    print("Distance"+", meidian: "+ str(median))

    plt.title(title)
    h_1d_output = plt.hist(dists, bins=bin_num)
    x_plot = np.linspace(range[0],range[1], 300)
    x_1d_fit = (h_1d_output[1][:-1]+h_1d_output[1][1:])/2
    y_1d_fit = h_1d_output[0]
    fit = curve_fit(gaussian, x_1d_fit, y_1d_fit, p0 = [55, std, median])
    print("Standard Deviation: "+str(std))

    #Fix printing this
    #print(fit)
    plt.plot(x_plot, gaussian(x_plot, *fit[0]), label ='Line of Best Fit')

    plt.xlim(range[0], range[1])
    plt.xlabel('Distance (pc)')
    plt.ylabel('Stars per bin')
    plt.legend()

    if save_plot:
        safe_name = file_name.replace(" ", "_")
        safe_name = f"{safe_name}.pdf"
        if default_folder is not None:
            os.makedirs(save_folder, exist_ok=True)
            filepath = os.path.join(save_folder, safe_name)
        else:
            filepath = safe_name
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {filepath}")
    plt.show()