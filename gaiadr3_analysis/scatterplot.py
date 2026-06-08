#Create a Ra vs Dec diagram.

import matplotlib.pyplot as plt
import pandas as pd

#Maybe change name to scatter_coordinates? or scatter_location?
def ra_dec(df: pd.DataFrame, xlim: int|float = None, ylim: int|float = None, color: str ='red', size: int|float = 0.5, title: str = 'Right Ascension Vs. Declination'):
    """
    Plot Right Ascension (RA) vs Declination (Dec) from a pandas DataFrame.

    Parameters:
        df (pd.DataFrame): A pandas DataFrame containing at least two columns, 'ra' for Right Ascension
                          and 'dec' for Declination.
        xlim (int|float, optional): The x-axis limits as a tuple (xmin, xmax). If None, the default limits are used. Default is None.
        ylim (int|float, optional): The y-axis limits as a tuple (ymin, ymax). If None, the default limits are used. Default is None.
        color (str, optional): Color of the plotted points. Default is 'red'.
        size (int|float, optional): Size of the plotted points. Default is 0.5.
        title (str, optional): Title of the plot. Default is 'Right Ascension Vs. Declination'.

    Raises:
        TypeError: If the input data is not a pandas DataFrame.

    Returns:
        None
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Data must be of type pandas.DataFrame')
    
    #RA, X-Value
    x = df['ra']
    #Declination, Y-Values
    y = df['dec']
    
    plt.scatter(x, y, c = color, s = size)
    plt.title(title)
    plt.xlabel("RA")
    plt.ylabel("Dec")
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.show()

#Proper motion
def proper_motion(df: pd.DataFrame, xlim:float=None, ylim=None, color: str ='red', size: int|float = 0.5, title: str = 'Right Ascension Vs. Declination in Proper Motion Space'):
    """
    Plot Right Ascension (RA) vs Declination (Dec) in proper motion space from a pandas DataFrame.

    Parameters:
        df (pd.DataFrame): A pandas DataFrame containing at least two columns, 'pmra' for Proper Motion in RA
                          and 'pmdec' for Proper Motion in Dec.
        xlim (float, optional): The x-axis limits as a tuple (xmin, xmax). If None, the default limits are used. Default is None.
        ylim (float, optional): The y-axis limits as a tuple (ymin, ymax). If None, the default limits are used. Default is None.
        color (str, optional): Color of the plotted points. Default is 'red'.
        size (int|float, optional): Size of the plotted points. Default is 0.5.
        title (str, optional): Title of the plot. Default is 'Right Ascension Vs. Declination in Proper Motion Space'.

    Raises:
        TypeError: If the input data is not a pandas DataFrame.

    Returns:
        None
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Data must be of type pandas.DataFrame')
    
    #RA, X-Value
    #Declination, Y-Values
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
    plt.show()