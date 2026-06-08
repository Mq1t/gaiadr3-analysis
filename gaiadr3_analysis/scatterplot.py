#Create a Ra vs Dec diagram.

import matplotlib.pyplot as plt
import pandas as pd

#Maybe change name to scatter_coordinates? or scatter_location?
def ra_dec(df: pd.DataFrame, xlim: int|float = None, ylim: int|float = None, color: str ='red', size: int|float = 0.5, title: str = 'Right Ascension Vs. Declination'):
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
    #Checks if data is of type pandas dataframe.
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