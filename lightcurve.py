import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

def phase(t, T_0, P):
    return ((t-T_0)/P)%1

#Plot G, Bp and Rp magnitude light curves in time.
def plotLightCurves(df, g, bp, rp, g_time, bp_time, rp_time, title:str='Flux Vs. Time', overplot:bool=False, rejectflags: bool=False, period:float=None, xlim:float=None, ylim:float=None):
    #Filter Rejections if true
    if rejectflags:
        g_df = df[df['variability_flag_g_reject'] == False]
        bp_df = df[df['variability_flag_bp_reject'] == False]
        rp_df = df[df['variability_flag_rp_reject'] == False]
    else:
        g_df = df
        bp_df = df
        rp_df = df

    print(f"Len g, bp, and rp datasets respectively: {len(g_df)}, {len(bp_df)}, {len(rp_df)}")
    #X-Value: G Transit time
    x_g = g_df[g_time] + 2455197.5
    x_bp = bp_df[bp_time] + 2455197.5
    x_rp = rp_df[rp_time] + 2455197.5
    x_label = "Time (JD)"
    
    #Phase x if true
    if period is not None:
        x_g = phase(x_g, x_g.median(), period)
        x_bp = phase(x_bp, x_g.median(), period)
        x_rp = phase(x_rp, x_g.median(), period)
        x_label = f"Phase, p = {p}"
        print(f"P value: {p}")
    
    #Y-Value: Magnitudes for light band
    y_g = g_df[g] 
    y_bp = bp_df[bp]
    y_rp = rp_df[rp]

    if overplot is not None:
        plt.xlabel(x_label)
        plt.ylabel("Band (app mag)")
        plt.scatter(x_g, y_g, c ='green', s = 3, label='G Band')
        plt.scatter(x_rp, y_rp, c ='red', s = 3, label='Rp Band')
        plt.scatter(x_bp, y_bp, c ='blue', s = 3, label='Bp Band')
        plt.title(title)
        plt.legend()
        plt.gca().invert_yaxis()
    else:
        fig, axes = plt.subplots(4, 1, figsize=(6, 10))
        # Plot on each subplot
        #axes[0].set_title(plot_title)
        axes[0].set_xlabel(x_label)
        axes[0].set_ylabel("G Band (app mag)")
        axes[0].scatter(x_g, y_g, c ='green', s = 4, label='G Band')
        axes[0].legend()
        axes[0].invert_yaxis()
    
        #axes[1].set_title(plot_title)
        axes[1].set_xlabel(x_label)
        axes[1].set_ylabel("Bp Band (app mag)")
        axes[1].scatter(x_bp, y_bp, c ='blue', s = 4, label='Bp Band')
        axes[1].legend()
        axes[1].invert_yaxis()
    
        #axes[2].set_title(plot_title)
        axes[2].set_xlabel(x_label)
        axes[2].set_ylabel("Rp Band (app mag)")
        axes[2].scatter(x_rp, y_rp, c ='red', s = 4, label='Rp Band')
        axes[2].legend()
        axes[2].invert_yaxis()

        #Overplot 
        #axes[2].set_title(plot_title)
        axes[3].set_xlabel(x_label)
        axes[3].set_ylabel("Band (app mag)")
        axes[3].scatter(x_g, y_g, c ='green', s = 3, label='G Band')
        axes[3].scatter(x_rp, y_rp, c ='red', s = 3, label='Rp Band')
        axes[3].scatter(x_bp, y_bp, c ='blue', s = 3, label='Bp Band')
        axes[3].legend()
        axes[3].invert_yaxis()

        if ylims == None:
            for ax in axes:
                ax.set_ylim(5.41, 5.25)
        else:
            for ax in axes: 
                ax.set_ylim(ylims[1], ylims[0])

    plt.tight_layout()
    plt.show()