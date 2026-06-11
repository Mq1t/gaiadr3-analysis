from astropy.timeseries import LombScargle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

#t is times in julian days. If time is given in non-julian date already then jd = False when calling the function.
#t is expected to be 'g_transit_time'
#mag is expected to be 'g_transit_mag'
def period(t: pd.DataFrame = None, mag: pd.DataFrame = None, period_range: list[float] = None, lims: list[float] = None, jd: bool=True, plot:bool=False):
    #Default example for if t & y are not inputted.
    if t is None or mag is None:
        rand = np.random.default_rng(42)
        #Time
        t = 100 * rand.random(100)
        #Frequency
        mag = np.sin(2 * np.pi * t) + 0.1 * rand.standard_normal(100)

    #Convert Julian Days
    if jd:
        t = t - t.min()

    #Frequency Range

    # If user didn't specify frequency_range, set it automatically
    if period_range is None:
        # Compute Nyquist period limit
        dt_median = np.median(np.diff(np.sort(t)))
        f_nyq = 1 / (2 * dt_median)
        P_min = 1 / f_nyq
        P_max = t.max() - t.min()
        period_range = [P_min, P_max]

        
    ls = LombScargle(t, mag)
    frequency, power = ls.autopower(
        minimum_frequency=1/period_range[1],    # max period
        maximum_frequency=1/period_range[0]  # min period
    )

    #Convert frequency to period
    period_days = 1/frequency
    
    #Convert to hours for inset
    period_hours = period_days * 24
    
    #Get the best period
    best_idx = np.argmax(power)
    best_period_days = period_days[best_idx]
    best_period_hours = best_period_days * 24
    print(f"Best period: {best_period_days:.6f} days ({best_period_hours:.3f} hours)")

    #Get the top 5 periods
    top5_idx = np.argsort(power)[-5:][::-1]
    
    # Print results
    for rank, idx in enumerate(top5_idx, start=1):
        print(
            f"{rank}. "
            f"Period: {period_days[idx]:.6f} days "
            f"({period_hours[idx]:.3f} hours), "
            f"Power: {power[idx]:.6f}, "
            f"FAP: {ls.false_alarm_probability(power[idx])}"
        )
    if plot:
        #Create main plot
        fig, ax = plt.subplots(figsize=(8,5))
        ax.plot(period_days, power)
        ax.set_xlabel("Period (days)")
        ax.set_xlim(lims[0], lims[1])
        ax.set_ylabel("Lomb-Scargle Power")
        ax.grid(True)


        #Create sub inset graph of best period
        sub_ax = inset_axes(
            parent_axes=ax,
            width="30%",
            height="30%",
            borderpad=2
        )

        # Define zoom window (+-10% around best period)
        zoom_width = 0.10 * best_period_hours
        mask = (period_hours > best_period_hours - zoom_width) & (period_hours < best_period_hours + zoom_width)
        sub_ax.plot(period_hours[mask], power[mask])
        sub_ax.set_xlabel("Period (hours)")
        sub_ax.grid(True)
    
        plt.show()
    
    return best_period_days
