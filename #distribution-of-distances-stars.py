import matplotlib.pyplot as plt


def hist(dists, bin_num, parallax=False):
    #Magnitude, Y-Values

    #Adjust if dist given in parallax
    if parallax:
        dists = (1000/dists)

    plt.title(name)
    plt.hist(dists, bins=bin_num)

    plt.xlabel(name)
    plt.ylabel('Stars per bin')
    plt.show()

def gaussian(x, A, sigma, mu):
    return A*(1/(sigma * np.sqrt(2*np.pi)) * np.exp(-1*(x - mu)**2 / (2*sigma**2)))

def fittedHist(dists, bin_num=50, range=[-500,500],parallax=False):
    #Magnitude, Y-Values
    if parallax:
        dists = (1000/dists)

    median = x.median()
    std = x.std()

    print(name+": "+ str(median))
    plt.title("Starcount Histogram: "+name)

    h_1d_output = plt.hist(x, bins=bin_num)
    x_plot = np.linspace(range[0],range[1], 300)
    x_1d_fit = (h_1d_output[1][:-1]+h_1d_output[1][1:])/2
    y_1d_fit = h_1d_output[0]
    fit = curve_fit(gaussian, x_1d_fit, y_1d_fit, p0 = [55, std, median])
    print("Standard Deviation: "+str(x.std()))

    #Fix printing this
    #print(fit)
    plt.plot(x_plot, gaussian(x_plot, *fit[0]), label ='Line of Best Fit')

    plt.xlim(range[0], range[1])
    plt.xlabel(name)
    plt.ylabel('Stars per bin')
    plt.legend()
    plt.show()