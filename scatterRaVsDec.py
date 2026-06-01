    #RA, X-Value
    x = df['ra']
    #Declination, Y-Values
    y = df['dec']
    
    plt.style.use('dark_background')
    plt.scatter(x, y, s = 0.2)
    plt.title(name+": RA vs Declination")
    plt.xlabel("RA")
    plt.ylabel("Declination")
    plt.show()
#Hello
