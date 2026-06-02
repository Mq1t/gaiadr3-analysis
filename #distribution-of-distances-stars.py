
import matplotlib.pyplot as plt

def plotDistances(df, title:str='Distribution of Distances of Stars'):

    plt.title(title)

#Calculating the distances of the filtered stars
    distances = (1000/df['parallax'])
    h_1d_output = plt.hist(distances, bins=100)

    plt.xlabel('Distances of Stars (pc)')
    plt.ylabel('Stars per bin')
    plt.show()


