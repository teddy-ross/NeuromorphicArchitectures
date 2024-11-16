import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class Som():

    def __init__(self, m, n):

        '''
        Constructor for network weights of the SOM model
        m = size of grid, m = 10 = 10x10 grid
        n = dimensionality of weights
        '''

        self.u = np.random.random((m,m,n)) # initial network weights
        self.u = (self.u / 10) + 0.45      # creating clustering around 0.5
    
    def winner(self, e):
        '''
        Returns index j,k of unit that "wins" the data item e
        '''

        m = self.u.shape[0]
        mindist = np.inf
        winner = None

        for j in range(m):

            for k in range(m):

                distance = self.eucdist(self.u[j,k], e)

                if distance < mindist:
                    mindist = distance
                    winner = (j,k)

        return winner


    def eucdist(self, x, y):
        '''
        Returns the Euclidean Distance of Two Points, x and y
        '''
        return np.sqrt(np.sum((x-y)**2))

    def learn(self, data, T, alpha0, d0):
        '''
        The SOM algorithm method for learning how to connect with other points into shapes
        '''
        
        # Iterate t from 0 to T
        for t in tqdm(range(0, T)):

            # Compute current neighborhood radius d and learning rate alpha
            alpha = alpha0*(1-(t/T))
            d = int(np.ceil(d0*(1-(t/T))))

            # Pick an input e from the training set at random
            e = data[np.random.randint(data.shape[0], size=1), :][0]
            
            # Find the winning unit whose weights are closest to this e
            x, y = self.winner(e)

            # Get bounds of winner's neighborhood
            m = self.u.shape[0]
            
            jlo = max(x-d, 0)
            jhi = min(x+d, m)
            klo = max(y-d, 0)
            khi = min(y+d, m)

            
            # Loop over neighborhood, adjust weights
            for j in range(jlo, jhi):
                for k in range(klo, khi):
                    self.u[j,k] += alpha * (e - self.u[j,k]) 


def somplot2d(som):
    '''
    Heavy-lifting function for plotting each neurons as well as interconnected lines ("synapses") between neurons
    '''
    m = som.u.shape[0]

    # Plot each neuron as a red circle in the location determined by 
    # its weight
    for j in range(m):
        for k in range(m): 
            x, y = som.u[j,k] 
            plt.plot(x, y, 'ro')

    # Plot a line between each neuron and the one in the next row
    for j in range(m):
        for k in range(m-1): 
            x1, y1 = som.u[j,k]
            x2, y2 = som.u[j,k+1]
            plt.plot([x1, x2], [y1, y2], 'k')

    # Plot a line between each neuron and the one in the next row
    for j in range(m-1):
        for k in range(m):
            x1, y1 = som.u[j,k]
            x2, y2 = som.u[j+1,k]
            plt.plot([x1, x2], [y1, y2], 'k')


def learnAndPlot(data, title):
    '''
    Creates som object instance and initializes plotting of graph and learning alogrithm
    '''
    som = Som(10, 2)
    plt.scatter(data[:,0], data[:,1], s=.2)

    plt.gca().set_aspect('equal')

    som.learn(data, 4000, 0.02, 4)

    somplot2d(som)

    plt.title(title)

    plt.show() 

def main():
    '''
    Driver function for program -- generates data and then plots respective version of model (square then ring)
    '''
    #generate random data for ring
    data = np.random.random((5000,2)) 

    #plot data
    learnAndPlot(data, "A square connected SOM model")
    
    #generate random data for ring
    data = np.random.random((5000,2))

    #split data into x and y
    x, y = data[:,0], data[:,1]
    
    #radius local variables for ring
    r1, r2 = 0.2, 0.1

    #Gets index logic for ring
    indices = np.logical_and(((x-0.5) ** 2) + (y-0.5) ** 2 < r1, (x-0.5) ** 2 + (y-0.5)**2>r2) 

    #Filters data to be a subset of points creating ring object
    ring = data[indices]

    #plot
    learnAndPlot(ring, "A ring shaped connected SOM model")
    
#main guard and main function
if __name__ == '__main__':
    main()
    