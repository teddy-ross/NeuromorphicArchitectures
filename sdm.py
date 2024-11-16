import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class SDM():

    def __init__(self, p, n): #p = patterns, n = vector size
        
        self.p = p
        self.n = n
        self.radius = 0.451 * n
        self.addresses = np.random.randint(0, 2, (p, n))
        self.data = np.zeros((p,n))

    def enter(self, data):
        
        for i in range(self.p):

            addr = self.addresses[i]

            distance = hammingDistance(data, addr)

            if distance <= self.radius:
                self.data[i] += 2 * data - 1



    def lookup(self, key):

        value = np.zeros(self.n)

        for i in range(self.p):

            addr = self.addresses[i]

            distance = hammingDistance(key, addr)

            if distance <= self.radius:

                value += self.data[i]

        return (value >= 0).astype(int)
        
    

def plot(array, numberOfColumns):

    for i in range(array.size):

        if array[i] == 1 and (i+1) % numberOfColumns == 0:
            print("*")

        elif array[i] == 1:
            print("* ", end="")

        elif (i+1) % numberOfColumns == 0:
            print(" ")

        else:
            print("  ", end="")
        
def ring():

    return np.asarray([0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,
                       0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,
                       0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,                
                       0,1,1,1,1,0,0,0,0,0,0,1,1,1,1,0,
                       0,1,1,1,0,0,0,0,0,0,0,0,1,1,1,0,
                       1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,
                       1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,
                       1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,
                       1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,
                       1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,
                       1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,
                       0,1,1,1,0,0,0,0,0,0,0,0,1,1,1,0,
                       0,1,1,1,1,0,0,0,0,0,0,1,1,1,1,0,
                       0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,
                       0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,
                       0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,])

def noisy_copy(a, p):

    b = a.copy()

    mask = np.random.random(a.shape) < p

    b[mask] = 1 - b[mask]

    return b

def hammingDistance(a, b):

    return np.sum(a != b)


def main():

    print("Part 1: Write your display function \n")
    testarray = np.random.randint(0, 2, 256)
    plot(testarray, 16)

    print("\nPart 2: One Ring to Rule Them All \n")
    r = ring()
    plot(r, 16)

    print("\nPart 3: Code up your SDM and enter and retrieve a pattern")
    sdm = SDM(2000, 256)
    sdm.enter(r)
    plot(sdm.lookup(r), 16)

    print("\nPart 4: Recover pattern after 25% noise added: \n")

    sdm = SDM(2000,256)

    noisyCopy = noisy_copy(r,0.25)

    plot(noisyCopy, 16)

    print("\n")

    sdm.enter(r)

    plot(sdm.lookup(noisyCopy), 16)

    print("\nPart 5: Learn with the following five noisy examples \n")

    sdm = SDM(2000,256)

    noisyArray = []

    for i in range(5):
        noisyArray.append(noisy_copy(r,0.1))
    
    for i in range((5)):
        sdm.enter(noisyArray[i])
        plot(noisyArray[i], 16)
        print('\n')

    print("Test with the following probe: \n")

    probe = noisy_copy(r,0.1)

    plot(probe, 16)

    print("Result: \n")


    plot(sdm.lookup(probe), 16)




    
    




if __name__ == "__main__":
    main()