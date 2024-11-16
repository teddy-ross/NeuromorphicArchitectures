import random
import time
import numpy as np
import matplotlib.pyplot as plt

def createList(size):
    '''Creates A List of SIZE Random Real Numbers between 0 and 1.0'''

    lyst = []
    
    for i in range(size):
        lyst.append(random.random())

    return lyst
        
def computeByHandDotProduct(list1, list2):
    '''Computes the Dot Product of Two Lists Using Mathematical Formula '''
    dotProduct = 0
    for i in range(len(list1)):
        dotProduct += (list1[i] * list2[i])
    return dotProduct

def computeNumpyDotProduct(list1, list2):
    '''Computes the Dot Product of Two Lists Using Numpy'''
    return np.dot(list1, list2)

def computeBothTimes(size):

    list1 = createList(size)
    list2 = createList(size)

    #Time Mathematical Dot Product Calc
    beginningTimeByHand = time.time()
    byHandDotProduct = computeByHandDotProduct(list1, list2)
    endTimeByHand = time.time()

    list3 = np.array(list1)
    list4 = np.array(list2)
    #Time Numpy Dot Product Calc
    beginningTimeNumpy = time.time()
    numpyDotProduct = computeNumpyDotProduct(list3, list4)
    endTimeNumpy = time.time()

    numpyDotProductTime = endTimeNumpy - beginningTimeNumpy 
    byHandDotProductTime = endTimeByHand - beginningTimeByHand

    return byHandDotProduct, byHandDotProductTime, numpyDotProduct, numpyDotProductTime


def main():

    byHandDotProduct, byHandDotProductTime, numpyDotProduct, numpyDotProductTime = computeBothTimes(1_000_000)

    #Print Both Dot Product Calculations to Check And See if It's working
    print("\nNumpy's dot product     : ", numpyDotProduct)
    print("Mathematical dot product: ", byHandDotProduct)
    
    #Explanation of expected random value
    print("\nThe expected value of this operation is 250,000. Given that the average ")
    print("random value between 0 and 1 is 0.5, when we consider the dot product ")
    print("operation we are essentially taking one 'average' entry of .5 and ")
    print("multiplying it by one 'average' entry of .5. This results in .25. ")
    print("Summed up 1,000,000 times results in the 250,000 value. This ")
    print("hypothesis is confirmed via testing.\n\n")
    
    print("The numPy dot product function took  ", numpyDotProductTime, "seconds") #output dot product times
    print("The by-hand dot product function took", byHandDotProductTime, "seconds")

    

    #Final Section/Part 3

    byhand_times = []
    numpy_times = []
    numberOfValues = []

    for size in range(1_000_000, 10_000_001, 1_000_000):

        byhand_times.append(computeBothTimes(size)[1])
        numpy_times.append(computeBothTimes(size)[3])
        numberOfValues.append(size)
        

    byhand_times = np.array(byhand_times)
    numpy_times = np.array(numpy_times)
    
    plt.plot(numberOfValues, byhand_times,label = "By Hand")
    plt.plot(numberOfValues, numpy_times, label = "NumPy")
    
    plt.title("By Hand versus NumPy Dot Product Computation Time")
    plt.xlabel("Number of Values")
    plt.ylabel("Time (seconds)")
    
    plt.legend()
    plt.show()
        

    

    


#Main Statement
main()