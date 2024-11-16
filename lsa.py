import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import re
from tqdm import tqdm

def sim(A, B):

    return np.dot(A,B)/(np.linalg.norm(A)*np.linalg.norm(B))


def main():

   

    print("Step 1: Set term weights and construct the term-document matrix A and query matrix \n")

    termList = ["a", "arrived", "damaged", "delivery", "fire", "gold", "in", "of", "shipment", "silver", "truck"]
    
    d1 = np.asarray([1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0])
    d2 = np.asarray([1, 1, 0, 1, 0, 0, 1, 1, 0, 2, 1])
    d3 = np.asarray([1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1])

    q = np.asarray ([0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1]) # querying the latent semantic index for gold silver truck

    a = np.asarray([d1, d2, d3])

    a = np.transpose(a)

    for i, item in enumerate(termList):
        # print(item, "      ", a[i], "         ", q[i])
        print("%10s" % item, end="  ")
        print(a[i], q[i])

    print("\nStep 2: Decompose matrix A matrix and find the U, S and V matrices,where A = USV^T\n")

    U, S, V = np.linalg.svd(a, full_matrices = False)

    print("U = \n", U)

    print("\nS = \n", S)

    print("\n")

    print("V = \n", V)
    
    print("\nStep 3: Implement a Rank 2 Approximation by keeping the first two columns of ")
    print("U and V and the first two columns and rows of S.\n")
    
    U = np.delete(U, 2, 1)
    
    S = np.delete(S, 2, 0)

    V = np.delete(V, 2, 0)

    print("U = \n", U)

    print("\nS = \n", S)

    print("\nV = \n", V)

    print("\nStep 4: Find the new document vector coordinates in this reduced 2-dimensional space.")

    d1 = V[:, 0:1]

    d1 = np.transpose(d1)

    d1 = d1[0]

    d2 = V[:, 1:2]

    d2 = np.transpose(d2)

    d2 = d2[0]

    d3 = V[:, 2:3]

    d3 = np.transpose(d3)

    d3 = d3[0]

    print("\nd1: ", d1)

    print("d2: ", d2)

    print("d3: ", d3)

    print("\nStep 5: Find the new query vector coordinates in the reduced 2-dimensional space.\n")

    S = np.asarray([[S[0], 0,],
                   [0,    S[1]]])

    q = np.matmul(np.matmul(np.transpose(q), U), np.linalg.inv(S))

    print(q)

    print("\nStep 6: Rank documents in decreasing order of query-document cosine similarities.\n")

    print("Similarity between query and d1: %+3.3f: " % sim(q, d1))

    print("Similarity between query and d2: %+3.3f: " % sim(q, d2))

    print("Similarity between query and d3: %+3.3f: " % sim(q, d3), "\n")

    lyst = [sim(q, d1), sim(q, d2), sim(q, d3)]

    largest = lyst[0]
    largestIndex = 0

    secondLargest = lyst[0]
    secondLargestIndex = 0

    thirdLargest = lyst[0]
    thirdLargestIndex = 0

    for i, data in enumerate(lyst):
        if data > largest:
            largest = data
            largestIndex = i

    if largestIndex == 0:
        print("Highest cosine is sim(q,d1):                   %+3.3f" % sim(q,d1))
        if lyst[1] > lyst[2]:
            secondLargest = lyst[1]
            secondLargestIndex = 1
        else:
            secondLargest = lyst[2]
            secondLargestIndex = 2
            thirdLargest = 1
            thirdLargestIndex = 1
    elif largestIndex == 1:
        print("Highest cosine is sim(q,d1):                   %+3.3f" % sim(q,d2))
        if lyst[0] > lyst[2]:
            secondLargest = lyst[0]
            secondLargestIndex = 0
        else:
            secondLargest = lyst[2]
            secondLargestIndex = 2
            thirdLargest = lyst[0]
            thirdLargestIndex = 0
    else:
        print("Highest cosine is sim(q,d3):                   %+3.3f" % sim(q,d3))
        if lyst[0] > lyst[1]:
            secondLargest = lyst[0]
            secondLargestIndex = 0
        else:
            secondLargest = lyst[1]
            secondLargestIndex = 1
            thirdLargest = lyst[0]
            thirdLargestIndex = 0

    if secondLargestIndex == 1:

        print("Second Highest cosine Similarity is sim(q,d2): %+3.3f" % secondLargest)
        print("Lowest cosine Similarity is sim(q,d3):         %+3.3f" % secondLargest)

    else:
        print("Second Highest cosine Similarity is sim(q,d3): %+3.3f" % secondLargest)
        print("Lowest cosine similarity is sim(q, d2):        %+3.3f"% thirdLargest)

    


    

if __name__ == "__main__":
    main()
