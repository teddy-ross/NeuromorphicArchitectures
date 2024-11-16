import numpy as np


def mag(a):

    return np.linalg.norm(a)


def cosine(a, b):

    return np.dot(a, b) / (mag(a) * mag(b))


def show_confusion(data1, data2):

    n = len(data1)

    for i in range(n):
        for j in range(i+1):
            print('%3.2f ' % cosine(data1[i], data2[j]), end='')
        print()


def noisy_copy(a, p):

    b = a.copy()

    mask = np.random.random(a.shape) < p

    b[mask] = 1 - b[mask]

    return b



class Hopfield:

    def __init__(self, n):

        self.T = np.zeros((n,n))

    def learn(self, data):

        for a in data:

            self.T += np.outer(2 * a - 1,  2 * a - 1)

        np.fill_diagonal(self.T, 0)

    def test(self, u, niter=3):

        for _ in range(niter): 

            u = np.dot(u, self.T) >= 0

        return u.astype(int)


def main():

    N = 30

    print('Part 1: Generate some random bit vectors ---------------------------------\n')

    data1 = np.random.randint(0, 2, (5, N))

    print(data1)

    print('\nPart 2: Show the confusion matrix of some vectors with themselves ------\n')

    show_confusion(data1, data1)

    print('\nPart 3: Confusion matrix with 25% noise --------------------------------\n')

    noisy = noisy_copy(data1, 0.25)

    show_confusion(data1, noisy)

    print('\nPart 4: Code up your Hopfield net --------------------------------------\n')

    net = Hopfield(N)

    net.learn(data1)

    print('\nRecover pattern, no noise:')

    a = data1[0]
    print('Input:  ', a)
    atest = net.test(a)
    print('Output: ', atest)
    print('Vector cosine = %3.3f' % cosine(a, atest))

    print('\nRecover pattern, 25% noise:')

    print('Input:  ', a)
    anoisy = noisy_copy(a, 0.25)
    noisy_test = net.test(anoisy)
    print('Output: ', noisy_test)
    print('Vector cosine = %3.3f' % cosine(a, noisy_test))

    print('\nPart 5: Recovering big patterns ----------------------------------------\n')

    N = 1000

    data1 = np.random.randint(0, 2, (10, N))

    print("Confusion matrix for 1000-element vectors with 25 percent noise: \n \n")

    noisy = noisy_copy(data1, 0.25)

    show_confusion(data1, noisy)

    print("\nRecovering patterns with 25 percent noise: \n")

    newNet = Hopfield(N)

    newNet.learn(data1)

    for i in range(10):

        a = data1[i]

        anoisy = noisy_copy(a, 0.25)

        atest = newNet.test(anoisy)

        print('Vector cosine on pattern',  i, ' = ', '%3.3f' % cosine(a, atest))

        
if __name__ == '__main__':
    main()