import numpy as np

class VSA:
    def __init__(self, n):
        """
        Constructor to initialize the vector size and store named vectors, 
        as well as permutation indices.
        """
        self.n = n
        self.vectors = {}
        
        self.permutationIndices = np.random.permutation(n)
        
        self.inversePermutationIndices = np.argsort(self.permutationIndices)

    def randvec(self, name):
        """
        Create a random vector of size n with values -1 or +1, and store it by name.
        """
        
        randomVector = np.random.randint(0, 2, size=self.n) * 2 - 1
        self.vectors[name] = randomVector
        return randomVector

    def cosineSim(self, v1, v2):
        """
        Compute the cosine similarity between two vectors.
        """
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def winner(self, queryVector):
        """
        Find the "winner" vector (most similar vector) based on cosine similarity.
        """
        similarities = {
            name: self.cosineSim(queryVector, v)
            for name, v in self.vectors.items()
        }
        winner = max(similarities, key=similarities.get)
        return winner

    def permute(self, vector):
        """
        Permute the given vector using the stored permutation indices.
        """
        return vector[self.permutationIndices]

    def perminv(self, vector):
        """
        Apply the inverse of the stored permutation to the vector.
        """
        return vector[self.inversePermutationIndices]

    def seqencode(self, seq):
        '''
        Returns vector encoding a given sequence of symbols.
        '''
        vec = np.zeros(self.n)
        
        for sym in seq[::-1]:

            vec = self.permute(vec + self.vectors[sym])

        return vec

    def seqdecode(self, vec, threshold = 0.2):
        '''
        Returns the sequences decode from a given vector.
        '''
        
        seq = []

        while True:
            
            vec = self.perminv(vec)

            winner = self.winner(vec)

            sim = self.cosineSim(vec,self.vectors[winner])

            if sim < threshold:
                break

            seq.append(self.winner(vec))

        return seq