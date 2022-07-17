import numpy as np
from scipy.sparse import csc_matrix
import sys
import collections
import itertools
import math
import time
import argparse

class Jaccard:
    def __init__(self, nr_bands, nr_rows, file_path):
        self.bands = nr_bands
        self.rows = nr_rows
        self.sig_length = self.bands * self.rows

        matrix = np.load(file_path)
        # creating scipy sparse column matrix
        self.matrix = csc_matrix((matrix[:, 2], (matrix[:, 1], matrix[:, 0])))

        self.movies, self.users = np.shape(self.matrix)

        # storing all the movies watched by a particular user
        self.users_movies = {i: self.matrix[:, i].nonzero()[0] for i in range(self.users)}



    def minHash(self):
        sig_matrix = np.zeros((self.sig_length, self.users))

        # minhashing implemented using random permutations of the rows for the sparse matrix
        for i in range(0, self.sig_length):
            self.matrix = self.matrix[np.random.permutation(self.movies)]

            for user in range(1, self.users):
                sig_matrix[i, user] = self.matrix.indices[self.matrix.indptr[user]:self.matrix.indptr[user+1]].min()

        return sig_matrix



    def lsh(self, sig_matrix):
        lsh_matrix = np.zeros((self.bands, self.users))

        # lsh implemented by hashing each band for each user
        for band, j in enumerate(range(0, self.sig_length, self.rows)):
            for user in range(0, self.users):
                lsh_matrix[band, user] = hash(sig_matrix[j:j + self.rows, user].tobytes())

        return lsh_matrix



    def pairs(self, sig_matrix, lsh_matrix):
        candidates = set()

        for band in range(0, self.bands):
            buckets, counts = np.unique(lsh_matrix[band], return_counts=True)

            # index values of the buckets that contain between 1 and 100 hashed values
            for i in (np.where((counts > 1) & (counts <= 100))[0]):
                # getting the index value of users which share a bucket
                idx = np.where(lsh_matrix[band] == buckets[i])[0]
                # checking the possible combination of pairs found in a bucket
                for user1, user2 in itertools.combinations(idx,2):
                    pair = tuple(sorted((user1, user2)))

                    if not pair in candidates:
                        candidates.add(pair)
                        # calculating the jaccard similairty
                        similarity_score = np.sum(sig_matrix[:, user1] == sig_matrix[:, user2]) / self.sig_length

                        if similarity_score >= 0.4:
                            u1_movies = self.users_movies[user1]
                            u2_movies = self.users_movies[user2]
                            # calculating the jaccard similairty
                            similarity_score_true = np.intersect1d(u1_movies, u2_movies).size / np.union1d(u1_movies, u2_movies).size

                            if similarity_score_true >= 0.5:
                                file = open("js.txt", 'a')
                                file.write("{}, {}\n".format(pair[0], pair[1]))
                                file.close()
        # print(len(candidates))



class Cosine:
    def __init__(self, nr_bands, nr_rows, file_path):
        self.bands = nr_bands
        self.rows = nr_rows
        self.sig_length = self.bands * self.rows


        matrix = np.load(file_path)
        self.matrix = csc_matrix((matrix[:, 2], (matrix[:, 1], matrix[:, 0])))

        self.movies, self.users = np.shape(self.matrix)



    def randomProjection(self):
        sig_matrix = np.zeros((self.sig_length, self.users))

        for i in range(0, self.sig_length):
            random_vector = np.random.choice([1, -1], self.movies)
            for user in range(1, self.users):
                indices = self.matrix.indices[self.matrix.indptr[user]:self.matrix.indptr[user+1]]
                data = self.matrix.data[self.matrix.indptr[user]:self.matrix.indptr[user+1]]
                val = np.dot(random_vector[indices], data)
                if  val > 0:
                    sig_matrix[i, user] = 1
                elif val < 0:
                    sig_matrix[i, user] = -1
                else:
                    sig_matrix[i, user] = np.random.choice([1, -1])

        return sig_matrix



    def lsh(self, sig_matrix):
        lsh_matrix = np.zeros((self.bands, self.users))

        for band, j in enumerate(range(0, self.sig_length, self.rows)):
            for user in range(0, self.users):
                lsh_matrix[band, user] = hash(sig_matrix[j:j + self.rows, user].tobytes())

        return lsh_matrix



    def pairs(self, sig_matrix, lsh_matrix):
        candidates = set()

        for band in range(0, self.bands):
            buckets, counts = np.unique(lsh_matrix[band], return_counts=True)

            for i in (np.where((counts > 1) & (counts <= 100))[0]):
                idx = np.where(lsh_matrix[band] == buckets[i])[0]

                for user1, user2 in itertools.combinations(idx,2):
                    pair = tuple(sorted((user1, user2)))

                    if not pair in candidates:
                        candidates.add(pair)
                        cos_theta = np.dot(sig_matrix[:, user1],  sig_matrix[:, user2]) / self.sig_length
                        similarity_score = (1-np.degrees(np.arccos(cos_theta))/180)

                        if similarity_score >= 0.6:
                            user1_data = self.matrix.getcol(user1).data
                            user2_data = self.matrix.getcol(user2).data
                            user1_indices = self.matrix.indices[self.matrix.indptr[user1]:self.matrix.indptr[user1+1]]
                            user2_indices = self.matrix.indices[self.matrix.indptr[user2]:self.matrix.indptr[user2+1]]
                            xy = np.intersect1d(user1_indices, user2_indices, return_indices=True)
                            cos_theta_true = np.dot(user1_data[xy[1]], user2_data[xy[2]]) /  math.sqrt((user1_data**2).sum() * (user2_data**2).sum())
                            similarity_score_true = 1 - np.degrees(np.arccos(cos_theta_true.data)) / 180

                            if similarity_score_true >= 0.73:
                                file = open("cs.txt", 'a')
                                file.write("{}, {}\n".format(pair[0], pair[1]))
                                file.close()
        # print(len(candidates))



class DiscreteCosine:
    def __init__(self, nr_bands, nr_rows, file_path):
        self.bands = nr_bands
        self.rows = nr_rows
        self.sig_length = self.bands * self.rows

        matrix = np.load(file_path)
        matrix[:, 2] = 1
        self.matrix = csc_matrix((matrix[:, 2], (matrix[:, 1], matrix[:, 0])))

        self.movies, self.users = np.shape(self.matrix)



    def randomProjection(self):
        sig_matrix = np.zeros((self.sig_length, self.users))

        for i in range(0, self.sig_length):
            random_vector = np.random.choice([1, -1], self.movies)
            for user in range(1, self.users):
                indices = self.matrix.indices[self.matrix.indptr[user]:self.matrix.indptr[user+1]]
                data = self.matrix.data[self.matrix.indptr[user]:self.matrix.indptr[user+1]]
                val = np.dot(random_vector[indices], data)
                if  val > 0:
                    sig_matrix[i, user] = 1
                elif val < 0:
                    sig_matrix[i, user] = -1
                else:
                    sig_matrix[i, user] = np.random.choice([1, -1])

        return sig_matrix



    def lsh(self, sig_matrix):
        lsh_matrix = np.zeros((self.bands, self.users))

        for band, j in enumerate(range(0, self.sig_length, self.rows)):
            for user in range(0, self.users):
                lsh_matrix[band, user] = hash(sig_matrix[j:j + self.rows, user].tobytes())

        return lsh_matrix



    def pairs(self, sig_matrix, lsh_matrix):
        candidates = set()

        for band in range(0, self.bands):
            buckets, counts = np.unique(lsh_matrix[band], return_counts=True)

            for i in (np.where((counts > 1) & (counts <= 100))[0]):
                idx = np.where(lsh_matrix[band] == buckets[i])[0]

                for user1, user2 in itertools.combinations(idx,2):
                    pair = tuple(sorted((user1, user2)))

                    if not pair in candidates:
                        candidates.add(pair)
                        cos_theta = np.dot(sig_matrix[:, user1],  sig_matrix[:, user2]) / self.sig_length
                        similarity_score = (1-np.degrees(np.arccos(cos_theta))/180)

                        if similarity_score >= 0.6:
                            user1_data = self.matrix.getcol(user1).data
                            user2_data = self.matrix.getcol(user2).data
                            user1_indices = self.matrix.indices[self.matrix.indptr[user1]:self.matrix.indptr[user1+1]]
                            user2_indices = self.matrix.indices[self.matrix.indptr[user2]:self.matrix.indptr[user2+1]]
                            xy = np.intersect1d(user1_indices, user2_indices, return_indices=True)
                            cos_theta_true = np.dot(user1_data[xy[1]], user2_data[xy[2]]) /  math.sqrt((user1_data**2).sum() * (user2_data**2).sum())
                            similarity_score_true = 1 - np.degrees(np.arccos(cos_theta_true.data)) / 180

                            if similarity_score_true >= 0.72:
                                file = open("dcs.txt", 'a')
                                file.write("{}, {}\n".format(pair[0], pair[1]))
                                file.close()
        # print(len(candidates))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_path", type=str,
                        help="file path to data (string)")

    parser.add_argument("-s", "--random_seed", type=int,
                        help="random seed (integer)")

    parser.add_argument("-m", "--similarity_measure", type=str,
                        help="type of similarity measure to use (string)")

    args = parser.parse_args()

    if args.data_path:
        path = args.data_path

    if args.random_seed:
        seed = args.random_seed

    if args.similarity_measure:
        sim_measure = args.similarity_measure

    np.random.seed(int(seed))

    def js():
        open("js.txt", "w").close()
        # start = time.process_time()
        js = Jaccard(30, 5, path)
        sig_matrix = js.minHash()
        lsh_matrix = js.lsh(sig_matrix)
        similar_pairs = js.pairs(sig_matrix, lsh_matrix)
        # print((time.process_time() - start)/60)


    def cs():
        open("cs.txt", "w").close()
        # start = time.process_time()
        cs = Cosine(34, 12, path)
        sig_matrix = cs.randomProjection()
        lsh_matrix = cs.lsh(sig_matrix)
        similar_pairs = cs.pairs(sig_matrix, lsh_matrix)
        # print((time.process_time() - start)/60)


    def dcs():
        open("dcs.txt", "w").close()
        # start = time.process_time()
        dcs = DiscreteCosine(34, 12, path)
        sig_matrix = dcs.randomProjection()
        lsh_matrix = dcs.lsh(sig_matrix)
        similar_pairs = dcs.pairs(sig_matrix, lsh_matrix)
        # print((time.process_time() - start)/60)


    def default():
        print("invalid similarity measure")

    dict = {
        'js' : js,
        'cs' : cs,
        'dcs' : dcs,
    }
    dict.get(sim_measure, default)()
