import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
import os

class FuzzyCluster(object):

    def __init__(self, file_path):
        """
        @param file_path: path to load the txt file
        """
        # load data as numpy array
        self.data = np.loadtxt(file_path)
        # number of data
        self.n = len(self.data)
        # separate the data, label
        self.data, self.label = self.data[:, :2], self.data[:, 2],
        # number of clusters
        self.k = len(np.unique(self.label))

        # randomly choose three points as initial cluster centers
        random_sample = np.random.choice(np.arange(self.n), self.k)
        self.clusters = self.data[random_sample]

        # initilize the partition matrix to store the distance later
        self.partition_matrix = np.zeros((self.n, self.k))

    def expectation_step(self, p=2):
        """
        compute the partition matrix
        @param p: hyperparameter to calculate the weight
        """
        for j in range(self.k):
            self.partition_matrix[:, j] = np.power(np.linalg.norm(self.data-self.clusters[j], ord=2, axis=1), 2 / (p-1))

        # for data to be set as cluter center, weight of this cluter = 1, while weight of other cluters are 0
        cluster_center = np.where(np.prod(self.partition_matrix, axis=1)==0)[0]
        if len(cluster_center) > 0:
            self.partition_matrix[cluster_center] = np.where(self.partition_matrix[cluster_center] == 0, 1, 0)
        
        self.partition_matrix = np.divide(1.0, self.partition_matrix, 
                                out=np.zeros_like(self.partition_matrix), 
                                where=self.partition_matrix != 0)

        self.partition_matrix = self.partition_matrix / np.sum(self.partition_matrix, axis=1, keepdims=True)

    def maximization_step(self, p=2):
        """
        calculate the centroids according to the partition matrix
        @param p: hyperparameter to calculate the weight
        """
        for j in range(self.k):
            weights = np.power(self.partition_matrix[:, j], p).reshape(self.n, 1)
            self.clusters[j] = np.sum(weights * self.data, axis=0) / np.sum(weights)
    
    def compute_sse(self, p=2):
        """
        calculate the SSE over all data and all centroids
        @param p: a hyperparameter of the weight
        """
        sse = 0.0
        for j in range(self.k):
            squared_dist = np.square(np.linalg.norm(self.data - self.clusters[j], ord=2, axis=1))
            sse = sse + np.dot(np.power(self.partition_matrix[:, j], p), squared_dist)
        return sse

    def print_clusters_and_sse(self, step, p=2):
        """
        print out the centers and SSE after each updated step
        @param step
        @param p: hyperparameter for calculating SSE
        """
        print("Step %d" % (step + 1))
        print("The current SSE is %.4f" % self.compute_sse(p))

        print("The updated centers are:")
        for j in range(self.k):
            print("center %d: (%.4f, %.4f)" % (j+1, self.clusters[j][0], self.clusters[j][1]))
        print()

    def run(self, num_iterations=5, epsilon=1e-3, p=2):
        """
        @param num_iterations: number of EM steps
        @param epsilon: a threshold used to check whether the clusters change significantly after each EM step
        @param p: hyperparameter to calculate SSE
        """

        for step in range(num_iterations):
            prev_clusters = self.clusters.copy()

            self.expectation_step(p)
            self.maximization_step(p)

            self.print_clusters_and_sse(step, p)

            # converging criterion, if 2-norm of the change of cluster centroids is smaller than epsilon
            if np.linalg.norm(prev_clusters - self.clusters, ord=2) <= epsilon:
                print("The clustering is already converged at step %d" % (step + 1))

                # plot clustering result
                self.plot_clusters(step)
                break

        self.plot_clusters(step)

    def plot_clusters(self, step):
        """
        plot clustering results compared to original label
        @param step: number of steps
        """

        # plot the true clustering first 
        fig = plt.figure(figsize=(12,5))
        ax1 = fig.add_subplot(121)
        scatter1 = ax1.scatter(self.data[:, 0], self.data[:, 1], c=self.label, s=10)
        ax1.set_title("True clustering")
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        plt.colorbar(scatter1)

        pred = np.argmax(self.partition_matrix, axis=1)
        ax2 = fig.add_subplot(122)
        scatter2 = ax2.scatter(self.data[:, 0], self.data[:, 1], c=pred, s=10)
        ax2.set_title("Fuzzy clustering after %d steps" % (step + 1))
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        plt.colorbar(scatter2)

        if not os.path.exists("./fuzzy_cluster"):
            os.makedirs("./fuzzy_cluster")
        save_path = "./fuzzy_cluster/step_%d.png" % (step+1)  
        plt.savefig(save_path)
        # plt.show()


clustering = FuzzyCluster(file_path="EM_Points.txt")
clustering.run(num_iterations=15, epsilon=1e-3, p=4)