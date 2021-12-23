import numpy as np
np.random.seed(10)
import matplotlib.pyplot as plt
import scipy.spatial.distance
from itertools import cycle
import os

class DBSCAN:
    def __init__(self, file_path, eps, minPts) -> None:
        """
        @param file_path: path to load DBSCAN points
        @param eps: maximum radius of neighborhood
        @param minPts: minimum number of points within the eps-neighborhood of a core point
        """
        self.data = np.loadtxt(file_path)
        self.n = len(self.data)
        self.eps = eps
        self.minPts = minPts
        # initialize the variables
        self.clusters = []
        self.outliers = []
        self.pair_dist = np.zeros((self.n, self.n))
    
    def compute_pairwise_distance(self) -> dict:
        """
        compute pairwise euclidean distance and find the eps-neighborhood for each point
        return the list of core points
        """
        # calculate pairwise distance, store in triangular matrix
        self.pair_dist[np.triu_indices(self.n, 1)] = scipy.spatial.distance.pdist(self.data, metric='euclidean')
        # convert the triangular matrix to symmetric matrix
        self.pair_dist = np.where(self.pair_dist, self.pair_dist, self.pair_dist.T)

        # construct eps-neighborhood dict: {point: [eps-neighbors]}
        core_points = dict()
        for x, y in zip(*np.where(self.pair_dist <= self.eps)):
            if x not in core_points.keys():
                core_points.update({x: [y]})
            else:
                core_points[x].append(y)

        # remove points that does not satisfy minPts condition
        for k in list(core_points.keys()):
            if len(core_points[k]) < self.minPts:
                del core_points[k]

        return core_points

    def run(self):
        """
        main program to run
        """
        # dict of core_points and their eps-neighbors
        core_points = self.compute_pairwise_distance()
        # an array to record whether a point is already scanned before
        visited = np.array([0] * self.n)
        
        def dfs(p, cluster) -> list:
            """
            retrieve all points density-reachable from curren point p 
            @param p: the current point
            @param cluster: the cluster where p is in
            return the formed cluster
            """
            if p in core_points.keys():
                for neighbor in core_points[p]:
                    if neighbor != p and visited[neighbor] == 0:
                        visited[neighbor] = 1
                        cluster.append(neighbor)
                        cluster = dfs(neighbor, cluster)
            
            return cluster

        # search for clusters when there are non-visited core points
        while len(np.intersect1d(np.where(visited == 0)[0], list(core_points.keys()))) > 0:
            # randomly pick one point from core points to start the scan
            nonvisit_core_points = np.intersect1d(np.where(visited == 0)[0], list(core_points.keys()))
            curr = np.random.choice(nonvisit_core_points, 1)[0]
            visited[curr] = 1
            self.clusters = self.clusters + [dfs(curr, [curr])]

        print("There are %d clusters" % (len(self.clusters)))

        # if there are non-visited points, these are outliers
        outliers = np.where(visited == 0)[0]
        if len(outliers) > 0:
            print("There are %d outliers" % (len(outliers)))
            self.outliers = outliers

    def plot_clusters(self):
        """
        plot clustering results
        """
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)
        # plot clusters
        color = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#bcbd22', '#17becf'])
        for i in range(len(self.clusters)):
            scatter = ax.scatter(self.data[self.clusters[i], 0], self.data[self.clusters[i], 1], c=next(color), s=10)
        # plot outliers
        if len(self.outliers) > 0:
            scatter = ax.scatter(self.data[self.outliers, 0], self.data[self.outliers, 1], c="black", s=10)
        ax.set_title("DBSCAN Clustering, eps = %.1f, minPts = %d" % (self.eps, self.minPts))
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        if not os.path.exists("clustering/pictures/DBSCAN"):
            os.makedirs("clustering/pictures/DBSCAN")
        save_path = "clustering/pictures/DBSCAN/eps_%.1f_minPts_%d.png" % (self.eps, self.minPts)  
        plt.savefig(save_path)
        # plt.show()



print("eps = 3, minPts = 5")
dbscan = DBSCAN("clustering/dataset/DBSCAN_Points.txt", eps=3, minPts=5)
dbscan.run()
dbscan.plot_clusters()
print()

print("eps = 2.5, minPts = 20")
dbscan = DBSCAN("clustering/dataset/DBSCAN_Points.txt", eps=2.5, minPts=20)
dbscan.run()
dbscan.plot_clusters()
print()

print("eps = 2, minPts = 20")
dbscan = DBSCAN("clustering/dataset/DBSCAN_Points.txt", eps=2, minPts=20)
dbscan.run()
dbscan.plot_clusters()
print()

print("eps = 2, minPts = 15")
dbscan = DBSCAN("clustering/dataset/DBSCAN_Points.txt", eps=2, minPts=15)
dbscan.run()
dbscan.plot_clusters()
print()

    