import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
import os


class NestLoop:

    def __init__(self, file_path:str, r:int, p:float, m_blocks:int) -> None:
        """
        @param file_path: path to load txt data
        @param r: distance threshold
        @param p: at least fraction p of the data lies greater than distance r from object o
        @param m_blocks: partition the data into m blocks
        """
        self.data = np.loadtxt(file_path)
        # add a dummy column to mark whether the data is an outlier, 1: outlier, 0: non-outlier
        label = np.empty((len(self.data), 1))
        label[:] = np.nan
        self.data = np.concatenate((self.data, label), axis=1)
        self.n = len(self.data)
        self.m_blocks = m_blocks
        self.blocks = self.split_into_blocks()
        self.r = r
        self.k = int(p * len(self.data))

    def split_into_blocks(self) -> list:
        """
        split data into m blocks
        """
        blocks = []
        size = self.n // self.m_blocks
        i = 0
        for c in range(self.m_blocks):
            if c != self.m_blocks - 1:
                blocks.append(self.data[i:i+size])
                i += size
            else:
                blocks.append(self.data[i:])
        return blocks

    def mark_non_outliers(self, arr1, arr2=None) -> np.array:
        """
        if arr2 is None, compute pairwise distance within the first array. otherwise, compute pairwise distance across two arrays
        mark a point as non-outlier if number of points within r-distance is larger than k
        """
        for i, ti in enumerate(arr1):
            cnt = 0

            if arr2 is None:
                for j, tj in enumerate(arr1):
                    if i != j:
                        dist = np.linalg.norm(ti[:2]-tj[:2], ord=2)
                        if dist <= self.r:
                            cnt += 1
            else:
                for j, tj in enumerate(arr2):
                    dist = np.linalg.norm(ti[:2]-tj[:2], ord=2)
                    if dist <= self.r:
                        cnt += 1
            
            # mark as non-outlier
            if cnt > self.k:
                arr1[i][2] = 0
        
        return arr1

    def run(self) -> np.array:
        
        # mark whether the block has served as the first array
        as_first_arr = [0] * self.m_blocks
        # fill the first array with the first block of data initially
        arr1_idx = 0
        as_first_arr[arr1_idx] = 1
        arr1 = self.blocks[arr1_idx].copy()

        for step in range(self.m_blocks):
            arr1 = self.mark_non_outliers(arr1)
            arr2_idx = None

            for i in range(self.m_blocks):
                if i != arr1_idx:
                    # save a block which has never served as the first array, for last
                    if as_first_arr[i] == 0 and arr2_idx is None:
                        arr2_idx = i
                        continue
                    else:
                        arr2 = self.blocks[i].copy()
                        arr1 = self.mark_non_outliers(arr1, arr2)

            if arr2_idx is not None:
                arr2 = self.blocks[arr2_idx].copy()
                arr1 = self.mark_non_outliers(arr1, arr2)

            # update the block, set unmarked data as outliers
            arr1[:, 2] = np.nan_to_num(arr1[:, 2], nan=1)
            self.blocks[arr1_idx] = arr1.copy()

            # swap arr1 and arr2 if arr2 has not been put in arr1
            if arr2_idx is not None:
                as_first_arr[arr2_idx] = 1
                arr1_idx, arr1, arr2 = arr2_idx, arr2, arr1
        
        data = np.concatenate(self.blocks, axis=0)
        outliers = data[np.where(data[:, 2] == 1)]
        return data, outliers

    def plot_outliers(self):
        """
        plot the outliers
        """
        data = np.concatenate(self.blocks, axis=0)
        outliers = data[np.where(data[:, 2] == 1)]
        normal_pts = data[np.where(data[:, 2] == 0)]
        
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)
        scatter = ax.scatter(normal_pts[:, 0], normal_pts[:, 1], c="black", s=5)
        scatter = ax.scatter(outliers[:, 0], outliers[:, 1], c="red", s=5)
        ax.set_title("Nested Loop Outlier Detection (%d outliers), r = %.1f, k = %d" % (len(outliers), self.r, self.k))
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        if not os.path.exists("./NestedLoop"):
            os.makedirs("./NestedLoop")
        save_path = "./NestedLoop/r_%.1f_k_%d.png" % (self.r, self.k)
        plt.savefig(save_path)
        # plt.show()


# have tune parameters: r in [10, 12.5, 15], p in [0.01, 0.015, 0.02], m_block in [3, 4]
r, p, m_block = 15, 0.01, 3
model = NestLoop("Nested_Points.txt", r=r, p=p, m_blocks=m_block)
data, outliers = model.run()
model.plot_outliers()

print("The parameters are: r = %.1f, p = %.3f, m_blocks = %d" % (r, p, m_block))
print("There are %d outliers out of %d data" % (len(outliers), len(data)))
print("The coordinates of the outliers are: ")
print(outliers[:, :2])
