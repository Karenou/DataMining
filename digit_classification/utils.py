import numpy as np
import matplotlib.pylab as plt


import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

def plot_class_dist(lbs, save_path):
    class_dist = lbs.value_counts()
    class_dist = class_dist.reset_index()
    class_dist.columns = ["class", "cnt"]
    class_dist = class_dist.sort_values(["class"])

    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar(class_dist["class"], class_dist["cnt"])
    ax.set_xticks(np.arange(10), [0,1,2,3,4,5,6,7,8,9])
    ax.set_title('Distribution of class labels')
    plt.xticks(np.arange(10))
    # plt.show()
    if save_path:
        plt.savefig(save_path)

def plot_img(idx, img_arr, label, pred):
    plt.imshow(img_arr[idx], vmin=0, vmax=1)
    plt.title("True label: %d, Predict label: %d" % (label.iloc[idx], pred.iloc[idx]))
    plt.show()

def prepare_data(data, labels, n_dims=(28, 28), batch_size=64, shuffle=True, drop_last=False):
    n_cols = n_dims[0] * n_dims[1]
    image_arr = data[data.columns[:n_cols]].to_numpy()
    image_arr = image_arr.reshape(len(data), n_dims[0], n_dims[1])
    
    X = torch.unsqueeze(torch.from_numpy(image_arr), dim=1)
    y = torch.from_numpy(labels.to_numpy()).long()

    data_tensor = torch.utils.data.TensorDataset(X, y)
    data_loader = DataLoader(data_tensor, batch_size=batch_size, num_workers=1, shuffle=shuffle, drop_last=drop_last)
    return data_loader

def calculat_acc(output, target):
    output = nn.functional.softmax(output, dim=1)
    output = torch.argmax(output, dim=1)
    correct = torch.sum(output == target).item()
    return correct