import numpy as np
import matplotlib.pyplot as plt

def plot_history(history, num_class=2, ylim=(0.94, 1.0), fig_save_path=None):
    """
    @param history: a dict saving the performance metrics
    @param num_class: 2 or 10
    @param ylim: a tuple 
    @param fig_save_path: path to save the fig if not None
    """
    if num_class == 2:
        x = history["n_hidden"]
    else:
        x = list(range(len(history["n_hidden"])))

    plt.figure(figsize=(16,4))

    plt.subplot(1, 2, 1)
    plt.title("Performance on Train Set")
    plt.ylim(*ylim)
    if num_class == 2:
        plt.xlabel("n_hidden")
    plt.plot(x, history["train_acc"], label="accuracy")
    plt.plot(x, history["train_auc"], label="auc score")
    plt.plot(x, history["train_f1"], label="f1 score")
    plt.legend(loc="lower right")

    plt.subplot(1, 2, 2)
    plt.ylim(*ylim)
    if num_class == 2:
        plt.xlabel("n_hidden")
    plt.title("Performance on Validation Set")
    plt.plot(x, history["val_acc"], label="accuracy")
    plt.plot(x, history["val_auc"], label="auc score")
    plt.plot(x, history["val_f1"], label="f1 score")
    plt.legend(loc="lower right")

    if fig_save_path:
        plt.savefig(fig_save_path)
        
    # plt.show()