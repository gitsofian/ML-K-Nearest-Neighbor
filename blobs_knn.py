import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_blobs

np.random.seed(0)

colors = ["r", "g", "b", "k"]

data_train, labels_train = make_blobs(
    n_samples=100, cluster_std=[2.0, 3.0, 1.5], random_state=170
)
colors_train = [colors[labels_train[i]] for i in range(labels_train.shape[0])]

# Aufgabe a) -->

# <--

fig, ax = plt.subplots()
ax.scatter(data_train[:, 0], data_train[:, 1], color=colors_train)

def onclick(event):
    sample = np.array([[event.xdata, event.ydata]])

    # Aufgabe b) -->

    pred = -1 

    # <--

    ax.clear()
    ax.scatter(data_train[:, 0], data_train[:, 1], color=colors_train)
    ax.scatter(sample[:,0], sample[:,1], marker="^")
    
    plt.draw()

cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()