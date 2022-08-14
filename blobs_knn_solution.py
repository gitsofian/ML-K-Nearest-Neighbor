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
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(data_train, colors_train)

fig, ax = plt.subplots()
ax.scatter(data_train[:, 0], data_train[:, 1], color=colors_train)

def onclick(event):
    sample = np.array([[event.xdata, event.ydata]])

    # Aufgabe b) -->
    pred = clf.predict(sample)
    neigh_dist, neigh_idx = clf.kneighbors(sample)
    neigh_idx = neigh_idx[0]

    ax.clear()
    ax.scatter(data_train[:, 0], data_train[:, 1], color=colors_train)
    ax.scatter(sample[:,0], sample[:,1], color=pred, marker="^")
    for idx in range(neigh_idx.shape[0]):
        ax.plot([sample[0,0], data_train[neigh_idx[idx], 0]], [sample[0,1], data_train[neigh_idx[idx], 1]], c=colors_train[neigh_idx[idx]])
    plt.draw()

cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()