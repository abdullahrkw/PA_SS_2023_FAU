import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.neighbors import KernelDensity

if __name__ == "__main__":
    iris = datasets.load_iris() # Sklearn.util.Bunch
    X, y = iris.data, iris.target
    X = X[:, 0::2] # Select feature 0 and 2
    # transform features
    X *= 10
    X = np.floor(X).astype(np.int32)

    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')

    kde = KernelDensity(kernel='gaussian', bandwidth=2.5).fit(X)
    x1, x2 = np.meshgrid(np.linspace(0, 100, num=100, endpoint=False), np.linspace(0, 100, num=100, endpoint=False))
    x1_ = x1.flatten()
    x2_ = x2.flatten()
    x1x2 = np.vstack((x1_, x2_)).T
    
    p_x1x2 = np.exp(kde.score_samples(x1x2))
    p_x1x2 = p_x1x2.reshape(100, 100, order='F')
    pnt3d = ax.scatter(x1,x2,p_x1x2, c=p_x1x2)
    ax.invert_yaxis()
    cbar=plt.colorbar(pnt3d)

    plt.show()
