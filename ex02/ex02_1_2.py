import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import convolve2d
from sklearn import datasets
from sklearn.neighbors import KernelDensity

if __name__ == "__main__":
    iris = datasets.load_iris() # Sklearn.util.Bunch
    X, y = iris.data, iris.target
    X = X[:, 0::2] # Select feature 0 and 2
    # transform features
    X *= 10
    X = np.floor(X).astype(np.int32)

    # initialize count array
    count = np.zeros((100, 100))

    # populate count array
    for i in range(X.shape[0]):
        x1, x2 = X[i]
        count[x1][x2] += 1

    fig, ax = plt.subplots(1,3)

    ax[0].imshow(count)
    ax[0].set_title("Samples from orig distribution")

    # Choice 1, using convolve2d from scipy.signal. For box kernel
    # h = 3
    # kernel = np.ones((h, h))
    # p_x1x2 = convolve2d(count, kernel, mode='same')
    # p_x1x2 /= np.sum(p_x1x2)

    # Choice 2, Using KernelDensity from sklearn
    kde = KernelDensity(kernel='gaussian', bandwidth=3).fit(X)
    x1, x2 = np.meshgrid(np.linspace(0, 100, num=100, endpoint=False), np.linspace(0, 100, num=100, endpoint=False))
    x1_ = x1.flatten()
    x2_ = x2.flatten()
    x1x2 = np.vstack((x1_, x2_)).T
    p_x1x2 = np.exp(kde.score_samples(x1x2))
    p_x1x2 = p_x1x2.reshape(100, 100, order='F')

    # Plot PDF
    ax[1].imshow(p_x1x2)
    ax[1].set_title("P(X1,X2)")

    # Sampling from PDF
    p_x1x2 = p_x1x2.reshape(-1)

    ## CDF computation
    cdf = np.cumsum(p_x1x2)

    ## Sample selection
    num_samples = 1500
    count = np.zeros((10000,))
    u = np.sort(np.random.rand(num_samples))
    idx = np.searchsorted(cdf, u)
    for i in idx:
        count[i] += 1
    
    # initialize count array
    count = count.reshape(100,100)

    ax[2].imshow(count)
    ax[2].set_title("Samples from est. distribution")
    plt.show()
