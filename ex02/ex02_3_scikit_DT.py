import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets, tree
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    # CART
    iris = datasets.load_iris() #
    X, y = iris.data, iris.target

    # Stratified splitting
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    print(np.bincount(y_test))
    print(np.bincount(y_train))

    d = []
    accuracy = []
    for i in range(1,40):
        clf = tree.DecisionTreeClassifier(criterion="gini", max_depth=i)
        clf = clf.fit(x_train, y_train)
        acc = clf.score(x_test, y_test)
        d.append(i)
        accuracy.append(acc)
        # tree.plot_tree(clf)
        # plt.show()
    plt.plot(d, accuracy)
    plt.xlabel("Depth")
    plt.ylabel("Accuracy")
    plt.title("CART")
    plt.show()