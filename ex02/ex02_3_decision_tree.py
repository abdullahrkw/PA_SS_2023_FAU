from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class Node(object):
    def __init__(self, samples, labels, depth) -> None:
        self.samples = samples
        self.labels = labels
        self.feature = None
        self.threshold = None
        self.impurity = None
        self.support_samples = None
        self.splitted = False
        self.visited = False
        self.left_child = None
        self.right_child = None
        self.depth = depth

    def calculate_gini(self, labels):
        n_classes = np.unique(y)
        g = 0
        for class_ in n_classes:
            p = len(labels[labels==class_])/len(labels)
            g += (p*(1-p))
        return g


    def calculate_node_impurity(self, criterion="gini"):
        x, y = self.samples, self.labels
        impurity = np.inf
        n_samples, n_features = x.shape
        n_classes = np.unique(y)
        selected_feature = None
        selected_threshold = None

        # Already a pure node
        if len(n_classes) == 1:
            return 0, selected_feature, selected_threshold

        # Iterate over all features and all unique values for single feature
        for feature in range(n_features):
            unique_vals = np.sort(np.unique(x[:,feature]))
            for val in unique_vals:
                left = x[:, feature] < val
                right = x[:, feature] >= val
                left_labels = y[left]
                right_labels = y[right]
                
                # skip if no sample in left or right split
                if not len(left_labels) > 0 or not len(right_labels) > 0:
                    continue

                # compute gini for left and right and do weighted sum
                g_l = self.calculate_gini(left_labels)
                g_r = self.calculate_gini(right_labels)
                g = ((len(left_labels)/len(y))*g_l) + ((len(right_labels)/len(y))*g_r)
                # update g if impurity is decreased
                if g < impurity:
                    impurity = g
                    selected_feature = feature
                    selected_threshold = val
        return impurity, selected_feature, selected_threshold

class MyDecisionTreeClassifier(object):
    def __init__(self, criterion="gini", max_depth=None, min_samples_leaf=1) -> None:
        self.tree_ = []
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.supported_criterion = ["gini"]
        self.criterion = criterion
        if self.criterion not in self.supported_criterion:
            raise ValueError(f"{self.criterion} is not supported for impurity calculation")

    def fit(self, x, y):
        stack = deque()
        root_node = Node(x, y, 0)
        self.tree_.append(root_node)
        stack.append(root_node)
        while len(stack) > 0:
            node = stack.pop()
            x, y = node.samples, node.labels
            if node.visited:
                continue
            node.visited = True
            g, feature, threshold = node.calculate_node_impurity(criterion=self.criterion)
            print("g", g, "feature", feature, "threshold", threshold)
            node.impurity = g
            # pure leaf node
            if node.impurity == 0.0 and feature is None:
                node.splitted = False
                node.support_samples = len(node.samples)
                print(node.support_samples)
            # possibly impure leaf node
            elif node.depth == self.max_depth:
                node.splitted = False
                node.support_samples = len(node.samples)
                print(node.support_samples)
            # node would be splitted now
            else:
                node.splitted = True
                node.feature = feature
                node.threshold = threshold
                left = x[:, feature] < threshold
                right = x[:, feature] >= threshold
                node.left_child = Node(x[left], y[left], node.depth + 1)
                node.right_child = Node(x[right], y[right], node.depth + 1)
                stack.append(node.right_child)
                stack.append(node.left_child)
        return self
    
    def predict_single_sample(self, x):
        node = self.tree_[0]
        while node.splitted:
            if x[node.feature] < node.threshold:
                node = node.left_child
            else:
                node = node.right_child
        # find majority class
        return np.argmax(np.bincount(node.labels))


    def predict(self, x):
        y = map(lambda x_: self.predict_single_sample(x_), x)
        return np.fromiter(y, dtype=np.int32)
    
    def score(self, x, y):
        y_pred = self.predict(x)
        return accuracy_score(y, y_pred)


if __name__ == "__main__":
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    # Stratified splitting
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    d = []
    accuracy = []
    for i in range(1,40):
        clf = MyDecisionTreeClassifier(max_depth=i)
        clf = clf.fit(x_train, y_train)
        acc = clf.score(x_test, y_test)
        d.append(i)
        accuracy.append(acc)
    plt.plot(d, accuracy)
    plt.xlabel("Depth")
    plt.ylabel("Accuracy")
    plt.title("CART")
    plt.show()
