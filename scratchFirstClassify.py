from sklearn import datasets  # , tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from scipy.spatial import distance


# from sklearn.neighbors import KNeighborsClassifier
def euclid(a, b):
    return distance.euclidean(a, b)


class KNN:
    def fit(self, X, Y):
        self.x = X
        self.y = Y

    def predict(self, x):
        predictions = []
        for row in x:
            label = self.closest(row)
            predictions.append(label)

        return predictions

    def closest(self, row):
        best_dist = euclid(row, self.x[0])
        best_index = 0
        for i in range(1, len(self.x)):
            dist = euclid(row, self.x[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i
        return self.y[best_index]


iris = datasets.load_iris()

x = iris.data
y = iris.target


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)
# my_classifier = tree.DecisionTreeClassifier()

# my_classifier = KNeighborsClassifier()


my_classifier = KNN()
my_classifier.fit(x_train, y_train)

predictions = my_classifier.predict(x_test)
print(predictions)
print(accuracy_score(y_test, predictions))
