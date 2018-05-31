
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
from scipy.spatial import distance

def euc(a,b):
    return distance.euclidean(a,b)

class BareBonesKNN():
    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        pass

    def predict(self, x_test):
        predictions = []
        for row in x_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions

    def closest(self, row):
        best_dist = euc(row, self.x_train[0])
        best_index = 0
        for i in range(1, len(self.x_train)):
            dist = euc(row , self.x_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i
        return self.y_train[best_index]

iris = load_iris()
test_idx = [8, 72, 123]

#training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis =0)


#testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]


clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

print (iris.target_names[(test_target)])
print (test_data)





print (iris.target_names[clf.predict(test_data)])


print("###############")
my_classifier = BareBonesKNN()

# from sklearn import tree
# my_classifier = tree.DecisionTreeClassifier()

my_classifier.fit(train_data, train_target)
predictions = my_classifier.predict(test_data)

print(iris.target_names[my_classifier.predict(test_data)])

print("Accuracy")
from sklearn.metrics import accuracy_score
print (accuracy_score(test_target,predictions))


# In[2]:



from sklearn import datasets



iris = datasets.load_iris()

x = iris.data
y = iris.target


from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y , test_size = .3)

print(y_train)
print(y_test)

