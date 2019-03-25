import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score

dataset = pd.read_csv('data.csv')

X = dataset.iloc[:, 1:179].values
y = dataset.iloc[:, 179:].values

for i in range(len(y)):
    if y[i] == 1:
        y[i] = 1
    else:
        y[i] = 0

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling (MUST BE APPLIED IN DIMENSIONALITY REDUCTION)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# # Applying kPCA (non-linear)
# from sklearn.decomposition import KernelPCA
# kpca = KernelPCA(n_components = 2, kernel = 'rbf')
# X_train = kpca.fit_transform(X_train)
# X_test = kpca.transform(X_test)

# Fitting Naive Bayes Classification to the Training set
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X_train, y_train.ravel())

# Predicting the Test set results
y_pred = clf.predict(X_test)
predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

score = clf.score(X_test, y_test)
print('Accuracy of sklearn: {0}%'.format(score*100))



