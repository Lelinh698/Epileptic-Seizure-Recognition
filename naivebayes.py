import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from sklearn.metrics import accuracy_score, confusion_matrix,roc_auc_score


dataset = pd.read_csv('data.csv')
print("The data has {} observations and {} features".format(dataset.shape[0], dataset.shape[1]))

class_names=[0,1]

X = dataset.iloc[:, 1:179].values
y = dataset.iloc[:, 179:].values

#Converting class label from 5 to binary, 0 for Normal Brain, 1 for Epilepsy
for i in range(len(y)):
    if y[i] == 1:
        y[i] = 1
    else:
        y[i] = 0

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

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
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,title='Confusion matrix, without normalization')
plt.show()


f = open('addFile.csv')
file_line = f.readlines()
second_line = file_line[1].strip()

def add_row(second_line):
    with open('data.csv', 'a') as f:
        f.write(second_line + "\n")

add_row(second_line)