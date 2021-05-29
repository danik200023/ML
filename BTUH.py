import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

data = pd.read_csv('data/pima-indians-diabetes.csv')
X = data.drop(data.columns[-1], axis=1)
Y = data[data.columns[-1]].astype('int')
X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.3)

logistic = SGDClassifier(loss='log')

clf = MultinomialNB()

disc = LinearDiscriminantAnalysis()

neural = MLPClassifier()


logistic.fit(X_train, Y_train)
logflgfg = logistic.predict(X_valid)
clf.fit(X_train, Y_train)
caccaadadad = clf.predict(X_valid)


disc.fit(X_train, Y_train)
dicsadadcasd = disc.predict(X_valid)
neural.fit(X_train, Y_train)
neudadasdadas = neural.predict(X_valid)
support = SVC(kernel="linear", C=0.025)
support.fit(X_train, Y_train)
support_pred = support.predict(X_valid)
print(str(
                round(np.around(accuracy_score(Y_valid, logflgfg),
                                decimals=4) * 100, 5)) + "%")

print(str(
                round(np.around(accuracy_score(Y_valid, caccaadadad),
                                decimals=4) * 100, 5)) + "%")

print(str(
                round(np.around(accuracy_score(Y_valid, dicsadadcasd),
                                decimals=4) * 100, 5)) + "%")

print(str(
                round(np.around(accuracy_score(Y_valid, neudadasdadas),
                                decimals=4) * 100, 5)) + "%")

print(str(
                round(np.around(accuracy_score(Y_valid, support_pred),
                                decimals=4) * 100, 5)) + "%")