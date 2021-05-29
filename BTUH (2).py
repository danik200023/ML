import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np
from sklearn import preprocessing


data = pd.read_csv('data/pima-indians-diabetes.csv')
# data = pd.read_csv('data/water_purification.csv')
X = data.drop(data.columns[-1], axis=1)
Y = data[data.columns[-1]].astype('int')
X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.3)
X_train_normalize = preprocessing.normalize(X_train)
X_train_scale = preprocessing.scale(X_train)
X_train_normalize_scale = preprocessing.scale(X_train_normalize)
X_train_scale_normalize = preprocessing.normalize(X_train_scale)
X_valid_normalize = preprocessing.normalize(X_valid)
X_valid_scale = preprocessing.scale(X_valid)
X_valid_normalize_scale = preprocessing.scale(X_valid_normalize)
X_valid_scale_normalize = preprocessing.normalize(X_valid_scale)

logistic = SGDClassifier(loss='log')
clf = MultinomialNB()
disc = LinearDiscriminantAnalysis()
neural = MLPClassifier()
support = SVC(kernel="linear", C=0.025)






logistic.fit(X_train, Y_train)
logistic_predict = logistic.predict(X_valid)

del(logistic)
logistic = SGDClassifier(loss='log')
logistic.fit(X_train_normalize, Y_train)
logistic_predict_normalize = logistic.predict(X_valid_normalize)

del(logistic)
logistic = SGDClassifier(loss='log')
logistic.fit(X_train_scale, Y_train)
logistic_predict_scale = logistic.predict(X_valid_scale)

del(logistic)
logistic = SGDClassifier(loss='log')
logistic.fit(X_train_normalize_scale, Y_train)
logistic_predict_normalize_scale = logistic.predict(X_valid_normalize_scale)

del(logistic)
logistic = SGDClassifier(loss='log')
logistic.fit(X_train_scale_normalize, Y_train)
logistic_predict_scale_normalize = logistic.predict(X_valid_scale_normalize)

print("logistic_predict = " + str(
                round(np.around(accuracy_score(Y_valid, logistic_predict),
                                decimals=4) * 100, 5)) + "%")

print("logistic_predict_normalize = " + str(
                round(np.around(accuracy_score(Y_valid, logistic_predict_normalize),
                                decimals=4) * 100, 5)) + "%")

print("logistic_predict_scale = " + str(
                round(np.around(accuracy_score(Y_valid, logistic_predict_scale),
                                decimals=4) * 100, 5)) + "%")

print("logistic_predict_normalize_scale = " + str(
                round(np.around(accuracy_score(Y_valid, logistic_predict_normalize_scale),
                                decimals=4) * 100, 5)) + "%")

print("logistic_predict_scale_normalize = " + str(
                round(np.around(accuracy_score(Y_valid, logistic_predict_scale_normalize),
                                decimals=4) * 100, 5)) + "%\n")


print("logistic_prec = " + str(
                round(np.around(precision_score(Y_valid, logistic_predict),
                                decimals=4) * 100, 5)) + "%")

print("logistic_prec_normalize = " + str(
                round(np.around(precision_score(Y_valid, logistic_predict_normalize),
                                decimals=4) * 100, 5)) + "%")

print("logistic_prec_scale = " + str(
                round(np.around(precision_score(Y_valid, logistic_predict_scale),
                                decimals=4) * 100, 5)) + "%")

print("logistic_prec_normalize_scale = " + str(
                round(np.around(precision_score(Y_valid, logistic_predict_normalize_scale),
                                decimals=4) * 100, 5)) + "%")

print("logistic_prec_scale_normalize = " + str(
                round(np.around(precision_score(Y_valid, logistic_predict_scale_normalize),
                                decimals=4) * 100, 5)) + "%\n")

print("logistic_recall = " + str(
                round(np.around(recall_score(Y_valid, logistic_predict),
                                decimals=4) * 100, 5)) + "%")

print("logistic_recall_normalize = " + str(
                round(np.around(recall_score(Y_valid, logistic_predict_normalize),
                                decimals=4) * 100, 5)) + "%")

print("logistic_recall_scale = " + str(
                round(np.around(recall_score(Y_valid, logistic_predict_scale),
                                decimals=4) * 100, 5)) + "%")

print("logistic_recall_normalize_scale = " + str(
                round(np.around(recall_score(Y_valid, logistic_predict_normalize_scale),
                                decimals=4) * 100, 5)) + "%")

print("logistic_recall_scale_normalize = " + str(
                round(np.around(recall_score(Y_valid, logistic_predict_scale_normalize),
                                decimals=4) * 100, 5)) + "%\n")

print("logistic_f1 = " + str(
                round(np.around(f1_score(Y_valid, logistic_predict),
                                decimals=4) * 100, 5)) + "%")

print("logistic_f1_normalize = " + str(
                round(np.around(f1_score(Y_valid, logistic_predict_normalize),
                                decimals=4) * 100, 5)) + "%")

print("logistic_f1_scale = " + str(
                round(np.around(f1_score(Y_valid, logistic_predict_scale),
                                decimals=4) * 100, 5)) + "%")

print("logistic_f1_normalize_scale = " + str(
                round(np.around(f1_score(Y_valid, logistic_predict_normalize_scale),
                                decimals=4) * 100, 5)) + "%")

print("logistic_f1_scale_normalize = " + str(
                round(np.around(f1_score(Y_valid, logistic_predict_scale_normalize),
                                decimals=4) * 100, 5)) + "%\n")

print("logistic_auc = " + str(
                round(np.around(roc_auc_score(Y_valid, logistic_predict),
                                decimals=4) * 100, 5)) + "%")

print("logistic_auc_normalize = " + str(
                round(np.around(roc_auc_score(Y_valid, logistic_predict_normalize),
                                decimals=4) * 100, 5)) + "%")

print("logistic_auc_scale = " + str(
                round(np.around(roc_auc_score(Y_valid, logistic_predict_scale),
                                decimals=4) * 100, 5)) + "%")

print("logistic_auc_normalize_scale = " + str(
                round(np.around(roc_auc_score(Y_valid, logistic_predict_normalize_scale),
                                decimals=4) * 100, 5)) + "%")

print("logistic_auc_scale_normalize = " + str(
                round(np.around(roc_auc_score(Y_valid, logistic_predict_scale_normalize),
                                decimals=4) * 100, 5)) + "%\n")




"""clf.fit(X_train, Y_train)
clf_predict = clf.predict(X_valid)

disc.fit(X_train, Y_train)
disc_predict = disc.predict(X_valid)

neural.fit(X_train, Y_train)
neural_predict = neural.predict(X_valid)

support.fit(X_train, Y_train)
support_predict = support.predict(X_valid)



print(str(
                round(np.around(accuracy_score(Y_valid, logistic_predict),
                                decimals=4) * 100, 5)) + "%")

print(str(
                round(np.around(accuracy_score(Y_valid, clf_predict),
                                decimals=4) * 100, 5)) + "%")

print(str(
                round(np.around(accuracy_score(Y_valid, disc_predict),
                                decimals=4) * 100, 5)) + "%")

print(str(
                round(np.around(accuracy_score(Y_valid, neural_predict),
                                decimals=4) * 100, 5)) + "%")

print(str(
                round(np.around(accuracy_score(Y_valid, support_predict),
                                decimals=4) * 100, 5)) + "%")




TN, FP, FN, TP = confusion_matrix(Y_valid, disc_predict).ravel()
#print("TN = " + str(TN))
#print("FN = " + str(FN))
#print("TP = " + str(TP))
#print("FP = " + str(FP))"""
