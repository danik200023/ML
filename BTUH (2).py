import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np
from statistics import mode
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

# del(logistic)
# logistic = SGDClassifier(loss='log')
logistic.fit(X_train_normalize, Y_train)
logistic_predict_normalize = logistic.predict(X_valid_normalize)

# del(logistic)
# logistic = SGDClassifier(loss='log')
logistic.fit(X_train_scale, Y_train)
logistic_predict_scale = logistic.predict(X_valid_scale)

# del(logistic)
# logistic = SGDClassifier(loss='log')
logistic.fit(X_train_normalize_scale, Y_train)
logistic_predict_normalize_scale = logistic.predict(X_valid_normalize_scale)

# del(logistic)
# logistic = SGDClassifier(loss='log')
logistic.fit(X_train_scale_normalize, Y_train)
logistic_predict_scale_normalize = logistic.predict(X_valid_scale_normalize)

logistic_list = ["logistic_predict", "logistic_predict_normalize", "logistic_predict_scale",
                 "logistic_predict_normalize_scale", "logistic_predict_scale_normalize"]

logistic_accuracy = [round(np.around(accuracy_score(Y_valid, logistic_predict),
                                     decimals=4), 5),
                     round(np.around(accuracy_score(Y_valid, logistic_predict_normalize),
                                     decimals=4), 5), round(np.around(accuracy_score(Y_valid, logistic_predict_scale),
                                                                      decimals=4), 5),
                     round(np.around(accuracy_score(Y_valid, logistic_predict_normalize_scale),
                                     decimals=4), 5),
                     round(np.around(accuracy_score(Y_valid, logistic_predict_scale_normalize),
                                     decimals=4), 5)]

logistic_precision = [round(np.around(precision_score(Y_valid, logistic_predict),
                                      decimals=4), 5),
                      round(np.around(precision_score(Y_valid, logistic_predict_normalize),
                                      decimals=4), 5), round(np.around(precision_score(Y_valid, logistic_predict_scale),
                                                                       decimals=4), 5),
                      round(np.around(precision_score(Y_valid, logistic_predict_normalize_scale),
                                      decimals=4), 5),
                      round(np.around(precision_score(Y_valid, logistic_predict_scale_normalize),
                                      decimals=4), 5)]

logistic_recall = [round(np.around(recall_score(Y_valid, logistic_predict),
                                   decimals=4), 5), round(np.around(recall_score(Y_valid, logistic_predict_normalize),
                                                                    decimals=4), 5),
                   round(np.around(recall_score(Y_valid, logistic_predict_scale),
                                   decimals=4), 5),
                   round(np.around(recall_score(Y_valid, logistic_predict_normalize_scale),
                                   decimals=4), 5),
                   round(np.around(recall_score(Y_valid, logistic_predict_scale_normalize),
                                   decimals=4), 5)]

logistic_f1 = [round(np.around(f1_score(Y_valid, logistic_predict),
                               decimals=4), 5), round(np.around(f1_score(Y_valid, logistic_predict_normalize),
                                                                decimals=4), 5),
               round(np.around(f1_score(Y_valid, logistic_predict_scale),
                               decimals=4), 5), round(np.around(f1_score(Y_valid, logistic_predict_normalize_scale),
                                                                decimals=4), 5),
               round(np.around(f1_score(Y_valid, logistic_predict_scale_normalize),
                               decimals=4), 5)]

logistic_roc_auc = [round(np.around(roc_auc_score(Y_valid, logistic_predict),
                                    decimals=4), 5), round(np.around(roc_auc_score(Y_valid, logistic_predict_normalize),
                                                                     decimals=4), 5),
                    round(np.around(roc_auc_score(Y_valid, logistic_predict_scale),
                                    decimals=4), 5),
                    round(np.around(roc_auc_score(Y_valid, logistic_predict_normalize_scale),
                                    decimals=4), 5),
                    round(np.around(roc_auc_score(Y_valid, logistic_predict_scale_normalize),
                                    decimals=4), 5)]

logistic_max_values_indexes = [logistic_accuracy.index(max(logistic_accuracy)),
                               logistic_precision.index(max(logistic_precision)),
                               logistic_recall.index(max(logistic_recall)),
                               logistic_f1.index(max(logistic_f1)),
                               logistic_roc_auc.index(max(logistic_roc_auc))]

print(logistic_list[max(set(logistic_max_values_indexes), key=logistic_max_values_indexes.count)])

# TN, FP, FN, TP = confusion_matrix(Y_valid, disc_predict).ravel()
# print("TN = " + str(TN))
# print("FN = " + str(FN))
# print("TP = " + str(TP))
# print("FP = " + str(FP))"""
