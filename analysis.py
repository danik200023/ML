import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import BaggingClassifier


def logistic_regression(window, x_train, x_valid, y_train, y_valid, x_train_normalize, x_train_scale,
                        x_train_normalize_scale, x_train_scale_normalize, x_valid_normalize, x_valid_scale,
                        x_valid_normalize_scale, x_valid_scale_normalize):
    logistic = SGDClassifier(loss='log')
    logistic.fit(x_train, y_train)
    logistic_predict = logistic.predict(x_valid)
    logistic.fit(x_train_normalize, y_train)
    logistic_predict_normalize = logistic.predict(x_valid_normalize)
    logistic.fit(x_train_scale, y_train)
    logistic_predict_scale = logistic.predict(x_valid_scale)
    logistic.fit(x_train_normalize_scale, y_train)
    logistic_predict_normalize_scale = logistic.predict(x_valid_normalize_scale)
    logistic.fit(x_train_scale_normalize, y_train)
    logistic_predict_scale_normalize = logistic.predict(x_valid_scale_normalize)
    window.logistic_accuracy = [round(np.around(accuracy_score(y_valid, logistic_predict),
                                         decimals=4), 5),
                         round(np.around(accuracy_score(y_valid, logistic_predict_normalize),
                                         decimals=4), 5),
                         round(np.around(accuracy_score(y_valid, logistic_predict_scale),
                                         decimals=4), 5),
                         round(np.around(accuracy_score(y_valid, logistic_predict_normalize_scale),
                                         decimals=4), 5),
                         round(np.around(accuracy_score(y_valid, logistic_predict_scale_normalize),
                                         decimals=4), 5)]

    window.logistic_precision = [round(np.around(precision_score(y_valid, logistic_predict),
                                          decimals=4), 5),
                          round(np.around(precision_score(y_valid, logistic_predict_normalize),
                                          decimals=4), 5),
                          round(np.around(precision_score(y_valid, logistic_predict_scale),
                                          decimals=4), 5),
                          round(np.around(precision_score(y_valid, logistic_predict_normalize_scale),
                                          decimals=4), 5),
                          round(np.around(precision_score(y_valid, logistic_predict_scale_normalize),
                                          decimals=4), 5)]

    window.logistic_recall = [round(np.around(recall_score(y_valid, logistic_predict),
                                       decimals=4), 5),
                       round(np.around(recall_score(y_valid, logistic_predict_normalize),
                                       decimals=4), 5),
                       round(np.around(recall_score(y_valid, logistic_predict_scale),
                                       decimals=4), 5),
                       round(np.around(recall_score(y_valid, logistic_predict_normalize_scale),
                                       decimals=4), 5),
                       round(np.around(recall_score(y_valid, logistic_predict_scale_normalize),
                                       decimals=4), 5)]

    window.logistic_f1 = [round(np.around(f1_score(y_valid, logistic_predict),
                                   decimals=4), 5), round(np.around(f1_score(y_valid, logistic_predict_normalize),
                                                                    decimals=4), 5),
                   round(np.around(f1_score(y_valid, logistic_predict_scale),
                                   decimals=4), 5), round(np.around(f1_score(y_valid, logistic_predict_normalize_scale),
                                                                    decimals=4), 5),
                   round(np.around(f1_score(y_valid, logistic_predict_scale_normalize),
                                   decimals=4), 5)]

    window.logistic_roc_auc = [round(np.around(roc_auc_score(y_valid, logistic_predict),
                                        decimals=4), 5),
                        round(np.around(roc_auc_score(y_valid, logistic_predict_normalize),
                                        decimals=4), 5),
                        round(np.around(roc_auc_score(y_valid, logistic_predict_scale),
                                        decimals=4), 5),
                        round(np.around(roc_auc_score(y_valid, logistic_predict_normalize_scale),
                                        decimals=4), 5),
                        round(np.around(roc_auc_score(y_valid, logistic_predict_scale_normalize),
                                        decimals=4), 5)]

    window.logistic_max_values_indexes = [window.logistic_accuracy.index(max(window.logistic_accuracy)),
                                   window.logistic_precision.index(max(window.logistic_precision)),
                                   window.logistic_recall.index(max(window.logistic_recall)),
                                   window.logistic_f1.index(max(window.logistic_f1)),
                                   window.logistic_roc_auc.index(max(window.logistic_roc_auc))]

    #print(window.logistic_roc_auc[max(set(window.logistic_max_values_indexes), key=window.logistic_max_values_indexes.count)])


def bayes(window, x_train, x_valid, y_train, y_valid, x_train_normalize, x_train_scale,
          x_train_normalize_scale, x_train_scale_normalize, x_valid_normalize, x_valid_scale,
          x_valid_normalize_scale, x_valid_scale_normalize):
    clf = MultinomialNB()
    try:
        clf.fit(x_train, y_train)
        clf_predict = clf.predict(x_valid)
        clf.fit(x_train_normalize, y_train)
        clf_predict_normalize = clf.predict(x_valid_normalize)
        clf.fit(x_train_scale, y_train)
        clf_predict_scale = clf.predict(x_valid_scale)
        clf.fit(x_train_normalize_scale, y_train)
        clf_predict_normalize_scale = clf.predict(x_valid_normalize_scale)
        clf.fit(x_train_scale_normalize, y_train)
        clf_predict_scale_normalize = clf.predict(x_valid_scale_normalize)
        window.clf_accuracy = [round(np.around(accuracy_score(y_valid, clf_predict),
                                        decimals=4), 5),
                        round(np.around(accuracy_score(y_valid, clf_predict_normalize),
                                        decimals=4), 5),
                        round(np.around(accuracy_score(y_valid, clf_predict_scale),
                                        decimals=4), 5),
                        round(np.around(accuracy_score(y_valid, clf_predict_normalize_scale),
                                        decimals=4), 5),
                        round(np.around(accuracy_score(y_valid, clf_predict_scale_normalize),
                                        decimals=4), 5)]

        window.clf_precision = [round(np.around(precision_score(y_valid, clf_predict),
                                         decimals=4), 5),
                         round(np.around(precision_score(y_valid, clf_predict_normalize),
                                         decimals=4), 5),
                         round(np.around(precision_score(y_valid, clf_predict_scale),
                                         decimals=4), 5),
                         round(np.around(precision_score(y_valid, clf_predict_normalize_scale),
                                         decimals=4), 5),
                         round(np.around(precision_score(y_valid, clf_predict_scale_normalize),
                                         decimals=4), 5)]

        window.clf_recall = [round(np.around(recall_score(y_valid, clf_predict),
                                      decimals=4), 5),
                      round(np.around(recall_score(y_valid, clf_predict_normalize),
                                      decimals=4), 5),
                      round(np.around(recall_score(y_valid, clf_predict_scale),
                                      decimals=4), 5),
                      round(np.around(recall_score(y_valid, clf_predict_normalize_scale),
                                      decimals=4), 5),
                      round(np.around(recall_score(y_valid, clf_predict_scale_normalize),
                                      decimals=4), 5)]

        window.clf_f1 = [round(np.around(f1_score(y_valid, clf_predict),
                                  decimals=4), 5), round(np.around(f1_score(y_valid, clf_predict_normalize),
                                                                   decimals=4), 5),
                  round(np.around(f1_score(y_valid, clf_predict_scale),
                                  decimals=4), 5), round(np.around(f1_score(y_valid, clf_predict_normalize_scale),
                                                                   decimals=4), 5),
                  round(np.around(f1_score(y_valid, clf_predict_scale_normalize),
                                  decimals=4), 5)]

        window.clf_roc_auc = [round(np.around(roc_auc_score(y_valid, clf_predict),
                                       decimals=4), 5),
                       round(np.around(roc_auc_score(y_valid, clf_predict_normalize),
                                       decimals=4), 5),
                       round(np.around(roc_auc_score(y_valid, clf_predict_scale),
                                       decimals=4), 5),
                       round(np.around(roc_auc_score(y_valid, clf_predict_normalize_scale),
                                       decimals=4), 5),
                       round(np.around(roc_auc_score(y_valid, clf_predict_scale_normalize),
                                       decimals=4), 5)]
    except ValueError:
        clf.fit(x_train, y_train)
        clf_predict = clf.predict(x_valid)
        clf.fit(x_train_normalize, y_train)
        clf_predict_normalize = clf.predict(x_valid_normalize)
        window.clf_accuracy = [round(np.around(accuracy_score(y_valid, clf_predict),
                                        decimals=4), 5),
                        round(np.around(accuracy_score(y_valid, clf_predict_normalize),
                                        decimals=4), 5)]

        window.clf_precision = [round(np.around(precision_score(y_valid, clf_predict, zero_division=0),
                                         decimals=4), 5),
                         round(np.around(precision_score(y_valid, clf_predict_normalize, zero_division=0),
                                         decimals=4), 5)]

        window.clf_recall = [round(np.around(recall_score(y_valid, clf_predict),
                                      decimals=4), 5),
                      round(np.around(recall_score(y_valid, clf_predict_normalize),
                                      decimals=4), 5)]

        window.clf_f1 = [round(np.around(f1_score(y_valid, clf_predict),
                                  decimals=4), 5), round(np.around(f1_score(y_valid, clf_predict_normalize),
                                                                   decimals=4), 5)]

        window.clf_roc_auc = [round(np.around(roc_auc_score(y_valid, clf_predict),
                                       decimals=4), 5),
                       round(np.around(roc_auc_score(y_valid, clf_predict_normalize),
                                       decimals=4), 5)]
    finally:
        window.clf_max_values_indexes = [window.clf_accuracy.index(max(window.clf_accuracy)),
                                  window.clf_precision.index(max(window.clf_precision)),
                                  window.clf_recall.index(max(window.clf_recall)),
                                  window.clf_f1.index(max(window.clf_f1)),
                                  window.clf_roc_auc.index(max(window.clf_roc_auc))]

def discriminant_analysis(window, x_train, x_valid, y_train, y_valid, x_train_normalize, x_train_scale,
                          x_train_normalize_scale, x_train_scale_normalize, x_valid_normalize, x_valid_scale,
                          x_valid_normalize_scale, x_valid_scale_normalize):
    disc = LinearDiscriminantAnalysis()
    disc.fit(x_train, y_train)
    disc_predict = disc.predict(x_valid)
    disc.fit(x_train_normalize, y_train)
    disc_predict_normalize = disc.predict(x_valid_normalize)
    disc.fit(x_train_scale, y_train)
    disc_predict_scale = disc.predict(x_valid_scale)
    disc.fit(x_train_normalize_scale, y_train)
    disc_predict_normalize_scale = disc.predict(x_valid_normalize_scale)
    disc.fit(x_train_scale_normalize, y_train)
    disc_predict_scale_normalize = disc.predict(x_valid_scale_normalize)
    window.disc_accuracy = [round(np.around(accuracy_score(y_valid, disc_predict),
                                     decimals=4), 5),
                     round(np.around(accuracy_score(y_valid, disc_predict_normalize),
                                     decimals=4), 5),
                     round(np.around(accuracy_score(y_valid, disc_predict_scale),
                                     decimals=4), 5),
                     round(np.around(accuracy_score(y_valid, disc_predict_normalize_scale),
                                     decimals=4), 5),
                     round(np.around(accuracy_score(y_valid, disc_predict_scale_normalize),
                                     decimals=4), 5)]

    window.disc_precision = [round(np.around(precision_score(y_valid, disc_predict),
                                      decimals=4), 5),
                      round(np.around(precision_score(y_valid, disc_predict_normalize),
                                      decimals=4), 5),
                      round(np.around(precision_score(y_valid, disc_predict_scale),
                                      decimals=4), 5),
                      round(np.around(precision_score(y_valid, disc_predict_normalize_scale),
                                      decimals=4), 5),
                      round(np.around(precision_score(y_valid, disc_predict_scale_normalize),
                                      decimals=4), 5)]

    window.disc_recall = [round(np.around(recall_score(y_valid, disc_predict),
                                   decimals=4), 5),
                   round(np.around(recall_score(y_valid, disc_predict_normalize),
                                   decimals=4), 5),
                   round(np.around(recall_score(y_valid, disc_predict_scale),
                                   decimals=4), 5),
                   round(np.around(recall_score(y_valid, disc_predict_normalize_scale),
                                   decimals=4), 5),
                   round(np.around(recall_score(y_valid, disc_predict_scale_normalize),
                                   decimals=4), 5)]

    window.disc_f1 = [round(np.around(f1_score(y_valid, disc_predict),
                               decimals=4), 5), round(np.around(f1_score(y_valid, disc_predict_normalize),
                                                                decimals=4), 5),
               round(np.around(f1_score(y_valid, disc_predict_scale),
                               decimals=4), 5), round(np.around(f1_score(y_valid, disc_predict_normalize_scale),
                                                                decimals=4), 5),
               round(np.around(f1_score(y_valid, disc_predict_scale_normalize),
                               decimals=4), 5)]

    window.disc_roc_auc = [round(np.around(roc_auc_score(y_valid, disc_predict),
                                    decimals=4), 5),
                    round(np.around(roc_auc_score(y_valid, disc_predict_normalize),
                                    decimals=4), 5),
                    round(np.around(roc_auc_score(y_valid, disc_predict_scale),
                                    decimals=4), 5),
                    round(np.around(roc_auc_score(y_valid, disc_predict_normalize_scale),
                                    decimals=4), 5),
                    round(np.around(roc_auc_score(y_valid, disc_predict_scale_normalize),
                                    decimals=4), 5)]

    window.disc_max_values_indexes = [window.disc_accuracy.index(max(window.disc_accuracy)),
                               window.disc_precision.index(max(window.disc_precision)),
                               window.disc_recall.index(max(window.disc_recall)),
                               window.disc_f1.index(max(window.disc_f1)),
                               window.disc_roc_auc.index(max(window.disc_roc_auc))]

    #print(window.disc_roc_auc[max(set(disc_max_values_indexes), key=disc_max_values_indexes.count)])


def svm_vectors(window, x_train, x_valid, y_train, y_valid, x_train_normalize, x_train_scale,
                x_train_normalize_scale, x_train_scale_normalize, x_valid_normalize, x_valid_scale,
                x_valid_normalize_scale, x_valid_scale_normalize):
    support = SVC(kernel="linear", C=0.025)
    support.fit(x_train, y_train)
    support_predict = support.predict(x_valid)
    support.fit(x_train_normalize, y_train)
    support_predict_normalize = support.predict(x_valid_normalize)
    support.fit(x_train_scale, y_train)
    support_predict_scale = support.predict(x_valid_scale)
    support.fit(x_train_normalize_scale, y_train)
    support_predict_normalize_scale = support.predict(x_valid_normalize_scale)
    support.fit(x_train_scale_normalize, y_train)
    support_predict_scale_normalize = support.predict(x_valid_scale_normalize)
    window.support_accuracy = [round(np.around(accuracy_score(y_valid, support_predict),
                                        decimals=4), 5),
                        round(np.around(accuracy_score(y_valid, support_predict_normalize),
                                        decimals=4), 5),
                        round(np.around(accuracy_score(y_valid, support_predict_scale),
                                        decimals=4), 5),
                        round(np.around(accuracy_score(y_valid, support_predict_normalize_scale),
                                        decimals=4), 5),
                        round(np.around(accuracy_score(y_valid, support_predict_scale_normalize),
                                        decimals=4), 5)]

    window.support_precision = [round(np.around(precision_score(y_valid, support_predict),
                                         decimals=4), 5),
                         round(np.around(precision_score(y_valid, support_predict_normalize),
                                         decimals=4), 5),
                         round(np.around(precision_score(y_valid, support_predict_scale),
                                         decimals=4), 5),
                         round(np.around(precision_score(y_valid, support_predict_normalize_scale),
                                         decimals=4), 5),
                         round(np.around(precision_score(y_valid, support_predict_scale_normalize),
                                         decimals=4), 5)]

    window.support_recall = [round(np.around(recall_score(y_valid, support_predict),
                                      decimals=4), 5),
                      round(np.around(recall_score(y_valid, support_predict_normalize),
                                      decimals=4), 5),
                      round(np.around(recall_score(y_valid, support_predict_scale),
                                      decimals=4), 5),
                      round(np.around(recall_score(y_valid, support_predict_normalize_scale),
                                      decimals=4), 5),
                      round(np.around(recall_score(y_valid, support_predict_scale_normalize),
                                      decimals=4), 5)]

    window.support_f1 = [round(np.around(f1_score(y_valid, support_predict),
                                  decimals=4), 5), round(np.around(f1_score(y_valid, support_predict_normalize),
                                                                   decimals=4), 5),
                  round(np.around(f1_score(y_valid, support_predict_scale),
                                  decimals=4), 5), round(np.around(f1_score(y_valid, support_predict_normalize_scale),
                                                                   decimals=4), 5),
                  round(np.around(f1_score(y_valid, support_predict_scale_normalize),
                                  decimals=4), 5)]

    window.support_roc_auc = [round(np.around(roc_auc_score(y_valid, support_predict),
                                       decimals=4), 5),
                       round(np.around(roc_auc_score(y_valid, support_predict_normalize),
                                       decimals=4), 5),
                       round(np.around(roc_auc_score(y_valid, support_predict_scale),
                                       decimals=4), 5),
                       round(np.around(roc_auc_score(y_valid, support_predict_normalize_scale),
                                       decimals=4), 5),
                       round(np.around(roc_auc_score(y_valid, support_predict_scale_normalize),
                                       decimals=4), 5)]

    window.support_max_values_indexes = [window.support_accuracy.index(max(window.support_accuracy)),
                                  window.support_precision.index(max(window.support_precision)),
                                  window.support_recall.index(max(window.support_recall)),
                                  window.support_f1.index(max(window.support_f1)),
                                  window.support_roc_auc.index(max(window.support_roc_auc))]

    #print(window.support_roc_auc[max(set(support_max_values_indexes), key=support_max_values_indexes.count)])


def tree(window, x_train, x_valid, y_train, y_valid, x_train_normalize, x_train_scale,
         x_train_normalize_scale, x_train_scale_normalize, x_valid_normalize, x_valid_scale,
         x_valid_normalize_scale, x_valid_scale_normalize):
    tree = DecisionTreeClassifier()
    tree.fit(x_train, y_train)
    tree_predict = tree.predict(x_valid)
    tree.fit(x_train_normalize, y_train)
    tree_predict_normalize = tree.predict(x_valid_normalize)
    tree.fit(x_train_scale, y_train)
    tree_predict_scale = tree.predict(x_valid_scale)
    tree.fit(x_train_normalize_scale, y_train)
    tree_predict_normalize_scale = tree.predict(x_valid_normalize_scale)
    tree.fit(x_train_scale_normalize, y_train)
    tree_predict_scale_normalize = tree.predict(x_valid_scale_normalize)
    window.tree_accuracy = [round(np.around(accuracy_score(y_valid, tree_predict),
                                     decimals=4), 5),
                     round(np.around(accuracy_score(y_valid, tree_predict_normalize),
                                     decimals=4), 5),
                     round(np.around(accuracy_score(y_valid, tree_predict_scale),
                                     decimals=4), 5),
                     round(np.around(accuracy_score(y_valid, tree_predict_normalize_scale),
                                     decimals=4), 5),
                     round(np.around(accuracy_score(y_valid, tree_predict_scale_normalize),
                                     decimals=4), 5)]

    window.tree_precision = [round(np.around(precision_score(y_valid, tree_predict),
                                      decimals=4), 5),
                      round(np.around(precision_score(y_valid, tree_predict_normalize),
                                      decimals=4), 5),
                      round(np.around(precision_score(y_valid, tree_predict_scale),
                                      decimals=4), 5),
                      round(np.around(precision_score(y_valid, tree_predict_normalize_scale),
                                      decimals=4), 5),
                      round(np.around(precision_score(y_valid, tree_predict_scale_normalize),
                                      decimals=4), 5)]

    window.tree_recall = [round(np.around(recall_score(y_valid, tree_predict),
                                   decimals=4), 5),
                   round(np.around(recall_score(y_valid, tree_predict_normalize),
                                   decimals=4), 5),
                   round(np.around(recall_score(y_valid, tree_predict_scale),
                                   decimals=4), 5),
                   round(np.around(recall_score(y_valid, tree_predict_normalize_scale),
                                   decimals=4), 5),
                   round(np.around(recall_score(y_valid, tree_predict_scale_normalize),
                                   decimals=4), 5)]

    window.tree_f1 = [round(np.around(f1_score(y_valid, tree_predict),
                               decimals=4), 5), round(np.around(f1_score(y_valid, tree_predict_normalize),
                                                                decimals=4), 5),
               round(np.around(f1_score(y_valid, tree_predict_scale),
                               decimals=4), 5), round(np.around(f1_score(y_valid, tree_predict_normalize_scale),
                                                                decimals=4), 5),
               round(np.around(f1_score(y_valid, tree_predict_scale_normalize),
                               decimals=4), 5)]

    window.tree_roc_auc = [round(np.around(roc_auc_score(y_valid, tree_predict),
                                    decimals=4), 5),
                    round(np.around(roc_auc_score(y_valid, tree_predict_normalize),
                                    decimals=4), 5),
                    round(np.around(roc_auc_score(y_valid, tree_predict_scale),
                                    decimals=4), 5),
                    round(np.around(roc_auc_score(y_valid, tree_predict_normalize_scale),
                                    decimals=4), 5),
                    round(np.around(roc_auc_score(y_valid, tree_predict_scale_normalize),
                                    decimals=4), 5)]

    window.tree_max_values_indexes = [window.tree_accuracy.index(max(window.tree_accuracy)),
                               window.tree_precision.index(max(window.tree_precision)),
                               window.tree_recall.index(max(window.tree_recall)),
                               window.tree_f1.index(max(window.tree_f1)),
                               window.tree_roc_auc.index(max(window.tree_roc_auc))]

    #print(window.tree_roc_auc[max(set(tree_max_values_indexes), key=tree_max_values_indexes.count)])


def neural_network(window, x_train, x_valid, y_train, y_valid, x_train_normalize, x_train_scale,
                   x_train_normalize_scale, x_train_scale_normalize, x_valid_normalize, x_valid_scale,
                   x_valid_normalize_scale, x_valid_scale_normalize):
    neural = MLPClassifier()
    neural.fit(x_train, y_train)
    neural_predict = neural.predict(x_valid)
    neural.fit(x_train_normalize, y_train)
    neural_predict_normalize = neural.predict(x_valid_normalize)
    neural.fit(x_train_scale, y_train)
    neural_predict_scale = neural.predict(x_valid_scale)
    neural.fit(x_train_normalize_scale, y_train)
    neural_predict_normalize_scale = neural.predict(x_valid_normalize_scale)
    neural.fit(x_train_scale_normalize, y_train)
    neural_predict_scale_normalize = neural.predict(x_valid_scale_normalize)
    window.neural_accuracy = [round(np.around(accuracy_score(y_valid, neural_predict),
                                       decimals=4), 5),
                       round(np.around(accuracy_score(y_valid, neural_predict_normalize),
                                       decimals=4), 5),
                       round(np.around(accuracy_score(y_valid, neural_predict_scale),
                                       decimals=4), 5),
                       round(np.around(accuracy_score(y_valid, neural_predict_normalize_scale),
                                       decimals=4), 5),
                       round(np.around(accuracy_score(y_valid, neural_predict_scale_normalize),
                                       decimals=4), 5)]

    window.neural_precision = [round(np.around(precision_score(y_valid, neural_predict),
                                        decimals=4), 5),
                        round(np.around(precision_score(y_valid, neural_predict_normalize),
                                        decimals=4), 5),
                        round(np.around(precision_score(y_valid, neural_predict_scale),
                                        decimals=4), 5),
                        round(np.around(precision_score(y_valid, neural_predict_normalize_scale),
                                        decimals=4), 5),
                        round(np.around(precision_score(y_valid, neural_predict_scale_normalize),
                                        decimals=4), 5)]

    window.neural_recall = [round(np.around(recall_score(y_valid, neural_predict),
                                     decimals=4), 5),
                     round(np.around(recall_score(y_valid, neural_predict_normalize),
                                     decimals=4), 5),
                     round(np.around(recall_score(y_valid, neural_predict_scale),
                                     decimals=4), 5),
                     round(np.around(recall_score(y_valid, neural_predict_normalize_scale),
                                     decimals=4), 5),
                     round(np.around(recall_score(y_valid, neural_predict_scale_normalize),
                                     decimals=4), 5)]

    window.neural_f1 = [round(np.around(f1_score(y_valid, neural_predict),
                                 decimals=4), 5), round(np.around(f1_score(y_valid, neural_predict_normalize),
                                                                  decimals=4), 5),
                 round(np.around(f1_score(y_valid, neural_predict_scale),
                                 decimals=4), 5), round(np.around(f1_score(y_valid, neural_predict_normalize_scale),
                                                                  decimals=4), 5),
                 round(np.around(f1_score(y_valid, neural_predict_scale_normalize),
                                 decimals=4), 5)]

    window.neural_roc_auc = [round(np.around(roc_auc_score(y_valid, neural_predict),
                                      decimals=4), 5),
                      round(np.around(roc_auc_score(y_valid, neural_predict_normalize),
                                      decimals=4), 5),
                      round(np.around(roc_auc_score(y_valid, neural_predict_scale),
                                      decimals=4), 5),
                      round(np.around(roc_auc_score(y_valid, neural_predict_normalize_scale),
                                      decimals=4), 5),
                      round(np.around(roc_auc_score(y_valid, neural_predict_scale_normalize),
                                      decimals=4), 5)]

    window.neural_max_values_indexes = [window.neural_accuracy.index(max(window.neural_accuracy)),
                                 window.neural_precision.index(max(window.neural_precision)),
                                 window.neural_recall.index(max(window.neural_recall)),
                                 window.neural_f1.index(max(window.neural_f1)),
                                 window.neural_roc_auc.index(max(window.neural_roc_auc))]

    #print(window.neural_roc_auc[max(set(neural_max_values_indexes), key=neural_max_values_indexes.count)])
