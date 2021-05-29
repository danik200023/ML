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
    logistic_accuracy = [round(np.around(accuracy_score(y_valid, logistic_predict),
                                         decimals=4), 5),
                         round(np.around(accuracy_score(y_valid, logistic_predict_normalize),
                                         decimals=4), 5),
                         round(np.around(accuracy_score(y_valid, logistic_predict_scale),
                                         decimals=4), 5),
                         round(np.around(accuracy_score(y_valid, logistic_predict_normalize_scale),
                                         decimals=4), 5),
                         round(np.around(accuracy_score(y_valid, logistic_predict_scale_normalize),
                                         decimals=4), 5)]

    logistic_precision = [round(np.around(precision_score(y_valid, logistic_predict),
                                          decimals=4), 5),
                          round(np.around(precision_score(y_valid, logistic_predict_normalize),
                                          decimals=4), 5),
                          round(np.around(precision_score(y_valid, logistic_predict_scale),
                                          decimals=4), 5),
                          round(np.around(precision_score(y_valid, logistic_predict_normalize_scale),
                                          decimals=4), 5),
                          round(np.around(precision_score(y_valid, logistic_predict_scale_normalize),
                                          decimals=4), 5)]

    logistic_recall = [round(np.around(recall_score(y_valid, logistic_predict),
                                       decimals=4), 5),
                       round(np.around(recall_score(y_valid, logistic_predict_normalize),
                                       decimals=4), 5),
                       round(np.around(recall_score(y_valid, logistic_predict_scale),
                                       decimals=4), 5),
                       round(np.around(recall_score(y_valid, logistic_predict_normalize_scale),
                                       decimals=4), 5),
                       round(np.around(recall_score(y_valid, logistic_predict_scale_normalize),
                                       decimals=4), 5)]

    logistic_f1 = [round(np.around(f1_score(y_valid, logistic_predict),
                                   decimals=4), 5), round(np.around(f1_score(y_valid, logistic_predict_normalize),
                                                                    decimals=4), 5),
                   round(np.around(f1_score(y_valid, logistic_predict_scale),
                                   decimals=4), 5), round(np.around(f1_score(y_valid, logistic_predict_normalize_scale),
                                                                    decimals=4), 5),
                   round(np.around(f1_score(y_valid, logistic_predict_scale_normalize),
                                   decimals=4), 5)]

    logistic_roc_auc = [round(np.around(roc_auc_score(y_valid, logistic_predict),
                                        decimals=4), 5),
                        round(np.around(roc_auc_score(y_valid, logistic_predict_normalize),
                                        decimals=4), 5),
                        round(np.around(roc_auc_score(y_valid, logistic_predict_scale),
                                        decimals=4), 5),
                        round(np.around(roc_auc_score(y_valid, logistic_predict_normalize_scale),
                                        decimals=4), 5),
                        round(np.around(roc_auc_score(y_valid, logistic_predict_scale_normalize),
                                        decimals=4), 5)]

    logistic_max_values_indexes = [logistic_accuracy.index(max(logistic_accuracy)),
                                   logistic_precision.index(max(logistic_precision)),
                                   logistic_recall.index(max(logistic_recall)),
                                   logistic_f1.index(max(logistic_f1)),
                                   logistic_roc_auc.index(max(logistic_roc_auc))]

    print(logistic_roc_auc[max(set(logistic_max_values_indexes), key=logistic_max_values_indexes.count)])


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
        clf_accuracy = [round(np.around(accuracy_score(y_valid, clf_predict),
                                        decimals=4), 5),
                        round(np.around(accuracy_score(y_valid, clf_predict_normalize),
                                        decimals=4), 5),
                        round(np.around(accuracy_score(y_valid, clf_predict_scale),
                                        decimals=4), 5),
                        round(np.around(accuracy_score(y_valid, clf_predict_normalize_scale),
                                        decimals=4), 5),
                        round(np.around(accuracy_score(y_valid, clf_predict_scale_normalize),
                                        decimals=4), 5)]

        clf_precision = [round(np.around(precision_score(y_valid, clf_predict),
                                         decimals=4), 5),
                         round(np.around(precision_score(y_valid, clf_predict_normalize),
                                         decimals=4), 5),
                         round(np.around(precision_score(y_valid, clf_predict_scale),
                                         decimals=4), 5),
                         round(np.around(precision_score(y_valid, clf_predict_normalize_scale),
                                         decimals=4), 5),
                         round(np.around(precision_score(y_valid, clf_predict_scale_normalize),
                                         decimals=4), 5)]

        clf_recall = [round(np.around(recall_score(y_valid, clf_predict),
                                      decimals=4), 5),
                      round(np.around(recall_score(y_valid, clf_predict_normalize),
                                      decimals=4), 5),
                      round(np.around(recall_score(y_valid, clf_predict_scale),
                                      decimals=4), 5),
                      round(np.around(recall_score(y_valid, clf_predict_normalize_scale),
                                      decimals=4), 5),
                      round(np.around(recall_score(y_valid, clf_predict_scale_normalize),
                                      decimals=4), 5)]

        clf_f1 = [round(np.around(f1_score(y_valid, clf_predict),
                                  decimals=4), 5), round(np.around(f1_score(y_valid, clf_predict_normalize),
                                                                   decimals=4), 5),
                  round(np.around(f1_score(y_valid, clf_predict_scale),
                                  decimals=4), 5), round(np.around(f1_score(y_valid, clf_predict_normalize_scale),
                                                                   decimals=4), 5),
                  round(np.around(f1_score(y_valid, clf_predict_scale_normalize),
                                  decimals=4), 5)]

        clf_roc_auc = [round(np.around(roc_auc_score(y_valid, clf_predict),
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
        clf_accuracy = [round(np.around(accuracy_score(y_valid, clf_predict),
                                        decimals=4), 5),
                        round(np.around(accuracy_score(y_valid, clf_predict_normalize),
                                        decimals=4), 5)]

        clf_precision = [round(np.around(precision_score(y_valid, clf_predict, zero_division=0),
                                         decimals=4), 5),
                         round(np.around(precision_score(y_valid, clf_predict_normalize, zero_division=0),
                                         decimals=4), 5)]

        clf_recall = [round(np.around(recall_score(y_valid, clf_predict),
                                      decimals=4), 5),
                      round(np.around(recall_score(y_valid, clf_predict_normalize),
                                      decimals=4), 5)]

        clf_f1 = [round(np.around(f1_score(y_valid, clf_predict),
                                  decimals=4), 5), round(np.around(f1_score(y_valid, clf_predict_normalize),
                                                                   decimals=4), 5)]

        clf_roc_auc = [round(np.around(roc_auc_score(y_valid, clf_predict),
                                       decimals=4), 5),
                       round(np.around(roc_auc_score(y_valid, clf_predict_normalize),
                                       decimals=4), 5)]
    finally:
        clf_max_values_indexes = [clf_accuracy.index(max(clf_accuracy)),
                                  clf_precision.index(max(clf_precision)),
                                  clf_recall.index(max(clf_recall)),
                                  clf_f1.index(max(clf_f1)),
                                  clf_roc_auc.index(max(clf_roc_auc))]
        print(clf_accuracy[max(set(clf_max_values_indexes), key=clf_max_values_indexes.count)])
        print(clf_precision[max(set(clf_max_values_indexes), key=clf_max_values_indexes.count)])
        print(clf_recall[max(set(clf_max_values_indexes), key=clf_max_values_indexes.count)])
        print(clf_f1[max(set(clf_max_values_indexes), key=clf_max_values_indexes.count)])
        print(clf_roc_auc[max(set(clf_max_values_indexes), key=clf_max_values_indexes.count)])
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
    disc_accuracy = [round(np.around(accuracy_score(y_valid, disc_predict),
                                     decimals=4), 5),
                     round(np.around(accuracy_score(y_valid, disc_predict_normalize),
                                     decimals=4), 5),
                     round(np.around(accuracy_score(y_valid, disc_predict_scale),
                                     decimals=4), 5),
                     round(np.around(accuracy_score(y_valid, disc_predict_normalize_scale),
                                     decimals=4), 5),
                     round(np.around(accuracy_score(y_valid, disc_predict_scale_normalize),
                                     decimals=4), 5)]

    disc_precision = [round(np.around(precision_score(y_valid, disc_predict),
                                      decimals=4), 5),
                      round(np.around(precision_score(y_valid, disc_predict_normalize),
                                      decimals=4), 5),
                      round(np.around(precision_score(y_valid, disc_predict_scale),
                                      decimals=4), 5),
                      round(np.around(precision_score(y_valid, disc_predict_normalize_scale),
                                      decimals=4), 5),
                      round(np.around(precision_score(y_valid, disc_predict_scale_normalize),
                                      decimals=4), 5)]

    disc_recall = [round(np.around(recall_score(y_valid, disc_predict),
                                   decimals=4), 5),
                   round(np.around(recall_score(y_valid, disc_predict_normalize),
                                   decimals=4), 5),
                   round(np.around(recall_score(y_valid, disc_predict_scale),
                                   decimals=4), 5),
                   round(np.around(recall_score(y_valid, disc_predict_normalize_scale),
                                   decimals=4), 5),
                   round(np.around(recall_score(y_valid, disc_predict_scale_normalize),
                                   decimals=4), 5)]

    disc_f1 = [round(np.around(f1_score(y_valid, disc_predict),
                               decimals=4), 5), round(np.around(f1_score(y_valid, disc_predict_normalize),
                                                                decimals=4), 5),
               round(np.around(f1_score(y_valid, disc_predict_scale),
                               decimals=4), 5), round(np.around(f1_score(y_valid, disc_predict_normalize_scale),
                                                                decimals=4), 5),
               round(np.around(f1_score(y_valid, disc_predict_scale_normalize),
                               decimals=4), 5)]

    disc_roc_auc = [round(np.around(roc_auc_score(y_valid, disc_predict),
                                    decimals=4), 5),
                    round(np.around(roc_auc_score(y_valid, disc_predict_normalize),
                                    decimals=4), 5),
                    round(np.around(roc_auc_score(y_valid, disc_predict_scale),
                                    decimals=4), 5),
                    round(np.around(roc_auc_score(y_valid, disc_predict_normalize_scale),
                                    decimals=4), 5),
                    round(np.around(roc_auc_score(y_valid, disc_predict_scale_normalize),
                                    decimals=4), 5)]

    disc_max_values_indexes = [disc_accuracy.index(max(disc_accuracy)),
                               disc_precision.index(max(disc_precision)),
                               disc_recall.index(max(disc_recall)),
                               disc_f1.index(max(disc_f1)),
                               disc_roc_auc.index(max(disc_roc_auc))]

    print(disc_roc_auc[max(set(disc_max_values_indexes), key=disc_max_values_indexes.count)])


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
    support_accuracy = [round(np.around(accuracy_score(y_valid, support_predict),
                                        decimals=4), 5),
                        round(np.around(accuracy_score(y_valid, support_predict_normalize),
                                        decimals=4), 5),
                        round(np.around(accuracy_score(y_valid, support_predict_scale),
                                        decimals=4), 5),
                        round(np.around(accuracy_score(y_valid, support_predict_normalize_scale),
                                        decimals=4), 5),
                        round(np.around(accuracy_score(y_valid, support_predict_scale_normalize),
                                        decimals=4), 5)]

    support_precision = [round(np.around(precision_score(y_valid, support_predict),
                                         decimals=4), 5),
                         round(np.around(precision_score(y_valid, support_predict_normalize),
                                         decimals=4), 5),
                         round(np.around(precision_score(y_valid, support_predict_scale),
                                         decimals=4), 5),
                         round(np.around(precision_score(y_valid, support_predict_normalize_scale),
                                         decimals=4), 5),
                         round(np.around(precision_score(y_valid, support_predict_scale_normalize),
                                         decimals=4), 5)]

    support_recall = [round(np.around(recall_score(y_valid, support_predict),
                                      decimals=4), 5),
                      round(np.around(recall_score(y_valid, support_predict_normalize),
                                      decimals=4), 5),
                      round(np.around(recall_score(y_valid, support_predict_scale),
                                      decimals=4), 5),
                      round(np.around(recall_score(y_valid, support_predict_normalize_scale),
                                      decimals=4), 5),
                      round(np.around(recall_score(y_valid, support_predict_scale_normalize),
                                      decimals=4), 5)]

    support_f1 = [round(np.around(f1_score(y_valid, support_predict),
                                  decimals=4), 5), round(np.around(f1_score(y_valid, support_predict_normalize),
                                                                   decimals=4), 5),
                  round(np.around(f1_score(y_valid, support_predict_scale),
                                  decimals=4), 5), round(np.around(f1_score(y_valid, support_predict_normalize_scale),
                                                                   decimals=4), 5),
                  round(np.around(f1_score(y_valid, support_predict_scale_normalize),
                                  decimals=4), 5)]

    support_roc_auc = [round(np.around(roc_auc_score(y_valid, support_predict),
                                       decimals=4), 5),
                       round(np.around(roc_auc_score(y_valid, support_predict_normalize),
                                       decimals=4), 5),
                       round(np.around(roc_auc_score(y_valid, support_predict_scale),
                                       decimals=4), 5),
                       round(np.around(roc_auc_score(y_valid, support_predict_normalize_scale),
                                       decimals=4), 5),
                       round(np.around(roc_auc_score(y_valid, support_predict_scale_normalize),
                                       decimals=4), 5)]

    support_max_values_indexes = [support_accuracy.index(max(support_accuracy)),
                                  support_precision.index(max(support_precision)),
                                  support_recall.index(max(support_recall)),
                                  support_f1.index(max(support_f1)),
                                  support_roc_auc.index(max(support_roc_auc))]

    print(support_roc_auc[max(set(support_max_values_indexes), key=support_max_values_indexes.count)])


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
    tree_accuracy = [round(np.around(accuracy_score(y_valid, tree_predict),
                                     decimals=4), 5),
                     round(np.around(accuracy_score(y_valid, tree_predict_normalize),
                                     decimals=4), 5),
                     round(np.around(accuracy_score(y_valid, tree_predict_scale),
                                     decimals=4), 5),
                     round(np.around(accuracy_score(y_valid, tree_predict_normalize_scale),
                                     decimals=4), 5),
                     round(np.around(accuracy_score(y_valid, tree_predict_scale_normalize),
                                     decimals=4), 5)]

    tree_precision = [round(np.around(precision_score(y_valid, tree_predict),
                                      decimals=4), 5),
                      round(np.around(precision_score(y_valid, tree_predict_normalize),
                                      decimals=4), 5),
                      round(np.around(precision_score(y_valid, tree_predict_scale),
                                      decimals=4), 5),
                      round(np.around(precision_score(y_valid, tree_predict_normalize_scale),
                                      decimals=4), 5),
                      round(np.around(precision_score(y_valid, tree_predict_scale_normalize),
                                      decimals=4), 5)]

    tree_recall = [round(np.around(recall_score(y_valid, tree_predict),
                                   decimals=4), 5),
                   round(np.around(recall_score(y_valid, tree_predict_normalize),
                                   decimals=4), 5),
                   round(np.around(recall_score(y_valid, tree_predict_scale),
                                   decimals=4), 5),
                   round(np.around(recall_score(y_valid, tree_predict_normalize_scale),
                                   decimals=4), 5),
                   round(np.around(recall_score(y_valid, tree_predict_scale_normalize),
                                   decimals=4), 5)]

    tree_f1 = [round(np.around(f1_score(y_valid, tree_predict),
                               decimals=4), 5), round(np.around(f1_score(y_valid, tree_predict_normalize),
                                                                decimals=4), 5),
               round(np.around(f1_score(y_valid, tree_predict_scale),
                               decimals=4), 5), round(np.around(f1_score(y_valid, tree_predict_normalize_scale),
                                                                decimals=4), 5),
               round(np.around(f1_score(y_valid, tree_predict_scale_normalize),
                               decimals=4), 5)]

    tree_roc_auc = [round(np.around(roc_auc_score(y_valid, tree_predict),
                                    decimals=4), 5),
                    round(np.around(roc_auc_score(y_valid, tree_predict_normalize),
                                    decimals=4), 5),
                    round(np.around(roc_auc_score(y_valid, tree_predict_scale),
                                    decimals=4), 5),
                    round(np.around(roc_auc_score(y_valid, tree_predict_normalize_scale),
                                    decimals=4), 5),
                    round(np.around(roc_auc_score(y_valid, tree_predict_scale_normalize),
                                    decimals=4), 5)]

    tree_max_values_indexes = [tree_accuracy.index(max(tree_accuracy)),
                               tree_precision.index(max(tree_precision)),
                               tree_recall.index(max(tree_recall)),
                               tree_f1.index(max(tree_f1)),
                               tree_roc_auc.index(max(tree_roc_auc))]

    print(tree_roc_auc[max(set(tree_max_values_indexes), key=tree_max_values_indexes.count)])


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
    neural_accuracy = [round(np.around(accuracy_score(y_valid, neural_predict),
                                       decimals=4), 5),
                       round(np.around(accuracy_score(y_valid, neural_predict_normalize),
                                       decimals=4), 5),
                       round(np.around(accuracy_score(y_valid, neural_predict_scale),
                                       decimals=4), 5),
                       round(np.around(accuracy_score(y_valid, neural_predict_normalize_scale),
                                       decimals=4), 5),
                       round(np.around(accuracy_score(y_valid, neural_predict_scale_normalize),
                                       decimals=4), 5)]

    neural_precision = [round(np.around(precision_score(y_valid, neural_predict),
                                        decimals=4), 5),
                        round(np.around(precision_score(y_valid, neural_predict_normalize),
                                        decimals=4), 5),
                        round(np.around(precision_score(y_valid, neural_predict_scale),
                                        decimals=4), 5),
                        round(np.around(precision_score(y_valid, neural_predict_normalize_scale),
                                        decimals=4), 5),
                        round(np.around(precision_score(y_valid, neural_predict_scale_normalize),
                                        decimals=4), 5)]

    neural_recall = [round(np.around(recall_score(y_valid, neural_predict),
                                     decimals=4), 5),
                     round(np.around(recall_score(y_valid, neural_predict_normalize),
                                     decimals=4), 5),
                     round(np.around(recall_score(y_valid, neural_predict_scale),
                                     decimals=4), 5),
                     round(np.around(recall_score(y_valid, neural_predict_normalize_scale),
                                     decimals=4), 5),
                     round(np.around(recall_score(y_valid, neural_predict_scale_normalize),
                                     decimals=4), 5)]

    neural_f1 = [round(np.around(f1_score(y_valid, neural_predict),
                                 decimals=4), 5), round(np.around(f1_score(y_valid, neural_predict_normalize),
                                                                  decimals=4), 5),
                 round(np.around(f1_score(y_valid, neural_predict_scale),
                                 decimals=4), 5), round(np.around(f1_score(y_valid, neural_predict_normalize_scale),
                                                                  decimals=4), 5),
                 round(np.around(f1_score(y_valid, neural_predict_scale_normalize),
                                 decimals=4), 5)]

    neural_roc_auc = [round(np.around(roc_auc_score(y_valid, neural_predict),
                                      decimals=4), 5),
                      round(np.around(roc_auc_score(y_valid, neural_predict_normalize),
                                      decimals=4), 5),
                      round(np.around(roc_auc_score(y_valid, neural_predict_scale),
                                      decimals=4), 5),
                      round(np.around(roc_auc_score(y_valid, neural_predict_normalize_scale),
                                      decimals=4), 5),
                      round(np.around(roc_auc_score(y_valid, neural_predict_scale_normalize),
                                      decimals=4), 5)]

    neural_max_values_indexes = [neural_accuracy.index(max(neural_accuracy)),
                                 neural_precision.index(max(neural_precision)),
                                 neural_recall.index(max(neural_recall)),
                                 neural_f1.index(max(neural_f1)),
                                 neural_roc_auc.index(max(neural_roc_auc))]

    print(neural_roc_auc[max(set(neural_max_values_indexes), key=neural_max_values_indexes.count)])
