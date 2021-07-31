import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, fbeta_score, roc_auc_score, fbeta_score, f1_score
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import BaggingClassifier, StackingClassifier, AdaBoostClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns


def agregation(window):
    # window.cleaning()
    count = 0
    if window.checkBox_7.isChecked():
        count += 1

    if window.checkBox_8.isChecked():
        count += 1

    if window.checkBox_5.isChecked():
        count += 1

    if window.checkBox_6.isChecked():
        count += 1

    if window.checkBox_3.isChecked():
        count += 1

    if window.checkBox_4.isChecked():
        count += 1
    x = []
    y = []
    pred = []
    for i in range(count):
        x.append(window.x_train[
                 i * (len(window.x_train) // count):len(window.x_train) * (i + 1) // count])
        y.append(window.y_train[i * (len(window.y_train) // count):len(window.y_train) * (i + 1) // count])
        if window.label_2.text() != "Вы не открыли файл" and window.label_2.text() != " ":
            if window.lineEdit.text().isdigit():
                if 1 <= int(window.lineEdit.text()) <= 100:
                    # window.tableWidget.show()
                    # window.label_5.show()
                    # window.label_4.hide()
                    # window.pushButton_2.show()
                    if window.checkBox_7.isChecked():
                        # window.bayes(x[i], y[i], window.x_valid, window.y_valid)
                        pred.append(window.predprob_bayes)
                        window.checkBox_7.setChecked(False)
                        continue
                    if window.checkBox_8.isChecked():
                        # window.logistic_regression(x[i], y[i], window.x_valid, window.y_valid)
                        pred.append(window.predprob_logistic)
                        window.checkBox_8.setChecked(False)
                        continue
                    if window.checkBox_5.isChecked():
                        # window.svm_vectors(x[i], y[i], window.x_valid, window.y_valid)
                        pred.append(window.predprob_svm)
                        window.checkBox_5.setChecked(False)
                        continue
                    if window.checkBox_6.isChecked():
                        # window.discriminant_analysis(x[i], y[i], window.x_valid, window.y_valid)
                        pred.append(window.predprob_discriminant)
                        window.checkBox_6.setChecked(False)
                        continue
                    if window.checkBox_3.isChecked():
                        # window.tree(x[i], y[i], window.x_valid, window.y_valid)
                        pred.append(window.predprob_tree)
                        window.checkBox_3.setChecked(False)
                        continue
                    if window.checkBox_4.isChecked():
                        # window.neural_network(x[i], y[i], window.x_valid, window.y_valid)
                        pred.append(window.predprob_network)
                        window.checkBox_4.setChecked(False)
                        continue

    predd = []
    mean = []
    median = []
    voting = []
    for i in range(pred[0].shape[0]):
        preddd = []
        for j in range(count):
            preddd.append(pred[j][i][1])
            # print(pred[j][i][1])
        predd.append(preddd)
        mean.append(np.mean(predd[i]))
        median.append(np.median(predd[i]))

    predd = []
    for i in range(pred[0].shape[0]):
        preddd = []
        for j in range(count):
            if pred[j][i][1] >= 0.1:
                preddd.append(pred[j][i][1])
        predd.append(preddd)
        voting.append(np.mean(predd[i]))
    meann = [round(num) for num in mean]
    window.accuracy_score_agregation_mean = str(
        round(np.around(accuracy_score(window.y_valid, meann),
                        decimals=4) * 100, 5)) + "%"
    window.precision_score_agregation_mean = str(round(
        np.around(precision_score(window.y_valid, meann, zero_division=0),
                  decimals=4), 5))
    window.recall_score_agregation_mean = str(round(
        np.around(recall_score(window.y_valid, meann, zero_division=0), decimals=4), 5))
    window.f1_score_agregation_mean = str(round(
        np.around(f1_score(window.y_valid, meann, zero_division=0), decimals=4), 5))
    window.auc_score_agregation_mean = str(round(roc_auc_score(window.y_valid, meann), 5))
    mediann = [round(num) for num in median]
    window.accuracy_score_agregation_median = str(
        round(np.around(accuracy_score(window.y_valid, mediann),
                        decimals=4) * 100, 5)) + "%"
    window.precision_score_agregation_median = str(round(
        np.around(precision_score(window.y_valid, mediann, zero_division=0),
                  decimals=4), 5))
    window.recall_score_agregation_median = str(round(
        np.around(recall_score(window.y_valid, mediann, zero_division=0), decimals=4), 5))
    window.f1_score_agregation_median = str(round(
        np.around(f1_score(window.y_valid, mediann, zero_division=0), decimals=4), 5))
    votingg = [round(num) for num in voting]
    window.auc_score_agregation_voting = str(round(roc_auc_score(window.y_valid, votingg), 5))
    window.accuracy_score_agregation_voting = str(
        round(np.around(accuracy_score(window.y_valid, votingg),
                        decimals=4) * 100, 5)) + "%"
    window.precision_score_agregation_voting = str(round(
        np.around(precision_score(window.y_valid, votingg, zero_division=0),
                  decimals=4), 5))
    window.recall_score_agregation_voting = str(
        round(np.around(recall_score(window.y_valid, votingg, zero_division=0), decimals=4), 5))
    window.f1_score_agregation_voting = str(round(
        np.around(f1_score(window.y_valid, votingg, zero_division=0), decimals=4), 5))
    window.auc_score_agregation_voting = str(round(roc_auc_score(window.y_valid, votingg), 5))
    window.table()


def significant(window):
    last_name = window.data[window.data.columns[len(window.data.columns) - 1]].name
    window.count_of_false = len(window.data[window.data[last_name] == 0])
    window.count_of_true = len(window.data[window.data[last_name] == 1])

    if window.count_of_true > window.count_of_false:
        window.prevalence_percentage = window.count_of_true / (window.count_of_true + window.count_of_false)
    else:
        window.prevalence_percentage = window.count_of_false / (window.count_of_true + window.count_of_false)
    pass
    if window.prevalence_percentage >= 0.7:
        window.classes_disbalance = True
    else:
        window.classes_disbalance = False


def logistic_regression(window, x_train, x_valid, y_train, y_valid, x_train_normalize, x_train_scale,
                        x_train_normalize_scale, x_train_scale_normalize, x_valid_normalize, x_valid_scale,
                        x_valid_normalize_scale, x_valid_scale_normalize):
    if window.radioButton.isChecked():
        logistic = BaggingClassifier(SGDClassifier(loss='log'))
    elif window.radioButton_2.isChecked():
        logistic = StackingClassifier([('log', SGDClassifier(loss='log')), ('clf', MultinomialNB()), ('disc', LinearDiscriminantAnalysis()), ('sup', SVC(kernel="linear", C=0.025)), ('tree', DecisionTreeClassifier()), ('neur', MLPClassifier())], SGDClassifier(loss='log'))
    elif window.radioButton_3.isChecked():
        logistic = AdaBoostClassifier(SGDClassifier(loss='log'))
    else:
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

    window.logistic_f1 = [round(np.around(fbeta_score(y_valid, logistic_predict, 0.5),
                                          decimals=4), 5),
                          round(np.around(fbeta_score(y_valid, logistic_predict_normalize, 0.5),
                                          decimals=4), 5),
                          round(np.around(fbeta_score(y_valid, logistic_predict_scale, 0.5),
                                          decimals=4), 5),
                          round(np.around(fbeta_score(y_valid, logistic_predict_normalize_scale, 0.5),
                                          decimals=4), 5),
                          round(np.around(fbeta_score(y_valid, logistic_predict_scale_normalize, 0.5),
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

    window.logistic_accuracy = window.logistic_accuracy[
        max(set(window.logistic_max_values_indexes),
            key=window.logistic_max_values_indexes.count)]

    window.logistic_precision = window.logistic_precision[
        max(set(window.logistic_max_values_indexes),
            key=window.logistic_max_values_indexes.count)]

    window.logistic_recall = window.logistic_recall[
        max(set(window.logistic_max_values_indexes),
            key=window.logistic_max_values_indexes.count)]

    window.logistic_f1 = window.logistic_f1[
        max(set(window.logistic_max_values_indexes),
            key=window.logistic_max_values_indexes.count)]

    window.logistic_roc_auc = window.logistic_roc_auc[
        max(set(window.logistic_max_values_indexes),
            key=window.logistic_max_values_indexes.count)]

    logistic_predicts = [logistic_predict, logistic_predict_normalize, logistic_predict_scale,
                         logistic_predict_normalize_scale,
                         logistic_predict_scale_normalize]
    predict = logistic_predicts[max(set(window.logistic_max_values_indexes),
                                    key=window.logistic_max_values_indexes.count)]
    window.predprob_logistic = logistic.predict_proba[max(set(window.logistic_max_values_indexes),
                                    key=window.logistic_max_values_indexes.count)]
    cf_matrix = confusion_matrix(y_valid, predict)
    TN, FP, FN, TP = cf_matrix.ravel()
    group_names = ["TN", "FP", "FN", "TP"]
    group_counts = ["{0: 0.0f}".format(value) for value in cf_matrix.flatten()]
    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_names, group_counts)]
    labels = np.asarray(labels).reshape(2, 2)
    sns.heatmap(cf_matrix, annot=labels, fmt="", cmap="Blues", cbar=True, xticklabels=False,
                yticklabels=False)
    # plt.show()
    if window.logistic_roc_auc >= 0.7:
        window.logistic_roc_auc_significant = False
        window.logistic_f1_significant = True
        window.logistic_significant = window.logistic_f1
    if window.logistic_roc_auc < 0.7:
        window.logistic_roc_auc_significant = True
        window.logistic_f1_significant = False
        window.logistic_significant = window.logistic_roc_auc


def bayes(window, x_train, x_valid, y_train, y_valid, x_train_normalize, x_train_scale,
          x_train_normalize_scale, x_train_scale_normalize, x_valid_normalize, x_valid_scale,
          x_valid_normalize_scale, x_valid_scale_normalize):
    if window.radioButton.isChecked():
        clf = BaggingClassifier(MultinomialNB())
    elif window.radioButton_2.isChecked():
        clf = StackingClassifier([('log', SGDClassifier(loss='log')), ('clf', MultinomialNB()), ('disc', LinearDiscriminantAnalysis()), ('sup', SVC(kernel="linear", C=0.025)), ('tree', DecisionTreeClassifier()), ('neur', MLPClassifier())], MultinomialNB())
    elif window.radioButton_3.isChecked():
        clf = AdaBoostClassifier(MultinomialNB())
    else:
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

        window.clf_f1 = [round(np.around(fbeta_score(y_valid, clf_predict, 0.5),
                                         decimals=4), 5), round(np.around(fbeta_score(y_valid, clf_predict_normalize),
                                                                          decimals=4), 5),
                         round(np.around(fbeta_score(y_valid, clf_predict_scale, 0.5),
                                         decimals=4), 5),
                         round(np.around(fbeta_score(y_valid, clf_predict_normalize_scale, 0.5),
                                         decimals=4), 5),
                         round(np.around(fbeta_score(y_valid, clf_predict_scale_normalize, 0.5),
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

        window.clf_f1 = [round(np.around(fbeta_score(y_valid, clf_predict, 1),
                                         decimals=4), 5), round(np.around(fbeta_score(y_valid, clf_predict_normalize, 1),
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

        window.clf_accuracy = window.clf_accuracy[
            max(set(window.clf_max_values_indexes),
                key=window.clf_max_values_indexes.count)]

        window.clf_precision = window.clf_precision[
            max(set(window.clf_max_values_indexes),
                key=window.clf_max_values_indexes.count)]

        window.clf_recall = window.clf_recall[
            max(set(window.clf_max_values_indexes),
                key=window.clf_max_values_indexes.count)]

        window.clf_f1 = window.clf_f1[
            max(set(window.clf_max_values_indexes),
                key=window.clf_max_values_indexes.count)]

        window.clf_roc_auc = window.clf_roc_auc[
            max(set(window.clf_max_values_indexes),
                key=window.clf_max_values_indexes.count)]
        try:
            clf_predicts = [clf_predict, clf_predict_normalize, clf_predict_scale,
                            clf_predict_normalize_scale,
                            clf_predict_scale_normalize]
            predict = clf_predicts[max(set(window.clf_max_values_indexes),
                                       key=window.clf_max_values_indexes.count)]
            window.predprob_bayes = clf.predict_proba[max(set(window.clf_max_values_indexes),
                                       key=window.clf_max_values_indexes.count)]
            cf_matrix = confusion_matrix(y_valid, predict)
            TN, FP, FN, TP = cf_matrix.ravel()
            group_names = ["TN", "FP", "FN", "TP"]
            group_counts = ["{0: 0.0f}".format(value) for value in cf_matrix.flatten()]
            labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_names, group_counts)]
            labels = np.asarray(labels).reshape(2, 2)
            sns.heatmap(cf_matrix, annot=labels, fmt="", cmap="Blues", cbar=True, xticklabels=False,
                        yticklabels=False)
            # plt.show()
            len_cf_matrix = TN + FP + FN + TP
            if window.clf_roc_auc >= 0.7:
                window.clf_roc_auc_significant = False
                window.clf_f1_significant = True
                window.clf_significant = window.clf_f1
            if window.clf_roc_auc < 0.7:
                window.clf_roc_auc_significant = True
                window.clf_f1_significant = False
                window.clf_significant = window.clf_roc_auc
            pass
        except:
            clf_predicts = [clf_predict, clf_predict_normalize]
            predict = clf_predicts[max(set(window.clf_max_values_indexes),
                                       key=window.clf_max_values_indexes.count)]
            cf_matrix = confusion_matrix(y_valid, predict)
            TN, FP, FN, TP = cf_matrix.ravel()
            group_names = ["TN", "FP", "FN", "TP"]
            group_counts = ["{0: 0.0f}".format(value) for value in cf_matrix.flatten()]
            labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_names, group_counts)]
            labels = np.asarray(labels).reshape(2, 2)
            sns.heatmap(cf_matrix, annot=labels, fmt="", cmap="Blues", cbar=True, xticklabels=False,
                        yticklabels=False)
            # plt.show()
            len_cf_matrix = TN + FP + FN + TP
            if window.clf_roc_auc >= 0.7:
                window.clf_roc_auc_significant = False
                window.clf_f1_significant = True
                window.clf_significant = window.clf_f1
            if window.clf_roc_auc < 0.7:
                window.clf_roc_auc_significant = True
                window.clf_f1_significant = False
                window.clf_significant = window.clf_roc_auc


def discriminant_analysis(window, x_train, x_valid, y_train, y_valid, x_train_normalize, x_train_scale,
                          x_train_normalize_scale, x_train_scale_normalize, x_valid_normalize, x_valid_scale,
                          x_valid_normalize_scale, x_valid_scale_normalize):
    
    if window.radioButton.isChecked():
        disc = BaggingClassifier(LinearDiscriminantAnalysis())
    elif window.radioButton_2.isChecked():
        disc = StackingClassifier([('log', SGDClassifier(loss='log')), ('clf', MultinomialNB()), ('disc', LinearDiscriminantAnalysis()), ('sup', SVC(kernel="linear", C=0.025)), ('tree', DecisionTreeClassifier()), ('neur', MLPClassifier())], LinearDiscriminantAnalysis())
    elif window.radioButton_3.isChecked():
        disc = AdaBoostClassifier(LinearDiscriminantAnalysis())
    else:
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

    window.disc_f1 = [round(np.around(fbeta_score(y_valid, disc_predict, 0.5),
                                      decimals=4), 5), round(np.around(fbeta_score(y_valid, disc_predict_normalize, 0.5),
                                                                       decimals=4), 5),
                      round(np.around(fbeta_score(y_valid, disc_predict_scale, 0.5),
                                      decimals=4), 5), round(np.around(fbeta_score(y_valid, disc_predict_normalize_scale, 0.5),
                                                                       decimals=4), 5),
                      round(np.around(fbeta_score(y_valid, disc_predict_scale_normalize, 0.5),
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

    window.disc_accuracy = window.disc_accuracy[
        max(set(window.disc_max_values_indexes),
            key=window.disc_max_values_indexes.count)]

    window.disc_precision = window.disc_precision[
        max(set(window.disc_max_values_indexes),
            key=window.disc_max_values_indexes.count)]

    window.disc_recall = window.disc_recall[
        max(set(window.disc_max_values_indexes),
            key=window.disc_max_values_indexes.count)]

    window.disc_f1 = window.disc_f1[
        max(set(window.disc_max_values_indexes),
            key=window.disc_max_values_indexes.count)]

    window.disc_roc_auc = window.disc_roc_auc[
        max(set(window.disc_max_values_indexes),
            key=window.disc_max_values_indexes.count)]

    disc_predicts = [disc_predict, disc_predict_normalize, disc_predict_scale,
                     disc_predict_normalize_scale,
                     disc_predict_scale_normalize]
    predict = disc_predicts[max(set(window.disc_max_values_indexes),
                                key=window.disc_max_values_indexes.count)]
    cf_matrix = confusion_matrix(y_valid, predict)
    TN, FP, FN, TP = cf_matrix.ravel()
    group_names = ["TN", "FP", "FN", "TP"]
    group_counts = ["{0: 0.0f}".format(value) for value in cf_matrix.flatten()]
    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_names, group_counts)]
    labels = np.asarray(labels).reshape(2, 2)
    sns.heatmap(cf_matrix, annot=labels, fmt="", cmap="Blues", cbar=True, xticklabels=False,
                yticklabels=False)
    # plt.show()

    len_cf_matrix = TN + FP + FN + TP
    if window.disc_roc_auc >= 0.7:
        window.disc_roc_auc_significant = False
        window.disc_f1_significant = True
        window.disc_significant = window.disc_f1
    if window.disc_roc_auc < 0.7:
        window.disc_roc_auc_significant = True
        window.disc_f1_significant = False
        window.disc_significant = window.disc_roc_auc


def svm_vectors(window, x_train, x_valid, y_train, y_valid, x_train_normalize, x_train_scale,
                x_train_normalize_scale, x_train_scale_normalize, x_valid_normalize, x_valid_scale,
                x_valid_normalize_scale, x_valid_scale_normalize):
    if window.radioButton.isChecked():
        support = BaggingClassifier(SVC(kernel='linear', C=0.025))
    elif window.radioButton_2.isChecked():
        support = StackingClassifier([('log', SGDClassifier(loss='log')), ('clf', MultinomialNB()), ('disc', LinearDiscriminantAnalysis()), ('sup', SVC(kernel="linear", C=0.025)), ('tree', DecisionTreeClassifier()), ('neur', MLPClassifier())], SVC(kernel='linear', C=0.025))
    elif window.radioButton_3.isChecked():
        support = AdaBoostClassifier(SVC(kernel='linear', C=0.025), algorithm='SAMME')
    else:
        support = GridSearchCV(SVC(kernel='linear', C=0.025), {'max_iter': range(-1, 1000)}, scoring='roc_auc')
        # support = SVC(kernel='linear', C=0.025)
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

    window.support_f1 = [round(np.around(fbeta_score(y_valid, support_predict, 0.5),
                                         decimals=4), 5), round(np.around(fbeta_score(y_valid, support_predict_normalize, 0.5),
                                                                          decimals=4), 5),
                         round(np.around(fbeta_score(y_valid, support_predict_scale, 0.5),
                                         decimals=4), 5),
                         round(np.around(fbeta_score(y_valid, support_predict_normalize_scale, 0.5),
                                         decimals=4), 5),
                         round(np.around(fbeta_score(y_valid, support_predict_scale_normalize, 0.5),
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

    window.support_accuracy = window.support_accuracy[
        max(set(window.support_max_values_indexes),
            key=window.support_max_values_indexes.count)]

    window.support_precision = window.support_precision[
        max(set(window.support_max_values_indexes),
            key=window.support_max_values_indexes.count)]

    window.support_recall = window.support_recall[
        max(set(window.support_max_values_indexes),
            key=window.support_max_values_indexes.count)]

    window.support_f1 = window.support_f1[
        max(set(window.support_max_values_indexes),
            key=window.support_max_values_indexes.count)]

    window.support_roc_auc = window.support_roc_auc[
        max(set(window.support_max_values_indexes),
            key=window.support_max_values_indexes.count)]

    support_predicts = [support_predict, support_predict_normalize, support_predict_scale,
                        support_predict_normalize_scale,
                        support_predict_scale_normalize]
    predict = support_predicts[max(set(window.support_max_values_indexes),
                                   key=window.support_max_values_indexes.count)]
    cf_matrix = confusion_matrix(y_valid, predict)
    TN, FP, FN, TP = cf_matrix.ravel()
    group_names = ["TN", "FP", "FN", "TP"]
    group_counts = ["{0: 0.0f}".format(value) for value in cf_matrix.flatten()]
    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_names, group_counts)]
    labels = np.asarray(labels).reshape(2, 2)
    sns.heatmap(cf_matrix, annot=labels, fmt="", cmap="Blues", cbar=True, xticklabels=False,
                yticklabels=False)
    # plt.show()
    len_cf_matrix = TN + FP + FN + TP
    if window.support_roc_auc >= 0.7:
        window.support_roc_auc_significant = False
        window.support_f1_significant = True
        window.support_significant = window.support_f1
    if window.support_roc_auc < 0.7:
        window.support_roc_auc_significant = True
        window.support_f1_significant = False
        window.support_significant = window.support_roc_auc


def tree(window, x_train, x_valid, y_train, y_valid, x_train_normalize, x_train_scale,
         x_train_normalize_scale, x_train_scale_normalize, x_valid_normalize, x_valid_scale,
         x_valid_normalize_scale, x_valid_scale_normalize):
    if window.radioButton.isChecked():
        tree = BaggingClassifier(DecisionTreeClassifier())
    elif window.radioButton_2.isChecked():
        tree = StackingClassifier([('log', SGDClassifier(loss='log')), ('clf', MultinomialNB()), ('disc', LinearDiscriminantAnalysis()), ('sup', SVC(kernel="linear", C=0.025)), ('tree', DecisionTreeClassifier()), ('neur', MLPClassifier())], DecisionTreeClassifier())
    elif window.radioButton_3.isChecked():
        tree = AdaBoostClassifier(DecisionTreeClassifier())
    else:
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

    window.tree_f1 = [round(np.around(fbeta_score(y_valid, tree_predict, 0.5),
                                      decimals=4), 5), round(np.around(fbeta_score(y_valid, tree_predict_normalize, 0.5),
                                                                       decimals=4), 5),
                      round(np.around(fbeta_score(y_valid, tree_predict_scale, 0.5),
                                      decimals=4), 5), round(np.around(fbeta_score(y_valid, tree_predict_normalize_scale, 0.5),
                                                                       decimals=4), 5),
                      round(np.around(fbeta_score(y_valid, tree_predict_scale_normalize, 0.5),
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

    window.tree_accuracy = window.tree_accuracy[
        max(set(window.tree_max_values_indexes),
            key=window.tree_max_values_indexes.count)]

    window.tree_precision = window.tree_precision[
        max(set(window.tree_max_values_indexes),
            key=window.tree_max_values_indexes.count)]

    window.tree_recall = window.tree_recall[
        max(set(window.tree_max_values_indexes),
            key=window.tree_max_values_indexes.count)]

    window.tree_f1 = window.tree_f1[
        max(set(window.tree_max_values_indexes),
            key=window.tree_max_values_indexes.count)]

    window.tree_roc_auc = window.tree_roc_auc[
        max(set(window.tree_max_values_indexes),
            key=window.tree_max_values_indexes.count)]

    tree_predicts = [tree_predict, tree_predict_normalize, tree_predict_scale,
                     tree_predict_normalize_scale,
                     tree_predict_scale_normalize]
    predict = tree_predicts[max(set(window.tree_max_values_indexes),
                                key=window.tree_max_values_indexes.count)]
    cf_matrix = confusion_matrix(y_valid, predict)
    TN, FP, FN, TP = cf_matrix.ravel()
    group_names = ["TN", "FP", "FN", "TP"]
    group_counts = ["{0: 0.0f}".format(value) for value in cf_matrix.flatten()]
    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_names, group_counts)]
    labels = np.asarray(labels).reshape(2, 2)
    sns.heatmap(cf_matrix, annot=labels, fmt="", cmap="Blues", cbar=True, xticklabels=False,
                yticklabels=False)
    # plt.show()
    len_cf_matrix = TN + FP + FN + TP
    if window.tree_roc_auc >= 0.7:
        window.tree_roc_auc_significant = False
        window.tree_f1_significant = True
        window.tree_significant = window.tree_f1
    if window.tree_roc_auc < 0.7:
        window.tree_roc_auc_significant = True
        window.tree_f1_significant = False
        window.tree_significant = window.tree_roc_auc


def neural_network(window, x_train, x_valid, y_train, y_valid, x_train_normalize, x_train_scale,
                   x_train_normalize_scale, x_train_scale_normalize, x_valid_normalize, x_valid_scale,
                   x_valid_normalize_scale, x_valid_scale_normalize):
    if window.radioButton.isChecked():
        neural = BaggingClassifier(MLPClassifier())
    elif window.radioButton_2.isChecked():
        neural = StackingClassifier([('log', SGDClassifier(loss='log')), ('clf', MultinomialNB()), ('disc', LinearDiscriminantAnalysis()), ('sup', SVC(kernel="linear", C=0.025)), ('tree', DecisionTreeClassifier()), ('neur', MLPClassifier())], MLPClassifier())
    elif window.radioButton_3.isChecked():
        # neural = AdaBoostClassifier(MLPClassifier(), algorithm='SAMME')
        neural = MLPClassifier()
    else:
        neural = GridSearchCV(MLPClassifier(), {'max_iter': range(175, 225)}, scoring='roc_auc')
        # neural = MLPClassifier()
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

    window.neural_f1 = [round(np.around(fbeta_score(y_valid, neural_predict, 0.5),
                                        decimals=4), 5), round(np.around(fbeta_score(y_valid, neural_predict_normalize, 0.5),
                                                                         decimals=4), 5),
                        round(np.around(fbeta_score(y_valid, neural_predict_scale, 0.5),
                                        decimals=4), 5),
                        round(np.around(fbeta_score(y_valid, neural_predict_normalize_scale, 0.5),
                                        decimals=4), 5),
                        round(np.around(fbeta_score(y_valid, neural_predict_scale_normalize, 0.5),
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

    neural_predicts = [neural_predict, neural_predict_normalize, neural_predict_scale, neural_predict_normalize_scale,
                       neural_predict_scale_normalize]
    predict = neural_predicts[max(set(window.neural_max_values_indexes),
                                  key=window.neural_max_values_indexes.count)]
    cf_matrix = confusion_matrix(y_valid, predict)
    TN, FP, FN, TP = cf_matrix.ravel()
    group_names = ["TN", "FP", "FN", "TP"]
    group_counts = ["{0: 0.0f}".format(value) for value in cf_matrix.flatten()]
    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_names, group_counts)]
    labels = np.asarray(labels).reshape(2, 2)
    sns.heatmap(cf_matrix, annot=labels, fmt="", cmap="Blues", cbar=True, xticklabels=False,
                yticklabels=False)
    # plt.show()
    len_cf_matrix = TN + FP + FN + TP

    window.neural_accuracy = window.neural_accuracy[
        max(set(window.neural_max_values_indexes),
            key=window.neural_max_values_indexes.count)]

    window.neural_precision = window.neural_precision[
        max(set(window.neural_max_values_indexes),
            key=window.neural_max_values_indexes.count)]

    window.neural_recall = window.neural_recall[
        max(set(window.neural_max_values_indexes),
            key=window.neural_max_values_indexes.count)]

    window.neural_f1 = window.neural_f1[
        max(set(window.neural_max_values_indexes),
            key=window.neural_max_values_indexes.count)]

    window.neural_roc_auc = window.neural_roc_auc[
        max(set(window.neural_max_values_indexes),
            key=window.neural_max_values_indexes.count)]

    if window.neural_roc_auc >= 0.7:
        window.neural_roc_auc_significant = False
        window.neural_f1_significant = True
        window.neural_significant = window.neural_f1
    if window.neural_roc_auc < 0.7:
        window.neural_roc_auc_significant = True
        window.neural_f1_significant = False
        window.neural_significant = window.neural_roc_auc
