from numpy import mean
from scipy.stats import t, sem
from math import sqrt
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5.QtWidgets import QTableWidgetItem
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


def tstat(window):
    def independent_ttest(data1, data2, alpha):
        # calculate means
        mean1, mean2 = mean(data1), mean(data2)
        # calculate standard errors
        se1, se2 = sem(data1), sem(data2)
        # standard error on the difference between the samples
        sed = sqrt(se1 ** 2.0 + se2 ** 2.0)
        # calculate the t statistic
        t_stat = (mean1 - mean2) / sed
        # degrees of freedom
        df = len(data1) + len(data2) - 2
        # calculate the critical value
        cv = t.ppf(1.0 - alpha / 2, df)
        # calculate the p-value
        p = (1.0 - t.cdf(abs(t_stat), df)) * 2.0
        # return everything
        return t_stat, df, cv, p

    window.window2.tableWidget.setColumnCount(3)
    window.window2.tableWidget.setRowCount(len(window.old_data.columns) - 1)
    for i in range(len(window.old_data.columns)):
        window.window2.tableWidget.setVerticalHeaderItem(i, QTableWidgetItem(window.old_data.columns[i]))
    window.window2.tableWidget.setHorizontalHeaderItem(0, QTableWidgetItem("Значимость"))
    window.window2.tableWidget.setHorizontalHeaderItem(1, QTableWidgetItem("T - значение"))
    window.window2.tableWidget.setHorizontalHeaderItem(2, QTableWidgetItem("P - значение"))
    header = window.window2.tableWidget.horizontalHeader()  # По размеру колонок
    header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
    alpha = float(window.window2.lineEdit.text())

    true_t_metrics = []
    true_t_metrics_indexes = []
    false_t_metrics_indexes = []
    last_name = window.old_data[window.old_data.columns[len(window.old_data.columns) - 1]].name
    for i in range(len(window.old_data.columns) - 1):
        data1 = window.old_data[window.old_data[last_name] == 0][window.old_data.columns[i]].to_numpy()
        data2 = window.old_data[window.old_data[last_name] == 1][window.old_data.columns[i]].to_numpy()
        t_stat, df, cv, p = independent_ttest(data1, data2, alpha)
        window.window2.label_3.setText(str(df))
        window.window2.label_4.setText(str(np.round(cv, 2)))
        window.window2.tableWidget.setItem(i, 1, QTableWidgetItem(str(np.round(t_stat, 2))))
        window.window2.tableWidget.setItem(i, 2, QTableWidgetItem(str(np.round(p, 2))))
        if abs(t_stat) <= cv and p > alpha:
            true_t_metrics.append(str(window.old_data[window.old_data.columns[i]].name))
            true_t_metrics_indexes.append(i)
        else:
            false_t_metrics_indexes.append(i)

    true_t_metrics.append(window.old_data[window.old_data.columns[len(window.old_data.columns) - 1]].name)
    window.new_data = window.old_data[(true_t_metrics)]
    count_of_false = len(window.data[window.data[last_name] == 0])
    count_of_true = len(window.data[window.data[last_name] == 1])
    print(count_of_false)
    print(count_of_true)

    if count_of_true > count_of_false:
        prevalence_percentage = count_of_true / (count_of_true + count_of_false)
    else:
        prevalence_percentage = count_of_false / (count_of_true + count_of_false)
    if prevalence_percentage > 0.8 and count_of_true > count_of_false:
        print("Неравенство классов, " + str(
            round(prevalence_percentage, 2) * 100) + "% Преобладание положительного класса")
    elif prevalence_percentage > 0.8 and count_of_true < count_of_false:
        print("Неравенство классов, " + str(
            round(prevalence_percentage, 2) * 100) + "% Преобладание отрицательного класса")

    for i in true_t_metrics_indexes:
        window.window2.tableWidget.setItem(i, 0, QTableWidgetItem("✅"))
    for i in false_t_metrics_indexes:
        window.window2.tableWidget.setItem(i, 0, QTableWidgetItem("❌"))

    window.data = window.new_data
    window.Y = window.data[window.data.columns[-1]].astype('int')
    window.X = window.data.drop(window.data.columns[-1], axis=1)
    # print(window.X)
    window.logistic_pred_flag = False
    window.clf_pred_flag = False
    window.disc_pred_flag = False
    window.support_pred_flag = False
    window.tree_pred_flag = False
    window.neural_pred_flag = False
    window.new_data_flag = False
    window.crossval_count = int(window.lineEdit_2.text())
    window.X_train, window.X_valid, window.Y_train, window.Y_valid = train_test_split(window.X, window.Y,
                                                                                      test_size=window.percent)
    window.X_train_normalized = preprocessing.normalize(window.X_train)
    window.X_valid_normalized = preprocessing.normalize(window.X_valid)
    cleaning(window)

def cleaning(window):
    try:
        del window.logistic_accuracy
        del window.precision_score_logistic
        del window.recall_score_logistic
        del window.f1_score_logistic
        del window.auc_score_logistic
    except:
        pass
    try:
        del window.accuracy_score_bayes
        del window.precision_score_bayes
        del window.recall_score_bayes
        del window.f1_score_bayes
        del window.auc_score_bayes
    except:
        pass
    try:
        del window.accuracy_score_discriminant
        del window.precision_score_discriminant
        del window.recall_score_discriminant
        del window.f1_score_discriminant
        del window.auc_score_discriminant
    except:
        pass
    try:
        del window.accuracy_score_svm
        del window.precision_score_svm
        del window.recall_score_svm
        del window.f1_score_svm
        del window.auc_score_svm
    except:
        pass
    try:
        del window.accuracy_score_tree
        del window.precision_score_tree
        del window.recall_score_tree
        del window.f1_score_tree
        del window.auc_score_tree
    except:
        pass
    try:
        del window.accuracy_score_network
        del window.precision_score_network
        del window.recall_score_network
        del window.f1_score_network
        del window.auc_score_network
    except:
        pass
    try:
        del window.accuracy_score_agregation_mean
        del window.precision_score_agregation_mean
        del window.recall_score_agregation_mean
        del window.f1_score_agregation_mean
        del window.auc_score_agregation_mean
    except:
        pass
    try:
        del window.accuracy_score_agregation_median
        del window.precision_score_agregation_median
        del window.recall_score_agregation_median
        del window.f1_score_agregation_median
        del window.auc_score_agregation_median
    except:
        pass
    try:
        del window.accuracy_score_agregation_voting
        del window.precision_score_agregation_voting
        del window.recall_score_agregation_voting
        del window.f1_score_agregation_voting
        del window.auc_score_agregation_voting
    except:
        pass

def significance(window):
    if window.checkBox_9.isChecked():
        window.window2.move(1222, 150)
        window.window2.show()
        tstat(window)
    else:
        window.window2.hide()
        window.data = window.old_data
        window.Y = window.data[window.data.columns[-1]].astype('int')
        window.X = window.data.drop(window.data.columns[-1], axis=1)
        # print(window.X)
        window.logistic_pred_flag = False
        window.clf_pred_flag = False
        window.disc_pred_flag = False
        window.support_pred_flag = False
        window.tree_pred_flag = False
        window.neural_pred_flag = False
        window.new_data_flag = False
        window.crossval_count = int(window.lineEdit_2.text())
        window.X_train, window.X_valid, window.Y_train, window.Y_valid = train_test_split(window.X, window.Y,
                                                                                          test_size=window.percent)
        window.X_train_normalized = preprocessing.normalize(window.X_train)
        window.X_valid_normalized = preprocessing.normalize(window.X_valid)
        cleaning(window)
