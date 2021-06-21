import sys
import os
import xlwt
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5.QtWidgets import QTableWidgetItem
import design
import numpy as np
from numpy import mean
from scipy.stats import t, sem, ttest_ind
from math import sqrt
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import BaggingClassifier


class Window(QtWidgets.QMainWindow, design.Ui_MainWindow):
    auc_score_agregation_mean: str
    logistic_pred_flag: bool

    def __init__(self):
        # Это здесь нужно для доступа к переменным, методам
        # и т.д. в файле design.py
        super().__init__()
        self.setupUi(self)  # Это нужно для инициализации нашего дизайна
        self.window2 = Window_Two()
        self.pushButton.clicked.connect(self.browse_folder)
        self.pushButton_4.clicked.connect(self.radio_choise)
        self.pushButton_2.clicked.connect(self.output_analys)
        self.pushButton_3.clicked.connect(self.browse_new_data)
        self.pushButton_5.clicked.connect(self.radio_choise)
        self.pushButton_5.clicked.connect(self.tstat)
        self.pushButton_6.clicked.connect(self.agregation)
        self.checkBox_3.setChecked(True)
        self.checkBox_4.setChecked(True)
        self.checkBox_5.setChecked(True)
        self.checkBox_6.setChecked(True)
        self.checkBox_7.setChecked(True)
        self.checkBox_8.setChecked(True)
        self.checkBox.clicked.connect(self.checkCross)
        self.checkBox.clicked.connect(self.table)
        self.checkBox_9.clicked.connect(self.significance)
        self.window2.pushButton.clicked.connect(self.significance)
        self.window2.pushButton.clicked.connect(self.tstat)
        self.label_2.hide()
        self.label_5.hide()
        self.tableWidget.hide()
        self.pushButton_2.hide()
        self.lineEdit_2.hide()

    def repredict(self):
        self.window2.show()

    def tstat(self):

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

        self.window2.tableWidget.setColumnCount(3)
        self.window2.tableWidget.setRowCount(len(self.old_data.columns) - 1)
        for i in range(len(self.old_data.columns)):
            self.window2.tableWidget.setVerticalHeaderItem(i, QTableWidgetItem(self.old_data.columns[i]))
        self.window2.tableWidget.setHorizontalHeaderItem(0, QTableWidgetItem("Значимость"))
        self.window2.tableWidget.setHorizontalHeaderItem(1, QTableWidgetItem("T - значение"))
        self.window2.tableWidget.setHorizontalHeaderItem(2, QTableWidgetItem("P - значение"))
        header = self.window2.tableWidget.horizontalHeader()  # По размеру колонок
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        alpha = float(self.window2.lineEdit.text())


        true_t_metrics = []
        true_t_metrics_indexes = []
        false_t_metrics_indexes = []
        last_name = self.old_data[self.old_data.columns[len(self.old_data.columns) - 1]].name
        for i in range(len(self.old_data.columns) - 1):
            data1 = self.old_data[self.old_data[last_name] == 0][self.old_data.columns[i]].to_numpy()
            data2 = self.old_data[self.old_data[last_name] == 1][self.old_data.columns[i]].to_numpy()
            t_stat, df, cv, p = independent_ttest(data1, data2, alpha)
            self.window2.label_3.setText(str(df))
            self.window2.label_4.setText(str(np.round(cv, 2)))
            self.window2.tableWidget.setItem(i, 1, QTableWidgetItem(str(np.round(t_stat, 2))))
            self.window2.tableWidget.setItem(i, 2, QTableWidgetItem(str(np.round(p, 2))))
            if abs(t_stat) <= cv and p > alpha:
                true_t_metrics.append(str(self.old_data[self.old_data.columns[i]].name))
                true_t_metrics_indexes.append(i)
            else:
                false_t_metrics_indexes.append(i)

        true_t_metrics.append(self.old_data[self.old_data.columns[len(self.old_data.columns) - 1]].name)
        self.new_data = self.old_data[(true_t_metrics)]
        count_of_false = len(self.data[self.data[last_name] == 0])
        count_of_true = len(self.data[self.data[last_name] == 1])
        print(count_of_false)
        print(count_of_true)

        if count_of_true > count_of_false:
            prevalence_percentage = count_of_true / (count_of_true + count_of_false)
        else:
            prevalence_percentage = count_of_false / (count_of_true + count_of_false)
        if prevalence_percentage > 0.8 and count_of_true > count_of_false:
            print("Неравенство классов, " + str(round(prevalence_percentage, 2) * 100) + "% Преобладание положительного класса")
        elif prevalence_percentage > 0.8 and count_of_true < count_of_false:
            print("Неравенство классов, " + str(round(prevalence_percentage, 2) * 100) + "% Преобладание отрицательного класса")

        for i in true_t_metrics_indexes:
            self.window2.tableWidget.setItem(i, 0, QTableWidgetItem("✅"))
        for i in false_t_metrics_indexes:
            self.window2.tableWidget.setItem(i, 0, QTableWidgetItem("❌"))

        self.data = self.new_data
        self.Y = self.data[self.data.columns[-1]].astype('int')
        self.X = self.data.drop(self.data.columns[-1], axis=1)
        #print(self.X)
        self.logistic_pred_flag = False
        self.clf_pred_flag = False
        self.disc_pred_flag = False
        self.support_pred_flag = False
        self.tree_pred_flag = False
        self.neural_pred_flag = False
        self.new_data_flag = False
        self.crossval_count = int(self.lineEdit_2.text())
        self.X_train, self.X_valid, self.Y_train, self.Y_valid = train_test_split(self.X, self.Y, test_size=self.percent)
        self.X_train_normalized = preprocessing.normalize(self.X_train)
        self.X_valid_normalized = preprocessing.normalize(self.X_valid)
        self.Ycsv = self.Y_valid.to_frame()
        self.Ycsv.set_axis(['valid'], axis=1, inplace=True)
        self.Ycsv.to_excel('prediction.xls')
        self.cleaning()

    def cleaning(self):
        try:
            del self.logistic_accuracy
            del self.precision_score_logistic
            del self.recall_score_logistic
            del self.f1_score_logistic
            del self.auc_score_logistic
        except:
            pass
        try:
            del self.accuracy_score_bayes
            del self.precision_score_bayes
            del self.recall_score_bayes
            del self.f1_score_bayes
            del self.auc_score_bayes
        except:
            pass
        try:
            del self.accuracy_score_discriminant
            del self.precision_score_discriminant
            del self.recall_score_discriminant
            del self.f1_score_discriminant
            del self.auc_score_discriminant
        except:
            pass
        try:
            del self.accuracy_score_svm
            del self.precision_score_svm
            del self.recall_score_svm
            del self.f1_score_svm
            del self.auc_score_svm
        except:
            pass
        try:
            del self.accuracy_score_tree
            del self.precision_score_tree
            del self.recall_score_tree
            del self.f1_score_tree
            del self.auc_score_tree
        except:
            pass
        try:
            del self.accuracy_score_network
            del self.precision_score_network
            del self.recall_score_network
            del self.f1_score_network
            del self.auc_score_network
        except:
            pass
        try:
            del self.accuracy_score_agregation_mean
            del self.precision_score_agregation_mean
            del self.recall_score_agregation_mean
            del self.f1_score_agregation_mean
            del self.auc_score_agregation_mean
        except:
            pass
        try:
            del self.accuracy_score_agregation_median
            del self.precision_score_agregation_median
            del self.recall_score_agregation_median
            del self.f1_score_agregation_median
            del self.auc_score_agregation_median
        except:
            pass
        try:
            del self.accuracy_score_agregation_voting
            del self.precision_score_agregation_voting
            del self.recall_score_agregation_voting
            del self.f1_score_agregation_voting
            del self.auc_score_agregation_voting
        except:
            pass

    def table(self):
        self.tableWidget.setColumnCount(5)
        self.tableWidget.setRowCount(9)
        self.tableWidget.setHorizontalHeaderLabels(
            ["Доля верной классификации", "Точность", "Полнота", "F-мера", "AUC"])
        self.tableWidget.setVerticalHeaderLabels(
            ["Логистическая регрессия", "Байесовский классификатор", "Дискриминантный анализ", "Опорные вектора",
             "Деревья решений", "Нейронная сеть", "Агрегирование по среднему", "Агрегирование по медиане",
             "Агрегирование по голосованию"])
        column = 0  # столбец
        row = 0  # строка
        try:
            self.tabl
            self.tableWidget.setItem(row, column, QTableWidgetItem(self.accuracy_score_logistic))
            self.tableWidget.setItem(row, column + 1, QTableWidgetItem(self.precision_score_logistic))
            self.tableWidget.setItem(row, column + 2, QTableWidgetItem(self.recall_score_logistic))
            self.tableWidget.setItem(row, column + 3, QTableWidgetItem(self.f1_score_logistic))
            self.tableWidget.setItem(row, column + 4, QTableWidgetItem(self.auc_score_logistic))
            # row += 1
        except:
            pass
        row += 1
        try:
            self.tableWidget.setItem(row, column, QTableWidgetItem(self.accuracy_score_bayes))
            self.tableWidget.setItem(row, column + 1, QTableWidgetItem(self.precision_score_bayes))
            self.tableWidget.setItem(row, column + 2, QTableWidgetItem(self.recall_score_bayes))
            self.tableWidget.setItem(row, column + 3, QTableWidgetItem(self.f1_score_bayes))
            self.tableWidget.setItem(row, column + 4, QTableWidgetItem(self.auc_score_bayes))
            # row += 1
        except:
            pass
        row += 1
        try:
            self.tableWidget.setItem(row, column, QTableWidgetItem(self.accuracy_score_discriminant))
            self.tableWidget.setItem(row, column + 1, QTableWidgetItem(self.precision_score_discriminant))
            self.tableWidget.setItem(row, column + 2, QTableWidgetItem(self.recall_score_discriminant))
            self.tableWidget.setItem(row, column + 3, QTableWidgetItem(self.f1_score_discriminant))
            self.tableWidget.setItem(row, column + 4, QTableWidgetItem(self.auc_score_discriminant))
            # row += 1
        except:
            pass
        row += 1
        try:
            self.tableWidget.setItem(row, column, QTableWidgetItem(self.accuracy_score_svm))
            self.tableWidget.setItem(row, column + 1, QTableWidgetItem(self.precision_score_svm))
            self.tableWidget.setItem(row, column + 2, QTableWidgetItem(self.recall_score_svm))
            self.tableWidget.setItem(row, column + 3, QTableWidgetItem(self.f1_score_svm))
            self.tableWidget.setItem(row, column + 4, QTableWidgetItem(self.auc_score_svm))
            # row += 1
        except:
            pass
        row += 1
        try:
            self.tableWidget.setItem(row, column, QTableWidgetItem(self.accuracy_score_tree))
            self.tableWidget.setItem(row, column + 1, QTableWidgetItem(self.precision_score_tree))
            self.tableWidget.setItem(row, column + 2, QTableWidgetItem(self.recall_score_tree))
            self.tableWidget.setItem(row, column + 3, QTableWidgetItem(self.f1_score_tree))
            self.tableWidget.setItem(row, column + 4, QTableWidgetItem(self.auc_score_tree))
            # row += 1
        except:
            pass
        row += 1
        try:
            self.tableWidget.setItem(row, column, QTableWidgetItem(self.accuracy_score_network))
            self.tableWidget.setItem(row, column + 1, QTableWidgetItem(self.precision_score_network))
            self.tableWidget.setItem(row, column + 2, QTableWidgetItem(self.recall_score_network))
            self.tableWidget.setItem(row, column + 3, QTableWidgetItem(self.f1_score_network))
            self.tableWidget.item(row, column).setForeground(QtGui.QColor(0, 160, 0))
            self.tableWidget.setItem(row, column + 4, QTableWidgetItem(self.auc_score_network))
            # row += 1
        except:
            pass
        row += 1
        try:
            self.tableWidget.setItem(row, column, QTableWidgetItem(self.accuracy_score_agregation_mean))
            self.tableWidget.setItem(row, column + 1, QTableWidgetItem(self.precision_score_agregation_mean))
            self.tableWidget.setItem(row, column + 2, QTableWidgetItem(self.recall_score_agregation_mean))
            self.tableWidget.setItem(row, column + 3, QTableWidgetItem(self.f1_score_agregation_mean))
            self.tableWidget.setItem(row, column + 4, QTableWidgetItem(self.auc_score_agregation_mean))
        except:
            pass
        row += 1
        try:
            self.tableWidget.setItem(row, column, QTableWidgetItem(self.accuracy_score_agregation_median))
            self.tableWidget.setItem(row, column + 1, QTableWidgetItem(self.precision_score_agregation_median))
            self.tableWidget.setItem(row, column + 2, QTableWidgetItem(self.recall_score_agregation_median))
            self.tableWidget.setItem(row, column + 3, QTableWidgetItem(self.f1_score_agregation_median))
            self.tableWidget.setItem(row, column + 4, QTableWidgetItem(self.auc_score_agregation_median))
        except:
            pass
        row += 1
        try:
            self.tableWidget.setItem(row, column, QTableWidgetItem(self.accuracy_score_agregation_voting))
            self.tableWidget.setItem(row, column + 1, QTableWidgetItem(self.precision_score_agregation_voting))
            self.tableWidget.setItem(row, column + 2, QTableWidgetItem(self.recall_score_agregation_voting))
            self.tableWidget.setItem(row, column + 3, QTableWidgetItem(self.f1_score_agregation_voting))
            self.tableWidget.setItem(row, column + 4, QTableWidgetItem(self.auc_score_agregation_voting))
        except:
            pass
        row += 1

    def significance(self):
        if self.checkBox_9.isChecked():
            self.window2.move(1222, 150)
            self.window2.show()
            self.tstat()
        else:
            self.window2.hide()
            self.data = self.old_data
            self.Y = self.data[self.data.columns[-1]].astype('int')
            self.X = self.data.drop(self.data.columns[-1], axis=1)
            #print(self.X)
            self.logistic_pred_flag = False
            self.clf_pred_flag = False
            self.disc_pred_flag = False
            self.support_pred_flag = False
            self.tree_pred_flag = False
            self.neural_pred_flag = False
            self.new_data_flag = False
            self.crossval_count = int(self.lineEdit_2.text())
            self.X_train, self.X_valid, self.Y_train, self.Y_valid = train_test_split(self.X, self.Y, test_size=self.percent)
            self.X_train_normalized = preprocessing.normalize(self.X_train)
            self.X_valid_normalized = preprocessing.normalize(self.X_valid)
            self.Ycsv = self.Y_valid.to_frame()
            self.Ycsv.set_axis(['valid'], axis=1, inplace=True)
            self.Ycsv.to_excel('prediction.xls')
            self.cleaning()


    def browse_new_data(self):
        try:
            self.label_7.setText("")
            fname = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', 'data')[0]
            self.data = pd.read_csv(fname)
            self.old_data = self.data
            self.Y = self.data[self.data.columns[-1]].astype('int')
            self.X = self.data.drop(self.data.columns[-1], axis=1)
            self.new_data_flag = True
            self.Ycsv = self.Y_valid.to_frame()
            self.Ycsv.set_axis(['valid'], axis=1, inplace=True)
            self.Ycsv.to_excel('prediction.xls')
        except:
            self.label_7.setText("Введены неверные параметры!")

    """def repredict(self):

        if self.window.radioButton.isChecked():
            self.bayes()

        if self.radioButton_2.isChecked():
            self.logistic_regression()

        if self.radioButton_3.isChecked():
            self.svm_vectors()

        if self.radioButton_4.isChecked():
            self.discriminant_analysis()

        if self.radioButton_5.isChecked():
            self.tree()

        if self.radioButton_6.isChecked():
            self.neural_network()

        if self.radioButton_7.isChecked():
            self.bagging()"""

    def browse_folder(self):
        try:
            self.label_2.show()
            fname = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', 'data', 'Тип файлов (*csv)')[0]
            name = os.path.basename(fname)
            self.percent = float(self.lineEdit.text()) / 100
            index = name.index('.')
            self.label_2.setText("Выбранный файл:\n " + str(name[:index]))
            self.data = pd.read_csv(fname)
            self.old_data = self.data
            self.Y = self.data[self.data.columns[-1]].astype('int')
            self.X = self.data.drop(self.data.columns[-1], axis=1)
            #print(self.X)
            self.logistic_pred_flag = False
            self.clf_pred_flag = False
            self.disc_pred_flag = False
            self.support_pred_flag = False
            self.tree_pred_flag = False
            self.neural_pred_flag = False
            self.new_data_flag = False
            self.crossval_count = int(self.lineEdit_2.text())
            self.X_train, self.X_valid, self.Y_train, self.Y_valid = train_test_split(self.X, self.Y, test_size=self.percent)
            self.X_train_normalized = preprocessing.normalize(self.X_train)
            self.X_valid_normalized = preprocessing.normalize(self.X_valid)
            self.X_train_normalize = preprocessing.normalize(self.X_train)
            self.X_train_scale = preprocessing.scale(self.X_train)
            self.X_train_normalize_scale = preprocessing.scale(self.X_train_normalize)
            self.X_train_scale_normalize = preprocessing.normalize(self.X_train_scale)
            self.X_valid_normalize = preprocessing.normalize(self.X_valid)
            self.X_valid_scale = preprocessing.scale(self.X_valid)
            self.X_valid_normalize_scale = preprocessing.scale(self.X_valid_normalize)
            self.X_valid_scale_normalize = preprocessing.normalize(self.X_valid_scale)
            self.Ycsv = self.Y_valid.to_frame()
            self.Ycsv.set_axis(['valid'], axis=1, inplace=True)
            self.Ycsv.to_excel('prediction.xls')
            self.tableWidget.clear()
            self.tableWidget.hide()
            self.cleaning()
        except:
            self.label_2.show()
            self.label_2.setText("Вы не открыли файл")

    def checkCross(self):
        if self.checkBox.isChecked():
            self.lineEdit_2.show()
        else:
            self.lineEdit_2.hide()

    def agregation(self):
        self.cleaning()
        count = 0
        if self.checkBox_7.isChecked():
            count += 1

        if self.checkBox_8.isChecked():
            count += 1

        if self.checkBox_5.isChecked():
            count += 1

        if self.checkBox_6.isChecked():
            count += 1

        if self.checkBox_3.isChecked():
            count += 1

        if self.checkBox_4.isChecked():
            count += 1
        x = []
        y = []
        pred = []
        for i in range(count):
            x.append(self.X_train[
                     i * (len(self.X_train) // count):len(self.X_train) * (i + 1) // count])
            y.append(self.Y_train[i * (len(self.Y_train) // count):len(self.Y_train) * (i + 1) // count])
            if self.label_2.text() != "Вы не открыли файл" and self.label_2.text() != " ":
                if self.lineEdit.text().isdigit():
                    if 1 <= int(self.lineEdit.text()) <= 100:
                        self.tableWidget.show()
                        self.label_5.show()
                        self.label_4.hide()
                        self.pushButton_2.show()
                        if self.checkBox_7.isChecked():
                            self.bayes(x[i], y[i], self.X_valid, self.Y_valid)
                            pred.append(self.predprob_bayes)
                            self.checkBox_7.setChecked(False)
                            continue
                        if self.checkBox_8.isChecked():
                            self.logistic_regression(x[i], y[i], self.X_valid, self.Y_valid)
                            pred.append(self.predprob_logistic)
                            self.checkBox_8.setChecked(False)
                            continue
                        if self.checkBox_5.isChecked():
                            self.svm_vectors(x[i], y[i], self.X_valid, self.Y_valid)
                            pred.append(self.predprob_svm)
                            self.checkBox_5.setChecked(False)
                            continue
                        if self.checkBox_6.isChecked():
                            self.discriminant_analysis(x[i], y[i], self.X_valid, self.Y_valid)
                            pred.append(self.predprob_discriminant)
                            self.checkBox_6.setChecked(False)
                            continue
                        if self.checkBox_3.isChecked():
                            self.tree(x[i], y[i], self.X_valid, self.Y_valid)
                            pred.append(self.predprob_tree)
                            self.checkBox_3.setChecked(False)
                            continue
                        if self.checkBox_4.isChecked():
                            self.neural_network(x[i], y[i], self.X_valid, self.Y_valid)
                            pred.append(self.predprob_network)
                            self.checkBox_4.setChecked(False)
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
        self.accuracy_score_agregation_mean = str(
            round(np.around(accuracy_score(self.Y_valid, meann),
                            decimals=4) * 100, 5)) + "%"
        self.precision_score_agregation_mean = str(round(
            np.around(precision_score(self.Y_valid, meann, zero_division=0),
                      decimals=4), 5))
        self.recall_score_agregation_mean = str(round(
            np.around(recall_score(self.Y_valid, meann, zero_division=0), decimals=4), 5))
        self.f1_score_agregation_mean = str(round(
            np.around(f1_score(self.Y_valid, meann, zero_division=0), decimals=4), 5))
        self.auc_score_agregation_mean = str(round(roc_auc_score(self.Y_valid, meann), 5))
        mediann = [round(num) for num in median]
        self.accuracy_score_agregation_median = str(
            round(np.around(accuracy_score(self.Y_valid, mediann),
                            decimals=4) * 100, 5)) + "%"
        self.precision_score_agregation_median = str(round(
            np.around(precision_score(self.Y_valid, mediann, zero_division=0),
                      decimals=4), 5))
        self.recall_score_agregation_median = str(round(
            np.around(recall_score(self.Y_valid, mediann, zero_division=0), decimals=4), 5))
        self.f1_score_agregation_median = str(round(
            np.around(f1_score(self.Y_valid, mediann, zero_division=0), decimals=4), 5))
        votingg = [round(num) for num in voting]
        self.auc_score_agregation_voting = str(round(roc_auc_score(self.Y_valid, votingg), 5))
        self.accuracy_score_agregation_voting = str(
            round(np.around(accuracy_score(self.Y_valid, votingg),
                            decimals=4) * 100, 5)) + "%"
        self.precision_score_agregation_voting = str(round(
            np.around(precision_score(self.Y_valid, votingg, zero_division=0),
                      decimals=4), 5))
        self.recall_score_agregation_voting = str(
            round(np.around(recall_score(self.Y_valid, votingg, zero_division=0), decimals=4), 5))
        self.f1_score_agregation_voting = str(round(
            np.around(f1_score(self.Y_valid, votingg, zero_division=0), decimals=4), 5))
        self.auc_score_agregation_voting = str(round(roc_auc_score(self.Y_valid, votingg), 5))
        self.table()

    def logistic_regression(self, x, y, x_val, y_val, x_val_norm, ):
        if (self.new_data_flag == True) and (self.logistic_pred_flag == True):
            self.logistic_pred = self.logistic.predict(self.X)
        else:
            # self.cleaning()
            if self.checkBox_2.isChecked():
                logistic = BaggingClassifier(SGDClassifier(loss='log'))
            elif self.checkBox.isChecked():
                logistic = GridSearchCV(SGDClassifier(loss='logn'), {'max_iter': range(1, 1000)},
                                        cv=self.crossval_count)
            elif (self.checkBox.isChecked() == True) and (self.checkBox_2.isChecked() == True):
                logistic = GridSearchCV(BaggingClassifier(SGDClassifier(loss='log')), {'max_iter': range(1, 1000)},
                                        cv=self.crossval_count)
            else:
                logistic = SGDClassifier(loss='log')
            logistic.fit(x, y)
            self.predprob_logistic = logistic.predict_proba(x_val)
            logistic_pred = logistic.predict(x_val)
            self.logistic_pred_flag = True
            self.accuracy_score_logistic = str(round(np.around(accuracy_score(y_val, logistic_pred),
                                                               decimals=4) * 100, 5)) + "%"
            self.precision_score_logistic = str(round(
                np.around(precision_score(y_val, logistic_pred, zero_division=0),
                          decimals=4), 5))
            self.recall_score_logistic = str(round(
                np.around(recall_score(y_val, logistic_pred, zero_division=0), decimals=4), 5))
            self.f1_score_logistic = str(round(
                np.around(f1_score(y_val, logistic_pred, zero_division=0), decimals=4), 5))
            self.auc_score_logistic = str(round(roc_auc_score(y_val, logistic_pred), 5))
            self.Ycsv['logistic'] = logistic_pred
            self.Ycsv.to_excel('prediction.xls')
            self.table()

    def bayes(self, x, y, x_val, y_val):
        if (self.new_data_flag == True) and (self.clf_pred_flag == True):
            self.clf_pred = self.clf.predict(self.X)
        else:
            # self.cleaning()
            if self.checkBox_2.isChecked():
                clf = BaggingClassifier(MultinomialNB())
            elif self.checkBox.isChecked():
                clf = GridSearchCV(MultinomialNB(), {'fit_prior': range(0, 1)}, cv=self.crossval_count)
            elif (self.checkBox.isChecked() == True) and (self.checkBox_2.isChecked() == True):
                clf = GridSearchCV(BaggingClassifier(MultinomialNB()), {'fit_prior': range(0, 1)},
                                   cv=self.crossval_count)
            else:
                clf = MultinomialNB()
            clf.fit(x, y)
            clf_pred = clf.predict(x_val)
            self.predprob_bayes = clf.predict_proba(x_val)
            self.clf_pred_flag = True
            self.accuracy_score_bayes = str(
                round(np.around(accuracy_score(y_val, clf_pred),
                                decimals=4) * 100, 5)) + "%"
            self.precision_score_bayes = str(round(
                np.around(precision_score(y_val, clf_pred, zero_division=0),
                          decimals=4), 5))
            self.recall_score_bayes = str(round(
                np.around(recall_score(y_val, clf_pred, zero_division=0), decimals=4), 5))
            self.f1_score_bayes = str(round(
                np.around(f1_score(y_val, clf_pred, zero_division=0), decimals=4), 5))
            self.auc_score_bayes = str(round(roc_auc_score(y_val, clf_pred), 5))
            self.Ycsv['bayes'] = clf_pred
            self.Ycsv.to_excel('prediction.xls')
            self.table()

    def discriminant_analysis(self, x, y, x_val, y_val):
        if (self.new_data_flag == True) and (self.disc_pred_flag == True):
            self.disc_pred = self.disc.predict(self.X)
        else:
            if self.checkBox_2.isChecked():
                disc = BaggingClassifier(LinearDiscriminantAnalysis())
            elif self.checkBox.isChecked():
                disc = GridSearchCV(LinearDiscriminantAnalysis(), {'n_components': range(0, 500)},
                                    cv=self.crossval_count)
            elif (self.checkBox.isChecked() == True) and (self.checkBox_2.isChecked() == True):
                disc = GridSearchCV(BaggingClassifier(LinearDiscriminantAnalysis()), {'n_components': range(0, 500)},
                                    cv=self.crossval_count)
            else:
                disc = LinearDiscriminantAnalysis()
            disc.fit(x, y)
            disc_pred = disc.predict(x_val)
            predprob_discriminant = disc.predict_proba(x_val)
            list, listt = [], []
            #for i in predprob_discriminant:
            #    list.append(i[1])
            #    print(i[1])
            #for i in y_val:
            #    listt.append(i)
            #    print(i)
            #print(listt)

            #plt.ylim(0, 1)
            #plt.bar(height=listt, x=range(len(x_val)), color="gray")
            #plt.plot(list, color="red")
            #plt.show()
            self.disc_pred_flag = True
            disc_score = disc.decision_function(x_val)
            self.accuracy_score_discriminant = str(round(
                np.around(accuracy_score(y_val, disc_pred),
                          decimals=4) * 100, 5)) + "%"
            self.precision_score_discriminant = str(round(
                np.around(precision_score(y_val, disc_pred, zero_division=0),
                          decimals=4), 5))
            self.recall_score_discriminant = str(round(
                np.around(recall_score(y_val, disc_pred, zero_division=0), decimals=4), 5))
            self.f1_score_discriminant = str(round(
                np.around(f1_score(y_val, disc_pred, zero_division=0), decimals=4), 5))
            self.auc_score_discriminant = str(round(roc_auc_score(y_val, disc_score), 5))
            self.Ycsv['disc'] = disc_pred
            self.Ycsv.to_excel('prediction.xls')
            self.table()

    def svm_vectors(self, x, y, x_val, y_val):
        if (self.new_data_flag == True) and (self.support_pred_flag == True):
            support_pred = self.support.predict(self.X)
            #print(support_pred)
        else:
            if self.checkBox_2.isChecked():
                support = BaggingClassifier(SVC(probability=True))
            elif self.checkBox.isChecked():
                support = GridSearchCV(SVC(probability=True), {'max_iter': range(-1, 1000)}, cv=self.crossval_count)
            elif (self.checkBox.isChecked() == True) and (self.checkBox_2.isChecked() == True):
                support = GridSearchCV(BaggingClassifier(SVC(probability=True)), {'max_iter': range(-1, 1000)},
                                       cv=self.crossval_count)
            else:
                support = SVC(probability=True, kernel="linear", C=0.025)
            support.fit(x, y)
            support_pred = support.predict(x_val)
            self.predprob_svm = support.predict_proba(x_val)
            self.support_pred_flag = True
            support_score = support.decision_function(x_val)
            self.accuracy_score_svm = str(
                round(np.around(accuracy_score(y_val, support_pred),
                                decimals=4) * 100, 5)) + "%"
            self.precision_score_svm = str(round(
                np.around(precision_score(y_val, support_pred, zero_division=0),
                          decimals=4), 5))
            self.recall_score_svm = str(round(
                np.around(recall_score(y_val, support_pred, zero_division=0), decimals=4), 5))
            self.f1_score_svm = str(round(
                np.around(f1_score(y_val, support_pred, zero_division=0), decimals=4), 5))
            self.auc_score_svm = str(round(roc_auc_score(y_val, support_score), 5))
            self.Ycsv['vectors'] = support_pred
            self.Ycsv.to_excel('prediction.xls')
            self.table()

    def tree(self, x, y, x_val, y_val):
        if (self.new_data_flag == True) and (self.tree_pred_flag == True):
            self.tree_pred = self.tree.predict(self.X)
        else:
            if self.checkBox_2.isChecked():
                tree = BaggingClassifier(DecisionTreeClassifier())
            elif self.checkBox.isChecked():
                tree = GridSearchCV(DecisionTreeClassifier(), {'max_depth': range(1, 100)}, cv=self.crossval_count)
            elif (self.checkBox.isChecked() == True) and (self.checkBox_2.isChecked() == True):
                tree = GridSearchCV(BaggingClassifier(DecisionTreeClassifier()), {'max_depth': range(1, 100)},
                                    cv=self.crossval_count)
            else:
                tree = DecisionTreeClassifier()
            tree.fit(x, y)
            tree_pred = tree.predict(x_val)
            self.predprob_tree = tree.predict_proba(x_val)
            self.tree_pred_flag = True
            self.accuracy_score_tree = str(round(np.around(accuracy_score(y_val, tree_pred),
                                                           decimals=4) * 100, 5)) + "%"
            self.precision_score_tree = str(round(
                np.around(precision_score(y_val, tree_pred, zero_division=0),
                          decimals=4), 5))
            self.recall_score_tree = str(round(
                np.around(recall_score(y_val, tree_pred, zero_division=0), decimals=4), 5))
            self.f1_score_tree = str(round(
                np.around(f1_score(y_val, tree_pred, zero_division=0), decimals=4), 5))
            auc_score_tree = str(round(roc_auc_score(y_val, tree_pred), 5))
            self.Ycsv['tree'] = tree_pred
            self.Ycsv.to_excel('prediction.xls')
            self.table()

    def neural_network(self, x, y, x_val, y_val):
        if (self.new_data_flag == True) and (self.neural_pred_flag == True):
            self.neural_pred = self.neural.predict(self.X)
        else:
            # self.cleaning()
            if self.checkBox_2.isChecked():
                neural = BaggingClassifier(MLPClassifier())
            elif self.checkBox.isChecked():
                neural = GridSearchCV(MLPClassifier(), {'max_iter': range(175, 225)}, cv=self.crossval_count)
            elif (self.checkBox.isChecked() == True) and (self.checkBox_2.isChecked() == True):
                neural = GridSearchCV(BaggingClassifier(MLPClassifier()), {'max_iter': range(175, 225)},
                                      cv=self.crossval_count)
            else:
                neural = MLPClassifier()
            neural.fit(x, y)
            neural_pred = neural.predict(x_val)
            self.predprob_network = neural.predict_proba(x_val)
            self.neural_pred_flag = True
            self.accuracy_score_network = str(round(np.around(accuracy_score(y_val, neural_pred),
                                                         decimals=4) * 100, 5)) + "%"
            self.precision_score_network = str(round(
                np.around(precision_score(y_val, neural_pred, zero_division=0), decimals=4), 5))
            self.recall_score_network = str(round(
                np.around(recall_score(y_val, neural_pred, zero_division=0), decimals=4), 5))
            self.f1_score_network = str(round(
                np.around(f1_score(y_val, neural_pred, zero_division=0), decimals=4), 5))
            self.auc_score_network = str(round(roc_auc_score(y_val, neural_pred), 5))
            self.Ycsv['neural_network'] = neural_pred
            self.Ycsv.to_excel('prediction.xls')
            self.table()

    def radio_choise(self):
        self.cleaning()
        if self.label_2.text() != "Вы не открыли файл" and self.label_2.text() != " ":
            if self.lineEdit.text().isdigit():
                if 1 <= int(self.lineEdit.text()) <= 100:
                    self.tableWidget.show()
                    self.label_5.show()
                    self.label_4.hide()
                    self.pushButton_2.show()
                    if self.checkBox_7.isChecked():
                        self.bayes(self.X_train, self.Y_train, self.X_valid, self.Y_valid)

                    if self.checkBox_8.isChecked():
                        self.logistic_regression(self.X_train, self.Y_train, self.X_valid,
                                                 self.Y_valid)

                    if self.checkBox_5.isChecked():
                        self.svm_vectors(self.X_train, self.Y_train, self.X_valid, self.Y_valid)

                    if self.checkBox_6.isChecked():
                        self.discriminant_analysis(self.X_train, self.Y_train, self.X_valid,
                                                   self.Y_valid)

                    if self.checkBox_3.isChecked():
                        self.tree(self.X_train, self.Y_train, self.X_valid, self.Y_valid)

                    if self.checkBox_4.isChecked():
                        self.neural_network(self.X_train, self.Y_train, self.X_valid,
                                            self.Y_valid)

                else:
                    self.tableWidget.hide()
                    self.tableWidget.clear()
                    self.label_5.hide()
                    self.label_4.show()
                    self.pushButton_2.hide()
                    self.label_4.setText("Введите от 1 до 100!")
            else:
                self.label_5.hide()
                self.label_4.show()
                self.pushButton_2.hide()
                self.label_4.setText("Введите от 1 до 100!")

    def output_analys(self):
        wb = xlwt.Workbook()
        ws = wb.add_sheet('Output', cell_overwrite_ok=True)
        # ws = wb.add_sheet("Output", cell_overwrite_ok=True)
        ws.write(1, 0, "Доля верной классификации при кроссвалидации")
        ws.write(1, 0, "Доля верной классификации")
        ws.write(2, 0, "Точность")
        ws.write(3, 0, "Полнота")
        ws.write(4, 0, "F-мера")
        ws.write(5, 0, "Критерий AUC")

        ws.write(0, 1, "Логистическая регрессия")
        ws.write(0, 2, "Байесовский классификатор")
        ws.write(0, 3, "Дискриминантный анализ")
        ws.write(0, 4, "Опорные вектора")
        ws.write(0, 5, "Деревья решений")
        ws.write(0, 6, "Нейронная сеть")
        ws.write(0, 7, "Агрегирование по среднему")
        ws.write(0, 8, "Агрегирование по медиане")
        ws.write(0, 9, "Агрегирование по голосованию")

        try:
            ws.write(1, 1, self.accuracy_score_logistic)
            ws.write(2, 1, self.precision_score_logistic)
            ws.write(3, 1, self.recall_score_logistic)
            ws.write(4, 1, self.f1_score_logistic)
            ws.write(5, 1, self.auc_score_logistic)
        except:
            pass
        try:
            ws.write(1, 2, self.accuracy_score_bayes)
            ws.write(2, 2, self.precision_score_bayes)
            ws.write(3, 2, self.recall_score_bayes)
            ws.write(4, 2, self.f1_score_bayes)
            ws.write(5, 2, self.auc_score_bayes)
        except:
            pass
        try:
            ws.write(1, 3, self.accuracy_score_discriminant)
            ws.write(2, 3, self.precision_score_discriminant)
            ws.write(3, 3, self.recall_score_discriminant)
            ws.write(4, 3, self.f1_score_discriminant)
            ws.write(5, 3, self.auc_score_discriminant)
        except:
            pass
        try:
            ws.write(1, 4, self.accuracy_score_svm)
            ws.write(2, 4, self.precision_score_svm)
            ws.write(3, 4, self.recall_score_svm)
            ws.write(4, 4, self.f1_score_svm)
            ws.write(5, 4, self.auc_score_svm)
        except:
            pass
        try:
            ws.write(1, 5, self.accuracy_score_tree)
            ws.write(2, 5, self.precision_score_tree)
            ws.write(3, 5, self.recall_score_tree)
            ws.write(4, 5, self.f1_score_tree)
            ws.write(5, 5, self.auc_score_tree)
        except:
            pass
        try:
            ws.write(1, 6, self.accuracy_score_network)
            ws.write(2, 6, self.precision_score_network)
            ws.write(3, 6, self.recall_score_network)
            ws.write(4, 6, self.f1_score_network)
            ws.write(5, 6, self.auc_score_network)
        except:
            pass
        try:
            ws.write(1, 7, self.accuracy_score_agregation_mean)
            ws.write(2, 7, self.precision_score_agregation_mean)
            ws.write(3, 7, self.recall_score_agregation_mean)
            ws.write(4, 7, self.f1_score_agregation_mean)
            ws.write(5, 7, self.auc_score_agregation_mean)
        except:
            pass
        try:
            ws.write(1, 8, self.accuracy_score_agregation_median)
            ws.write(2, 8, self.precision_score_agregation_median)
            ws.write(3, 8, self.recall_score_agregation_median)
            ws.write(4, 8, self.f1_score_agregation_median)
            ws.write(5, 8, self.auc_score_agregation_median)
        except:
            pass
        try:
            ws.write(1, 9, self.accuracy_score_agregation_voting)
            ws.write(2, 9, self.precision_score_agregation_voting)
            ws.write(3, 9, self.recall_score_agregation_voting)
            ws.write(4, 9, self.f1_score_agregation_voting)
            ws.write(5, 9, self.auc_score_agregation_voting)
        except:
            pass
        wb.save('output.xls')

class Window_Two(QtWidgets.QMainWindow, design.Ui_SecondWindow):

    def __init__(self):
        # Это здесь нужно для доступа к переменным, методам
        # и т.д. в файле design.py
        super().__init__()
        self.setupUi(self)

def main():
    app = QtWidgets.QApplication(sys.argv)  # Новый экземпляр QApplication
    window = Window()  # Создаём объект класса Window
    window.show()  # Показываем окно
    app.exec_()  # и запускаем приложение


if __name__ == '__main__':  # Если мы запускаем файл напрямую, а не импортируем
    main()  # то запускаем функцию main()
