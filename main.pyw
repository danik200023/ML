import sys
import os
import xlwt
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QTableWidgetItem
import design
import numpy as np
import pandas as pd
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


class ExampleApp(QtWidgets.QMainWindow, design.Ui_MainWindow):
    def __init__(self):
        # Это здесь нужно для доступа к переменным, методам
        # и т.д. в файле design.py
        super().__init__()

        self.setupUi(self)  # Это нужно для инициализации нашего дизайна
        self.pushButton.clicked.connect(self.browse_folder)
        self.pushButton_4.clicked.connect(self.radio_choise)
        self.pushButton_2.clicked.connect(self.output_analys)
        self.pushButton_3.clicked.connect(self.browse_new_data)
        self.pushButton_5.clicked.connect(self.repredict)
        self.pushButton_6.clicked.connect(self.agregation)
        self.checkBox.clicked.connect(self.checkCross)
        self.checkBox.clicked.connect(self.table)
        self.label_2.hide()
        self.label_5.hide()
        self.tableWidget.hide()
        self.pushButton_2.hide()
        self.lineEdit_2.hide()
        #self.radioButton_2.setChecked(True)

    def cleaning(self):
        try:
            del globals()['accuracy_score_logistic']
            del globals()['precision_score_logistic']
            del globals()['recall_score_logistic']
            del globals()['f1_score_logistic']
            del globals()['auc_score_logistic']
        except:
            pass
        try:
            del globals()['accuracy_score_bayes']
            del globals()['precision_score_bayes']
            del globals()['recall_score_bayes']
            del globals()['f1_score_bayes']
            del globals()['auc_score_bayes']
        except:
            pass
        try:
            del globals()['accuracy_score_discriminant']
            del globals()['precision_score_discriminant']
            del globals()['recall_score_discriminant']
            del globals()['f1_score_discriminant']
            del globals()['auc_score_discriminant']
        except:
            pass
        try:
            del globals()['accuracy_score_svm']
            del globals()['precision_score_svm']
            del globals()['recall_score_svm']
            del globals()['f1_score_svm']
            del globals()['auc_score_svm']
        except:
            pass
        try:
            del globals()['accuracy_score_tree']
            del globals()['precision_score_tree']
            del globals()['recall_score_tree']
            del globals()['f1_score_tree']
            del globals()['auc_score_tree']
        except:
            pass
        try:
            del globals()['accuracy_score_network']
            del globals()['precision_score_network']
            del globals()['recall_score_network']
            del globals()['f1_score_network']
            del globals()['auc_score_network']
        except:
            pass
        try:
            del globals()['accuracy_score_agregation_mean']
            del globals()['precision_score_agregation_mean']
            del globals()['recall_score_agregation_mean']
            del globals()['f1_score_agregation_mean']
            del globals()['auc_score_agregation_mean']
        except:
            pass
        try:
            del globals()['accuracy_score_agregation_median']
            del globals()['precision_score_agregation_median']
            del globals()['recall_score_agregation_median']
            del globals()['f1_score_agregation_median']
            del globals()['auc_score_agregation_median']
        except:
            pass
        try:
            del globals()['accuracy_score_agregation_voting']
            del globals()['precision_score_agregation_voting']
            del globals()['recall_score_agregation_voting']
            del globals()['f1_score_agregation_voting']
            del globals()['auc_score_agregation_voting']
        except:
            pass

    def table(self):
        self.tableWidget.setColumnCount(5)
        self.tableWidget.setRowCount(9)
        self.tableWidget.setHorizontalHeaderLabels(
            ["Доля верной классификации", "Точность", "Полнота", "F-мера", "AUC"])
        self.tableWidget.setVerticalHeaderLabels(
            ["Логистическая регрессия", "Байесовский классификатор", "Дискриминантный анализ", "Опорные вектора",
             "Деревья решений", "Нейронная сеть", "Агрегирование по среднему", "Агрегирование по медиане", "Агрегирование по голосованию"])
        column = 0  # столбец
        row = 0  # строка

        try:
            self.tableWidget.setItem(row, column, QTableWidgetItem(accuracy_score_logistic))
            self.tableWidget.setItem(row, column + 1, QTableWidgetItem(precision_score_logistic))
            self.tableWidget.setItem(row, column + 2, QTableWidgetItem(recall_score_logistic))
            self.tableWidget.setItem(row, column + 3, QTableWidgetItem(f1_score_logistic))
            self.tableWidget.setItem(row, column + 4, QTableWidgetItem(auc_score_logistic))
            # row += 1
        except:
            pass
        row += 1
        try:
            self.tableWidget.setItem(row, column, QTableWidgetItem(accuracy_score_bayes))
            self.tableWidget.setItem(row, column + 1, QTableWidgetItem(precision_score_bayes))
            self.tableWidget.setItem(row, column + 2, QTableWidgetItem(recall_score_bayes))
            self.tableWidget.setItem(row, column + 3, QTableWidgetItem(f1_score_bayes))
            self.tableWidget.setItem(row, column + 4, QTableWidgetItem(auc_score_bayes))
            # row += 1
        except:
            pass
        row += 1
        try:
            self.tableWidget.setItem(row, column, QTableWidgetItem(accuracy_score_discriminant))
            self.tableWidget.setItem(row, column + 1, QTableWidgetItem(precision_score_discriminant))
            self.tableWidget.setItem(row, column + 2, QTableWidgetItem(recall_score_discriminant))
            self.tableWidget.setItem(row, column + 3, QTableWidgetItem(f1_score_discriminant))
            self.tableWidget.setItem(row, column + 4, QTableWidgetItem(auc_score_discriminant))
            # row += 1
        except:
            pass
        row += 1
        try:
            self.tableWidget.setItem(row, column, QTableWidgetItem(accuracy_score_svm))
            self.tableWidget.setItem(row, column + 1, QTableWidgetItem(precision_score_svm))
            self.tableWidget.setItem(row, column + 2, QTableWidgetItem(recall_score_svm))
            self.tableWidget.setItem(row, column + 3, QTableWidgetItem(f1_score_svm))
            self.tableWidget.setItem(row, column + 4, QTableWidgetItem(auc_score_svm))
            # row += 1
        except:
            pass
        row += 1
        try:
            self.tableWidget.setItem(row, column, QTableWidgetItem(accuracy_score_tree))
            self.tableWidget.setItem(row, column + 1, QTableWidgetItem(precision_score_tree))
            self.tableWidget.setItem(row, column + 2, QTableWidgetItem(recall_score_tree))
            self.tableWidget.setItem(row, column + 3, QTableWidgetItem(f1_score_tree))
            self.tableWidget.setItem(row, column + 4, QTableWidgetItem(auc_score_tree))
            # row += 1
        except:
            pass
        row += 1
        try:
            self.tableWidget.setItem(row, column, QTableWidgetItem(accuracy_score_network))
            self.tableWidget.setItem(row, column + 1, QTableWidgetItem(precision_score_network))
            self.tableWidget.setItem(row, column + 2, QTableWidgetItem(recall_score_network))
            self.tableWidget.setItem(row, column + 3, QTableWidgetItem(f1_score_network))
            self.tableWidget.setItem(row, column + 4, QTableWidgetItem(auc_score_network))
            # row += 1
        except:
            pass
        row += 1
        try:
            self.tableWidget.setItem(row, column, QTableWidgetItem(accuracy_score_agregation_mean))
            self.tableWidget.setItem(row, column + 1, QTableWidgetItem(precision_score_agregation_mean))
            self.tableWidget.setItem(row, column + 2, QTableWidgetItem(recall_score_agregation_mean))
            self.tableWidget.setItem(row, column + 3, QTableWidgetItem(f1_score_agregation_mean))
            self.tableWidget.setItem(row, column + 4, QTableWidgetItem(auc_score_agregation_mean))
        except:
            pass
        row += 1
        try:
            self.tableWidget.setItem(row, column, QTableWidgetItem(accuracy_score_agregation_median))
            self.tableWidget.setItem(row, column + 1, QTableWidgetItem(precision_score_agregation_median))
            self.tableWidget.setItem(row, column + 2, QTableWidgetItem(recall_score_agregation_median))
            self.tableWidget.setItem(row, column + 3, QTableWidgetItem(f1_score_agregation_median))
            self.tableWidget.setItem(row, column + 4, QTableWidgetItem(auc_score_agregation_median))
        except:
            pass
        row += 1
        try:
            self.tableWidget.setItem(row, column, QTableWidgetItem(accuracy_score_agregation_voting))
            self.tableWidget.setItem(row, column + 1, QTableWidgetItem(precision_score_agregation_voting))
            self.tableWidget.setItem(row, column + 2, QTableWidgetItem(recall_score_agregation_voting))
            self.tableWidget.setItem(row, column + 3, QTableWidgetItem(f1_score_agregation_voting))
            self.tableWidget.setItem(row, column + 4, QTableWidgetItem(auc_score_agregation_voting))
        except:
            pass
        row += 1

    def browse_new_data(self):
        try:
            self.label_7.setText("")
            fname = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', 'data')[0]
            data = pd.read_csv(fname)
            global X, Y
            Y = data[data.columns[-1]].astype('int')
            X = data.drop(data.columns[-1], axis=1)
            global new_data_flag
            new_data_flag = True
            global Ycsv
            Ycsv = Y_valid.to_frame()
            Ycsv.set_axis(['valid'], axis=1, inplace=True)
            Ycsv.to_excel('prediction.xls')
        except:
            self.label_7.setText("Введены неверные параметры!")

    def repredict(self):
        self.cleaning()
        if self.radioButton.isChecked():
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
            self.bagging()

    def browse_folder(self):
        try:
            self.label_2.show()
            fname = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', 'data')[0]
            name = os.path.basename(fname)
            percent = float(self.lineEdit.text()) / 100
            index = name.index('.')
            self.label_2.setText("Выбранный файл:\n " + str(name[:index]))
            global data
            data = pd.read_csv(fname)
            Y = data[data.columns[-1]].astype('int')
            X = data.drop(data.columns[-1], axis=1)
            global X_train, X_valid, Y_train, Y_valid, X_train_normalized, X_valid_normalized, crossval_count, \
                logistic_pred_flag, clf_pred_flag, disc_pred_flag, support_pred_flag, tree_pred_flag, neural_pred_flag, \
                new_data_flag
            logistic_pred_flag = False
            clf_pred_flag = False
            disc_pred_flag = False
            support_pred_flag = False
            tree_pred_flag = False
            neural_pred_flag = False
            new_data_flag = False
            crossval_count = int(self.lineEdit_2.text())
            X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=percent)
            X_train_normalized = preprocessing.normalize(X_train)
            X_valid_normalized = preprocessing.normalize(X_valid)

            global Ycsv
            Ycsv = Y_valid.to_frame()
            Ycsv.set_axis(['valid'], axis=1, inplace=True)
            Ycsv.to_excel('prediction.xls')
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
            x.append(X_train_normalized[i * (len(X_train_normalized)//count):len(X_train_normalized)*(i+1)//count])
            y.append(Y_train[i * (len(Y_train) // count):len(Y_train) * (i + 1) // count])
            if self.label_2.text() != "Вы не открыли файл" and self.label_2.text() != " ":
                if self.lineEdit.text().isdigit():
                    if 1 <= int(self.lineEdit.text()) <= 100:
                        self.tableWidget.show()
                        self.label_5.show()
                        self.label_4.hide()
                        self.pushButton_2.show()
                        if self.checkBox_7.isChecked():
                            self.bayes(x[i], y[i], X_valid_normalized, Y_valid)
                            pred.append(predprob_bayes)
                            continue
                        if self.checkBox_8.isChecked():
                            self.logistic_regression(x[i], y[i], X_valid_normalized, Y_valid)
                            pred.append(predprob_logistic)
                            continue
                        if self.checkBox_5.isChecked():
                            self.svm_vectors(x[i], y[i], X_valid_normalized, Y_valid)
                            pred.append(predprob_svm)
                            continue
                        if self.checkBox_6.isChecked():
                            self.discriminant_analysis(x[i], y[i], X_valid_normalized, Y_valid)
                            pred.append(predprob_discriminant)
                            continue
                        if self.checkBox_3.isChecked():
                            self.tree(x[i], y[i], X_valid_normalized, Y_valid)
                            pred.append(predprob_tree)
                            continue
                        if self.checkBox_4.isChecked():
                            self.neural_network(x[i], y[i], X_valid_normalized, Y_valid)
                            pred.append(predprob_network)
                            continue

        predd = []
        mean = []
        median = []
        voting = []
        for i in range(pred[0].shape[0]):
            preddd = []
            for j in range(count):
                preddd.append(pred[j][i][1])
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
        global accuracy_score_agregation_mean, precision_score_agregation_mean, recall_score_agregation_mean, f1_score_agregation_mean, auc_score_agregation_mean, accuracy_score_agregation_median, precision_score_agregation_median, recall_score_agregation_median, f1_score_agregation_median, auc_score_agregation_median, accuracy_score_agregation_voting, precision_score_agregation_voting, recall_score_agregation_voting, f1_score_agregation_voting, auc_score_agregation_voting
        meann = [round(num) for num in mean]
        accuracy_score_agregation_mean = str(
            round(np.around(accuracy_score(Y_valid, meann),
                            decimals=4) * 100, 5)) + "%"
        precision_score_agregation_mean = str(round(
            np.around(precision_score(Y_valid, meann, zero_division=0),
                      decimals=4), 5))
        recall_score_agregation_mean = str(round(
            np.around(recall_score(Y_valid, meann, zero_division=0), decimals=4), 5))
        f1_score_agregation_mean = str(round(
            np.around(f1_score(Y_valid, meann, zero_division=0), decimals=4), 5))
        auc_score_agregation_mean = str(round(roc_auc_score(Y_valid, meann), 5))
        global accuracy_score_agregation_voting, precision_score_agregation_voting, recall_score_agregation_voting, f1_score_agregation_voting, auc_score_agregation_voting
        mediann = [round(num) for num in median]
        accuracy_score_agregation_median = str(
            round(np.around(accuracy_score(Y_valid, mediann),
                            decimals=4) * 100, 5)) + "%"
        precision_score_agregation_median = str(round(
            np.around(precision_score(Y_valid, mediann, zero_division=0),
                      decimals=4), 5))
        recall_score_agregation_median = str(round(
            np.around(recall_score(Y_valid, mediann, zero_division=0), decimals=4), 5))
        f1_score_agregation_median = str(round(
            np.around(f1_score(Y_valid, mediann, zero_division=0), decimals=4), 5))
        votingg = [round(num) for num in voting]
        auc_score_agregation_voting = str(round(roc_auc_score(Y_valid, votingg), 5))
        accuracy_score_agregation_voting = str(
            round(np.around(accuracy_score(Y_valid, votingg),
                            decimals=4) * 100, 5)) + "%"
        precision_score_agregation_voting = str(round(
            np.around(precision_score(Y_valid, votingg, zero_division=0),
                      decimals=4), 5))
        recall_score_agregation_voting = str(round(
            np.around(recall_score(Y_valid, votingg, zero_division=0), decimals=4), 5))
        f1_score_agregation_voting = str(round(
            np.around(f1_score(Y_valid, votingg, zero_division=0), decimals=4), 5))
        auc_score_agregation_voting = str(round(roc_auc_score(Y_valid, votingg), 5))
        self.table()


    def logistic_regression(self, x, y, x_val, y_val):
        global logistic, logistic_pred_flag, logistic_pred
        if (new_data_flag == True) and (logistic_pred_flag == True):
            logistic_pred = logistic.predict(X)
        else:
            global accuracy_score_logistic, precision_score_logistic, recall_score_logistic, f1_score_logistic, auc_score_logistic, predprob_logistic
            if self.checkBox_2.isChecked():
                logistic = BaggingClassifier(SGDClassifier(loss='log'))
            elif self.checkBox.isChecked():
                logistic = GridSearchCV(SGDClassifier(loss='log'), {'max_iter': range(1, 1000)}, cv=crossval_count)
            elif (self.checkBox.isChecked() == True) and (self.checkBox_2.isChecked() == True):
                logistic = GridSearchCV(BaggingClassifier(SGDClassifier(loss='log')), {'max_iter': range(1, 1000)},
                                        cv=crossval_count)
            else:
                logistic = SGDClassifier(loss='log')
            logistic.fit(x, y)
            predprob_logistic = logistic.predict_proba(x_val)
            logistic_pred = logistic.predict(x_val)
            logistic_pred_flag = True
            accuracy_score_logistic = str(round(np.around(accuracy_score(y_val, logistic_pred),
                                                          decimals=4) * 100, 5)) + "%"
            precision_score_logistic = str(round(
                np.around(precision_score(y_val, logistic_pred, zero_division=0),
                          decimals=4), 5))
            recall_score_logistic = str(round(
                np.around(recall_score(y_val, logistic_pred, zero_division=0), decimals=4), 5))
            f1_score_logistic = str(round(
                np.around(f1_score(y_val, logistic_pred, zero_division=0), decimals=4), 5))
            auc_score_logistic = str(round(roc_auc_score(y_val, logistic_pred), 5))
            Ycsv['logistic'] = logistic_pred
            Ycsv.to_excel('prediction.xls')
            self.table()

    def bayes(self, x, y, x_val, y_val):
        global clf, clf_pred_flag, clf_pred
        if (new_data_flag == True) and (clf_pred_flag == True):
            clf_pred = clf.predict(X)
        else:
            global accuracy_score_bayes, precision_score_bayes, recall_score_bayes, f1_score_bayes, auc_score_bayes, predprob_bayes
            if self.checkBox_2.isChecked():
                clf = BaggingClassifier(MultinomialNB())
            elif self.checkBox.isChecked():
                clf = GridSearchCV(MultinomialNB(), {'fit_prior': range(0, 1)}, cv=crossval_count)
            elif (self.checkBox.isChecked() == True) and (self.checkBox_2.isChecked() == True):
                clf = GridSearchCV(BaggingClassifier(MultinomialNB()), {'fit_prior': range(0, 1)}, cv=crossval_count)
            else:
                clf = MultinomialNB()
            clf.fit(x, y)
            clf_pred = clf.predict(x_val)
            predprob_bayes = clf.predict_proba(x_val)
            clf_pred_flag = True
            accuracy_score_bayes = str(
                round(np.around(accuracy_score(y_val, clf_pred),
                                decimals=4) * 100, 5)) + "%"
            precision_score_bayes = str(round(
                np.around(precision_score(y_val, clf_pred, zero_division=0),
                          decimals=4), 5))
            recall_score_bayes = str(round(
                np.around(recall_score(y_val, clf_pred, zero_division=0), decimals=4), 5))
            f1_score_bayes = str(round(
                np.around(f1_score(y_val, clf_pred, zero_division=0), decimals=4), 5))
            auc_score_bayes = str(round(roc_auc_score(y_val, clf_pred), 5))
            Ycsv['bayes'] = clf_pred
            Ycsv.to_excel('prediction.xls')
            self.table()

    def discriminant_analysis(self, x, y, x_val, y_val):
        global disc, disc_pred_flag, disc_pred
        if (new_data_flag == True) and (disc_pred_flag == True):
            disc_pred = disc.predict(X)
        else:
            global accuracy_score_discriminant, precision_score_discriminant, recall_score_discriminant, f1_score_discriminant, auc_score_discriminant, predprob_discriminant
            if self.checkBox_2.isChecked():
                disc = BaggingClassifier(LinearDiscriminantAnalysis())
            elif self.checkBox.isChecked():
                disc = GridSearchCV(LinearDiscriminantAnalysis(), {'n_components': range(0, 500)}, cv=crossval_count)
            elif (self.checkBox.isChecked() == True) and (self.checkBox_2.isChecked() == True):
                disc = GridSearchCV(BaggingClassifier(LinearDiscriminantAnalysis()), {'n_components': range(0, 500)}, cv=crossval_count)
            else:
                disc = LinearDiscriminantAnalysis()
            disc.fit(x, y)
            disc_pred = disc.predict(x_val)
            predprob_discriminant = disc.predict_proba(x_val)
            disc_pred_flag = True
            disc_score = disc.decision_function(x_val)
            accuracy_score_discriminant = str(round(
                np.around(accuracy_score(y_val, disc_pred),
                          decimals=4) * 100, 5)) + "%"
            precision_score_discriminant = str(round(
                np.around(precision_score(y_val, disc_pred, zero_division=0),
                          decimals=4), 5))
            recall_score_discriminant = str(round(
                np.around(recall_score(y_val, disc_pred, zero_division=0), decimals=4), 5))
            f1_score_discriminant = str(round(
                np.around(f1_score(y_val, disc_pred, zero_division=0), decimals=4), 5))
            auc_score_discriminant = str(round(roc_auc_score(y_val, disc_score), 5))
            # print(predprob_discriminant[][1])
            # Ycsv['disc'] = predprob_discriminant[:][1]
            Ycsv.to_excel('prediction.xls')
            self.table()

    def svm_vectors(self, x, y, x_val, y_val):
        global support, support_pred_flag, support_pred
        if (new_data_flag == True) and (support_pred_flag == True):
            support_pred = support.predict(X)
            print(support_pred)
        else:
            global accuracy_score_svm, precision_score_svm, recall_score_svm, f1_score_svm, auc_score_svm, predprob_svm
            if self.checkBox_2.isChecked():
                support = BaggingClassifier(SVC(probability=True))
            elif self.checkBox.isChecked():
                support = GridSearchCV(SVC(probability=True), {'max_iter': range(-1, 1000)}, cv=crossval_count)
            elif (self.checkBox.isChecked() == True) and (self.checkBox_2.isChecked() == True):
                support = GridSearchCV(BaggingClassifier(SVC(probability=True)), {'max_iter': range(-1, 1000)}, cv=crossval_count)
            else:
                support = SVC(probability=True)
            support.fit(x, y)
            support_pred = support.predict(x_val)
            predprob_svm = support.predict_proba(x_val)
            support_pred_flag = True
            support_score = support.decision_function(x_val)
            accuracy_score_svm = str(
                round(np.around(accuracy_score(y_val, support_pred),
                                decimals=4) * 100, 5)) + "%"
            precision_score_svm = str(round(
                np.around(precision_score(y_val, support_pred, zero_division=0),
                          decimals=4), 5))
            recall_score_svm = str(round(
                np.around(recall_score(y_val, support_pred, zero_division=0), decimals=4), 5))
            f1_score_svm = str(round(
            np.around(f1_score(y_val, support_pred, zero_division=0), decimals=4), 5))
            auc_score_svm = str(round(roc_auc_score(y_val, support_score), 5))
            Ycsv['vectors'] = support_pred
            Ycsv.to_excel('prediction.xls')
            self.table()

    def tree(self, x, y, x_val, y_val):
        global tree, tree_pred_flag, tree_pred
        if (new_data_flag == True) and (tree_pred_flag == True):
            tree_pred = tree.predict(X)
        else:
            global accuracy_score_tree, precision_score_tree, recall_score_tree, f1_score_tree, auc_score_tree, predprob_tree
            if self.checkBox_2.isChecked():
                tree = BaggingClassifier(DecisionTreeClassifier())
            elif self.checkBox.isChecked():
                tree = GridSearchCV(DecisionTreeClassifier(), {'max_depth': range(1, 100)}, cv=crossval_count)
            elif (self.checkBox.isChecked() == True) and (self.checkBox_2.isChecked() == True):
                tree = GridSearchCV(BaggingClassifier(DecisionTreeClassifier()), {'max_depth': range(1, 100)},
                                    cv=crossval_count)
            else:
                tree = DecisionTreeClassifier()
            tree.fit(x, y)
            tree_pred = tree.predict(x_val)
            predprob_tree = tree.predict_proba(x_val)
            tree_pred_flag = True
            accuracy_score_tree = str(round(np.around(accuracy_score(y_val, tree_pred),
                                                      decimals=4) * 100, 5)) + "%"
            precision_score_tree = str(round(
                np.around(precision_score(y_val, tree_pred, zero_division=0),
                          decimals=4), 5))
            recall_score_tree = str(round(
                np.around(recall_score(y_val, tree_pred, zero_division=0), decimals=4), 5))
            f1_score_tree = str(round(
                np.around(f1_score(y_val, tree_pred, zero_division=0), decimals=4), 5))
            auc_score_tree = str(round(roc_auc_score(y_val, tree_pred), 5))
            Ycsv['tree'] = tree_pred
            Ycsv.to_excel('prediction.xls')
            self.table()

    def neural_network(self, x, y, x_val, y_val):
        global neural, neural_pred_flag, neural_pred
        if (new_data_flag == True) and (neural_pred_flag == True):
            neural_pred = neural.predict(X)
        else:
            global accuracy_score_network, precision_score_network, recall_score_network, f1_score_network, auc_score_network, predprob_network
            if self.checkBox_2.isChecked():
                neural = BaggingClassifier(MLPClassifier())
            elif self.checkBox.isChecked():
                neural = GridSearchCV(MLPClassifier(), {'max_iter': range(175, 225)}, cv=crossval_count)
            elif (self.checkBox.isChecked() == True) and (self.checkBox_2.isChecked() == True):
                neural = GridSearchCV(BaggingClassifier(MLPClassifier()), {'max_iter': range(175, 225)}, cv=crossval_count)
            else:
                neural = MLPClassifier()
            neural.fit(x, y)
            neural_pred = neural.predict(x_val)
            predprob_network = neural.predict_proba(x_val)
            neural_pred_flag = True
            accuracy_score_network = str(round(np.around(accuracy_score(y_val, neural_pred),
                                                         decimals=4) * 100, 5)) + "%"
            precision_score_network = str(round(
                np.around(precision_score(y_val, neural_pred, zero_division=0), decimals=4), 5))
            recall_score_network = str(round(
                np.around(recall_score(y_val, neural_pred, zero_division=0), decimals=4), 5))
            f1_score_network = str(round(
                np.around(f1_score(y_val, neural_pred, zero_division=0), decimals=4), 5))
            auc_score_network = str(round(roc_auc_score(y_val, neural_pred), 5))
            Ycsv['neural_network'] = neural_pred
            Ycsv.to_excel('prediction.xls')
            print(str(round(np.around(accuracy_score(y_val, neural_pred),
                                      decimals=4) * 100, 5)) + "%")
            print(accuracy_score_network, precision_score_network, recall_score_network, f1_score_network,
                  auc_score_network)
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
                        self.bayes(X_train_normalized, Y_train, X_valid_normalized, Y_valid)

                    if self.checkBox_8.isChecked():
                        self.logistic_regression(X_train_normalized, Y_train, X_valid_normalized, Y_valid)

                    if self.checkBox_5.isChecked():
                        self.svm_vectors(X_train_normalized, Y_train, X_valid_normalized, Y_valid)

                    if self.checkBox_6.isChecked():
                        self.discriminant_analysis(X_train_normalized, Y_train, X_valid_normalized, Y_valid)

                    if self.checkBox_3.isChecked():
                        self.tree(X_train_normalized, Y_train, X_valid_normalized, Y_valid)

                    if self.checkBox_4.isChecked():
                        self.neural_network(X_train_normalized, Y_train, X_valid_normalized, Y_valid)

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
            ws.write(1, 1, accuracy_score_logistic)
            ws.write(2, 1, precision_score_logistic)
            ws.write(3, 1, recall_score_logistic)
            ws.write(4, 1, f1_score_logistic)
            ws.write(5, 1, auc_score_logistic)
        except:
            pass
        try:
            ws.write(1, 2, accuracy_score_bayes)
            ws.write(2, 2, precision_score_bayes)
            ws.write(3, 2, recall_score_bayes)
            ws.write(4, 2, f1_score_bayes)
            ws.write(5, 2, auc_score_bayes)
        except:
            pass
        try:
            ws.write(1, 3, accuracy_score_discriminant)
            ws.write(2, 3, precision_score_discriminant)
            ws.write(3, 3, recall_score_discriminant)
            ws.write(4, 3, f1_score_discriminant)
            ws.write(5, 3, auc_score_discriminant)
        except:
            pass
        try:
            ws.write(1, 4, accuracy_score_svm)
            ws.write(2, 4, precision_score_svm)
            ws.write(3, 4, recall_score_svm)
            ws.write(4, 4, f1_score_svm)
            ws.write(5, 4, auc_score_svm)
        except:
            pass
        try:
            ws.write(1, 5, accuracy_score_tree)
            ws.write(2, 5, precision_score_tree)
            ws.write(3, 5, recall_score_tree)
            ws.write(4, 5, f1_score_tree)
            ws.write(5, 5, auc_score_tree)
        except:
            pass
        try:
            ws.write(1, 6, accuracy_score_network)
            ws.write(2, 6, precision_score_network)
            ws.write(3, 6, recall_score_network)
            ws.write(4, 6, f1_score_network)
            ws.write(5, 6, auc_score_network)
        except:
            pass
        try:
            ws.write(1, 7, accuracy_score_agregation_mean)
            ws.write(2, 7, precision_score_agregation_mean)
            ws.write(3, 7, recall_score_agregation_mean)
            ws.write(4, 7, f1_score_agregation_mean)
            ws.write(5, 7, auc_score_agregation_mean)
        except:
            pass
        try:
            ws.write(1, 8, accuracy_score_agregation_median)
            ws.write(2, 8, precision_score_agregation_median)
            ws.write(3, 8, recall_score_agregation_median)
            ws.write(4, 8, f1_score_agregation_median)
            ws.write(5, 8, auc_score_agregation_median)
        except:
            pass
        try:
            ws.write(1, 9, accuracy_score_agregation_voting)
            ws.write(2, 9, precision_score_agregation_voting)
            ws.write(3, 9, recall_score_agregation_voting)
            ws.write(4, 9, f1_score_agregation_voting)
            ws.write(5, 9, auc_score_agregation_voting)
        except:
            pass
        wb.save('output.xls')


def main():
    app = QtWidgets.QApplication(sys.argv)  # Новый экземпляр QApplication
    window = ExampleApp()  # Создаём объект класса ExampleApp
    window.show()  # Показываем окно
    app.exec_()  # и запускаем приложение


if __name__ == '__main__':  # Если мы запускаем файл напрямую, а не импортируем
    main()  # то запускаем функцию main()
