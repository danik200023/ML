import sys
import os
import xlwt
from PyQt5 import QtWidgets
import design
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class ExampleApp(QtWidgets.QMainWindow, design.Ui_MainWindow):
    def __init__(self):
        # Это здесь нужно для доступа к переменным, методам
        # и т.д. в файле design.py
        super().__init__()

        self.setupUi(self)  # Это нужно для инициализации нашего дизайна
        self.pushButton.clicked.connect(self.browse_folder)
        self.pushButton_4.clicked.connect(self.radio_choise)
        self.pushButton_2.clicked.connect(self.output_analys)
        self.listWidget.hide()
        self.label_2.hide()
        self.label_5.hide()
        self.pushButton_2.hide()
        self.radioButton_2.setChecked(True)

    def browse_folder(self):
        try:
            self.label_2.show()
            fname = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', 'data')[0]
            global name
            name = os.path.basename(fname)
            global data
            percent = float(self.lineEdit.text()) / 100
            index = name.index('.')
            self.label_2.setText("Выбранный файл:\n " + str(name[:index]))
            data = pd.read_csv(fname)
            Y = data[data.columns[-1]].astype('int')
            X = data.drop(data.columns[-1], axis=1)
            global X_train, X_valid, Y_train, Y_valid
            X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=percent)
            global Ycsv
            Ycsv = Y_valid.to_frame()
            Ycsv.set_axis(['valid'], axis=1, inplace=True)
            Ycsv.to_excel('prediction.xls')
        except:
            self.label_2.show()
            self.label_2.setText("Вы не открыли файл")

    def logistic_regression(self):
        self.listWidget.clear()
        logistic = SGDClassifier()
        global cross_val_score_logistic, accuracy_score_logistic, precision_score_logistic, recall_score_logistic, f1_score_logistic
        cross_val_score_logistic = str(round(np.around(np.mean(cross_val_score(logistic,
                                                                               X_train,
                                                                               Y_train,
                                                                               cv=5)),
                                                       decimals=4) * 100, 5)) + "%"
        # self.listWidget.addItem("Доля верной классификации при кроссвалидации: " + cross_val_score_logistic)
        logistic.fit(X_train, Y_train)
        logistic_pred = logistic.predict(X_valid)
        accuracy_score_logistic = str(round(np.around(accuracy_score(Y_valid, logistic_pred),
                                                      decimals=4) * 100, 5)) + "%"
        self.listWidget.addItem("Доля верной классификации: " + accuracy_score_logistic)
        precision_score_logistic = str(round(
            np.around(precision_score(Y_valid, logistic_pred, zero_division=0),
                      decimals=4), 5))
        self.listWidget.addItem("Точность: " + precision_score_logistic)
        recall_score_logistic = str(round(
            np.around(recall_score(Y_valid, logistic_pred, zero_division=0), decimals=4), 5))
        self.listWidget.addItem("Полнота: " + recall_score_logistic)
        f1_score_logistic = str(round(
            np.around(f1_score(Y_valid, logistic_pred, zero_division=0), decimals=4), 5))
        self.listWidget.addItem("F-мера: " + f1_score_logistic)
        Ycsv['logistic'] = logistic_pred
        Ycsv.to_excel('prediction.xls')

    def bayes(self):
        self.listWidget.clear()
        global cross_val_score_bayes, accuracy_score_bayes, precision_score_bayes, recall_score_bayes, f1_score_bayes
        clf = MultinomialNB()
        cross_val_score_bayes = str(round(np.around(np.mean(cross_val_score(clf,
                                                                            X_train,
                                                                            Y_train,
                                                                            cv=5)),
                                                    decimals=4), 5))
        # self.listWidget.addItem("Доля верной классификации при кроссвалидации: " + cross_val_score_bayes)
        clf.fit(X_train, Y_train)
        clf_pred = clf.predict(X_valid)
        accuracy_score_bayes = str(
            round(np.around(accuracy_score(Y_valid, clf_pred),
                            decimals=4) * 100, 5)) + "%"
        self.listWidget.addItem("Доля верно классификации: " + accuracy_score_bayes)
        precision_score_bayes = str(round(
            np.around(precision_score(Y_valid, clf_pred, zero_division=0),
                      decimals=4), 5))
        self.listWidget.addItem("Точность: " + precision_score_bayes)
        recall_score_bayes = str(round(
            np.around(recall_score(Y_valid, clf_pred, zero_division=0), decimals=4), 5))
        self.listWidget.addItem("Полнота: " + recall_score_bayes)
        f1_score_bayes = str(round(
            np.around(f1_score(Y_valid, clf_pred, zero_division=0), decimals=4), 5))
        self.listWidget.addItem("F-мера: " + f1_score_bayes)
        Ycsv['bayes'] = clf_pred
        Ycsv.to_excel('prediction.xls')

    def discriminant_analysis(self):
        self.listWidget.clear()
        global cross_val_score_discriminant, accuracy_score_discriminant, precision_score_discriminant, recall_score_discriminant, f1_score_discriminant
        disc = LinearDiscriminantAnalysis()
        cross_val_score_discriminant = str(round(np.around(np.mean(cross_val_score(disc,
                                                                                   X_train,
                                                                                   Y_train,
                                                                                   cv=5)),
                                                           decimals=4) * 100, 5)) + "%"
        # self.listWidget.addItem("Доля верной классификации при кроссвалидации: " + cross_val_score_discriminant)
        disc.fit(X_train, Y_train)
        disc_pred = disc.predict(X_valid)
        accuracy_score_discriminant = str(round(
            np.around(accuracy_score(Y_valid, disc_pred),
                      decimals=4) * 100, 5)) + "%"
        self.listWidget.addItem("Доля верно классификации: " + accuracy_score_discriminant)
        precision_score_discriminant = str(round(
            np.around(precision_score(Y_valid, disc_pred, zero_division=0),
                      decimals=4), 5))
        self.listWidget.addItem("Точность: " + precision_score_discriminant)
        recall_score_discriminant = str(round(
            np.around(recall_score(Y_valid, disc_pred, zero_division=0), decimals=4), 5))
        self.listWidget.addItem("Полнота: " + recall_score_discriminant)
        f1_score_discriminant = str(round(
            np.around(f1_score(Y_valid, disc_pred, zero_division=0), decimals=4), 5))
        self.listWidget.addItem("F-мера: " + f1_score_discriminant)
        Ycsv['disc'] = disc_pred
        Ycsv.to_excel('prediction.xls')

    def svm_vectors(self):
        self.listWidget.clear()
        global cross_val_score_svm, accuracy_score_svm, precision_score_svm, recall_score_svm, f1_score_svm
        support = SVC()
        cross_val_score_svm = str(
            round(np.around(np.mean(cross_val_score(support,
                                                    X_train,
                                                    Y_train,
                                                    cv=5)),
                            decimals=4) * 100, 5)) + "%"
        # self.listWidget.addItem("Доля верной классификации при кроссвалидации: " + cross_val_score_svm)
        support.fit(X_train, Y_train)
        support_pred = support.predict(X_valid)
        accuracy_score_svm = str(
            round(np.around(accuracy_score(Y_valid, support_pred),
                            decimals=4) * 100, 5)) + "%"
        self.listWidget.addItem("Доля верно классификации: " + accuracy_score_svm)
        precision_score_svm = str(round(
            np.around(precision_score(Y_valid, support_pred, zero_division=0),
                      decimals=4), 5))
        self.listWidget.addItem("Точнось: " + precision_score_svm)
        recall_score_svm = str(round(
            np.around(recall_score(Y_valid, support_pred, zero_division=0), decimals=4), 5))
        self.listWidget.addItem("Полнота: " + recall_score_svm)
        f1_score_svm = str(round(
            np.around(f1_score(Y_valid, support_pred, zero_division=0), decimals=4), 5))
        self.listWidget.addItem("F-мера: " + f1_score_svm)
        Ycsv['vectors'] = support_pred
        Ycsv.to_excel('prediction.xls')

    def tree(self):
        self.listWidget.clear()
        global cross_val_score_tree, accuracy_score_tree, precision_score_tree, recall_score_tree, f1_score_tree
        tree = DecisionTreeClassifier()
        cross_val_score_tree = str(round(np.around(np.mean(cross_val_score(tree,
                                                                           X_train,
                                                                           Y_train,
                                                                           cv=5)),
                                                   decimals=4) * 100, 5)) + "%"
        # self.listWidget.addItem("Доля верной классификации при кроссвалидации: " + cross_val_score_tree)

        tree.fit(X_train, Y_train)
        tree_pred = tree.predict(X_valid)
        accuracy_score_tree = str(round(np.around(accuracy_score(Y_valid, tree_pred),
                                                  decimals=4) * 100, 5)) + "%"
        self.listWidget.addItem("Доля верно классификации: " + accuracy_score_tree)
        precision_score_tree = str(round(
            np.around(precision_score(Y_valid, tree_pred, zero_division=0),
                      decimals=4), 5))
        self.listWidget.addItem("Точность: " + precision_score_tree)
        recall_score_tree = str(round(
            np.around(recall_score(Y_valid, tree_pred, zero_division=0), decimals=4), 5))
        self.listWidget.addItem("Полнота: " + recall_score_tree)
        f1_score_tree = str(round(
            np.around(f1_score(Y_valid, tree_pred, zero_division=0), decimals=4), 5))
        self.listWidget.addItem("F-мера: " + f1_score_tree)
        Ycsv['tree'] = tree_pred
        Ycsv.to_excel('prediction.xls')

    def neural_network(self):
        self.listWidget.clear()
        global cross_val_score_network, accuracy_score_network, precision_score_network, recall_score_network, f1_score_network
        neural = MLPClassifier()
        cross_val_score_network = str(round(np.around(np.mean(cross_val_score(neural,
                                                                              X_train,
                                                                              Y_train,
                                                                              cv=5)),
                                                      decimals=4) * 100, 5)) + "%"
        # self.listWidget.addItem("Доля верной классификации при кроссвалидации: " + cross_val_score_network)
        neural.fit(X_train, Y_train)
        neural_pred = neural.predict(X_valid)
        accuracy_score_network = str(round(np.around(accuracy_score(Y_valid, neural_pred),
                                                     decimals=4) * 100, 5)) + "%"
        self.listWidget.addItem("Доля верно классификации: " + accuracy_score_network)
        precision_score_network = str(round(
            np.around(precision_score(Y_valid, neural_pred, zero_division=0), decimals=4), 5))
        self.listWidget.addItem("Точность: " + precision_score_network)
        recall_score_network = str(round(
            np.around(recall_score(Y_valid, neural_pred, zero_division=0), decimals=4), 5))
        self.listWidget.addItem("Полнота: " + recall_score_network)
        f1_score_network = str(round(
            np.around(f1_score(Y_valid, neural_pred, zero_division=0), decimals=4), 5))
        self.listWidget.addItem("F-мера: " + f1_score_network)
        Ycsv['neural_network'] = neural_pred
        Ycsv.to_excel('prediction.xls')

    def radio_choise(self):
        if self.label_2.text() != "Вы не открыли файл" and self.label_2.text() != " ":
            if self.lineEdit.text().isdigit() == True:
                if int(self.lineEdit.text()) >= 1 and int(self.lineEdit.text()) <= 100:
                    self.listWidget.show()
                    self.label_5.show()
                    self.label_4.hide()
                    self.pushButton_2.show()
                    if self.radioButton.isChecked() == True:
                        self.bayes()

                    if self.radioButton_2.isChecked() == True:
                        self.logistic_regression()

                    if self.radioButton_3.isChecked() == True:
                        self.svm_vectors()

                    if self.radioButton_4.isChecked() == True:
                        self.discriminant_analysis()

                    if self.radioButton_5.isChecked() == True:
                        self.tree()

                    if self.radioButton_6.isChecked() == True:
                        self.neural_network()
                else:
                    self.listWidget.hide()
                    self.listWidget.clear()
                    self.label_5.hide()
                    self.label_4.show()
                    self.pushButton_2.hide()
                    self.label_4.setText("Введите от 1 до 100!")
            else:
                self.listWidget.hide()
                self.listWidget.clear()
                self.label_5.hide()
                self.label_4.show()
                self.pushButton_2.hide()
                self.label_4.setText("Введите от 1 до 100!")

    def output_analys(self):
        wb = xlwt.Workbook()
        ws = wb.add_sheet('Output')  # , cell_overwrite_ok = True)
        # ws = wb.add_sheet("Output", cell_overwrite_ok=True)
        #ws.write(1, 0, "Доля верной классификации при кроссвалидации")
        ws.write(1, 0, "Доля верной классификации")
        ws.write(2, 0, "Точность")
        ws.write(3, 0, "Полнота")
        ws.write(4, 0, "F-мера")

        ws.write(0, 1, "Логистическая регрессия")
        ws.write(0, 2, "Байесовский классификатор")
        ws.write(0, 3, "Дискриминантный анализ")
        ws.write(0, 4, "Опорные вектора")
        ws.write(0, 5, "Деревья решений")
        ws.write(0, 6, "Нейронная сеть")

        try:
            #ws.write(1, 1, cross_val_score_logistic)
            ws.write(1, 1, accuracy_score_logistic)
            ws.write(2, 1, precision_score_logistic)
            ws.write(3, 1, recall_score_logistic)
            ws.write(4, 1, f1_score_logistic)
        except:
            pass
        try:
            #ws.write(1, 2, cross_val_score_bayes)
            ws.write(1, 2, accuracy_score_bayes)
            ws.write(2, 2, precision_score_bayes)
            ws.write(3, 2, recall_score_bayes)
            ws.write(4, 2, f1_score_bayes)
        except:
            pass
        try:
            #ws.write(1, 3, cross_val_score_discriminant)
            ws.write(1, 3, accuracy_score_discriminant)
            ws.write(2, 3, precision_score_discriminant)
            ws.write(3, 3, recall_score_discriminant)
            ws.write(4, 3, f1_score_discriminant)
        except:
            pass
        try:
            #ws.write(1, 4, cross_val_score_svm)
            ws.write(1, 4, accuracy_score_svm)
            ws.write(2, 4, precision_score_svm)
            ws.write(3, 4, recall_score_svm)
            ws.write(4, 4, f1_score_svm)
        except:
            pass
        try:
            #ws.write(1, 5, cross_val_score_tree)
            ws.write(1, 5, accuracy_score_tree)
            ws.write(2, 5, precision_score_tree)
            ws.write(3, 5, recall_score_tree)
            ws.write(4, 5, f1_score_tree)
        except:
            pass
        try:
            #ws.write(1, 6, cross_val_score_network)
            ws.write(1, 6, accuracy_score_network)
            ws.write(2, 6, precision_score_network)
            ws.write(3, 6, recall_score_network)
            ws.write(4, 6, f1_score_network)
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

# string = "306"
# string.isdigit()
