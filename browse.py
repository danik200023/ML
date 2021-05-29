import os
from PyQt5 import QtWidgets
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


def browse_folder(window):
    try:
        window.label_2.show()
        fname = QtWidgets.QFileDialog.getOpenFileName(window, 'Open file', 'data', 'Тип файлов (*csv)')[0]
        name = os.path.basename(fname)
        percent = float(window.lineEdit.text()) / 100
        index = name.index('.')
        window.label_2.setText("Выбранный файл:\n " + str(name[:index]))
        window.data = pd.read_csv(fname)
        window.Y = window.data[window.data.columns[-1]].astype('int')
        window.X = window.data.drop(window.data.columns[-1], axis=1)
        window.X_train, window.X_valid, window.Y_train, window.Y_valid = train_test_split(window.X, window.Y,
                                                                                          test_size=percent)
        window.X_train_normalize = preprocessing.normalize(window.X_train)
        window.X_train_scale = preprocessing.scale(window.X_train)
        window.X_train_normalize_scale = preprocessing.scale(window.X_train_normalize)
        window.X_train_scale_normalize = preprocessing.normalize(window.X_train_scale)
        window.X_valid_normalize = preprocessing.normalize(window.X_valid)
        window.X_valid_scale = preprocessing.scale(window.X_valid)
        window.X_valid_normalize_scale = preprocessing.scale(window.X_valid_normalize)
        window.X_valid_scale_normalize = preprocessing.normalize(window.X_valid_scale)
    except:
        window.label_2.show()
        window.label_2.setText("Вы не открыли файл")
