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
        window.y = window.data[window.data.columns[-1]].astype('int')
        window.x = window.data.drop(window.data.columns[-1], axis=1)
        window.x_train, window.x_valid, window.y_train, window.y_valid = train_test_split(window.x, window.y,
                                                                                          test_size=percent)
        window.x_train_normalize = preprocessing.normalize(window.x_train)
        window.x_train_scale = preprocessing.scale(window.x_train)
        window.x_train_normalize_scale = preprocessing.scale(window.x_train_normalize)
        window.x_train_scale_normalize = preprocessing.normalize(window.x_train_scale)
        window.x_valid_normalize = preprocessing.normalize(window.x_valid)
        window.x_valid_scale = preprocessing.scale(window.x_valid)
        window.x_valid_normalize_scale = preprocessing.scale(window.x_valid_normalize)
        window.x_valid_scale_normalize = preprocessing.normalize(window.x_valid_scale)
        return 1
    except:
        window.label_2.show()
        window.label_2.setText("Вы не открыли файл")
