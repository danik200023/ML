import sys
from init_class import Window
import browse
import analysis
from PyQt5 import QtWidgets


def main():
    app = QtWidgets.QApplication(sys.argv)  # Новый экземпляр QApplication
    window = Window()  # Создаём объект класса Window
    window.show()  # Показываем окно
    window.pushButton.clicked.connect(lambda: browse.browse_folder(window))
    """window.pushButton_4.clicked.connect(
        lambda: analysis.logistic_regression(window, window.x_train, window.x_valid, window.y_train, window.y_valid,
                                             window.x_train_normalize, window.x_train_scale,
                                             window.x_train_normalize_scale, window.x_train_scale_normalize,
                                             window.x_valid_normalize, window.x_valid_scale,
                                             window.x_valid_normalize_scale, window.x_valid_scale_normalize))
                                             """
    window.pushButton_4.clicked.connect(
        lambda: analysis.bayes(window, window.x_train, window.x_valid, window.y_train, window.y_valid,
                              window.x_train_normalize, window.x_train_scale,
                              window.x_train_normalize_scale, window.x_train_scale_normalize,
                              window.x_valid_normalize, window.x_valid_scale,
                              window.x_valid_normalize_scale, window.x_valid_scale_normalize))
    """window.pushButton_4.clicked.connect(
        lambda: analysis.tree(window, window.x_train, window.x_valid, window.y_train, window.y_valid,
                               window.x_train_normalize, window.x_train_scale,
                               window.x_train_normalize_scale, window.x_train_scale_normalize,
                               window.x_valid_normalize, window.x_valid_scale,
                               window.x_valid_normalize_scale, window.x_valid_scale_normalize))
                               """
    app.exec_()  # и запускаем приложение


if __name__ == '__main__':  # Если мы запускаем файл напрямую, а не импортируем
    main()  # то запускаем функцию main()
