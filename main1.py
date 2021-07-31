import sys
import tstat
import table
from init_class import Window
import browse
import analysis
from PyQt5 import QtWidgets, QtCore


def main():
    app = QtWidgets.QApplication(sys.argv)  # Новый экземпляр QApplication
    window = Window()  # Создаём объект класса Window
    window.show()  # Показываем окно
    window.pushButton.clicked.connect(lambda: browse.browse_folder(window))
    # window.pushButton_4.clicked.connect(lambda: table.cleaning(window))
    window.checkBox_9.clicked.connect(lambda: tstat.significance(window))
    window.pushButton_4.clicked.connect(lambda: analysis.significant(window))
    window.pushButton_4.clicked.connect(
        lambda: analysis.logistic_regression(window, window.x_train, window.x_valid, window.y_train, window.y_valid,
                                             window.x_train_normalize, window.x_train_scale,
                                             window.x_train_normalize_scale, window.x_train_scale_normalize,
                                             window.x_valid_normalize, window.x_valid_scale,
                                             window.x_valid_normalize_scale, window.x_valid_scale_normalize))
    window.pushButton_4.clicked.connect(
        lambda: analysis.bayes(window, window.x_train, window.x_valid, window.y_train, window.y_valid,
                               window.x_train_normalize, window.x_train_scale,
                               window.x_train_normalize_scale, window.x_train_scale_normalize,
                               window.x_valid_normalize, window.x_valid_scale,
                               window.x_valid_normalize_scale, window.x_valid_scale_normalize))

    # window.pushButton_4.clicked.connect(
    #     lambda: analysis.discriminant_analysis(window, window.x_train, window.x_valid, window.y_train, window.y_valid,
    #                                            window.x_train_normalize, window.x_train_scale,
    #                                            window.x_train_normalize_scale, window.x_train_scale_normalize,
    #                                            window.x_valid_normalize, window.x_valid_scale,
    #                                            window.x_valid_normalize_scale, window.x_valid_scale_normalize))
    window.pushButton_4.clicked.connect(
        lambda: analysis.svm_vectors(window, window.x_train, window.x_valid, window.y_train, window.y_valid,
                                     window.x_train_normalize, window.x_train_scale,
                                     window.x_train_normalize_scale, window.x_train_scale_normalize,
                                     window.x_valid_normalize, window.x_valid_scale,
                                     window.x_valid_normalize_scale, window.x_valid_scale_normalize))
    window.pushButton_4.clicked.connect(
        lambda: analysis.tree(window, window.x_train, window.x_valid, window.y_train, window.y_valid,
                              window.x_train_normalize, window.x_train_scale,
                              window.x_train_normalize_scale, window.x_train_scale_normalize,
                              window.x_valid_normalize, window.x_valid_scale,
                              window.x_valid_normalize_scale, window.x_valid_scale_normalize))
    window.pushButton_4.clicked.connect(
        lambda: analysis.neural_network(window, window.x_train, window.x_valid, window.y_train, window.y_valid,
                                        window.x_train_normalize, window.x_train_scale,
                                        window.x_train_normalize_scale, window.x_train_scale_normalize,
                                        window.x_valid_normalize, window.x_valid_scale,
                                        window.x_valid_normalize_scale, window.x_valid_scale_normalize))
    # window.pushButton_4.clicked.connect(
    #    lambda: table.cleaning(window)
    # )
    window.pushButton_4.clicked.connect(
        lambda: analysis.agregation(window)
    )
    window.pushButton_4.clicked.connect(
        lambda: table.table(window)
    )

    app.exec_()  # и запускаем приложение


if __name__ == '__main__':  # Если мы запускаем файл напрямую, а не импортируем
    main()  # то запускаем функцию main()
