import sys
from init_class import Window
import browse
from PyQt5 import QtWidgets


def main():
    app = QtWidgets.QApplication(sys.argv)  # Новый экземпляр QApplication
    window = Window()  # Создаём объект класса Window
    window.show()  # Показываем окно
    window.pushButton.clicked.connect(lambda: browse.browse_folder(window))
    app.exec_()  # и запускаем приложение


if __name__ == '__main__':  # Если мы запускаем файл напрямую, а не импортируем
    main()  # то запускаем функцию main()
