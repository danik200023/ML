import design
from PyQt5 import QtWidgets


class Window(QtWidgets.QMainWindow, design.Ui_MainWindow):

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.window2 = Second_Window()
        # self.pushButton.clicked.connect(browse.browse_folder(self))
        """
        self.pushButton_4.clicked.connect(self.radio_choise)
        self.pushButton_2.clicked.connect(self.output_analys)
        self.pushButton_3.clicked.connect(self.browse_new_data)
        self.pushButton_5.clicked.connect(self.radio_choise)
        self.pushButton_5.clicked.connect(self.tstat)
        self.pushButton_6.clicked.connect(self.agregation)
        self.checkBox.clicked.connect(self.checkCross)
        self.checkBox.clicked.connect(self.table)
        self.checkBox_9.clicked.connect(self.significance)
        self.window2.pushButton.clicked.connect(self.significance)
        self.window2.pushButton.clicked.connect(self.tstat)
        """
        self.checkBox_3.setChecked(True)
        self.checkBox_4.setChecked(True)
        self.checkBox_5.setChecked(True)
        self.checkBox_6.setChecked(True)
        self.checkBox_7.setChecked(True)
        self.checkBox_8.setChecked(True)
        self.label_2.hide()
        self.label_5.hide()
        self.tableWidget.hide()
        #self.pushButton_2.hide()
        self.lineEdit_2.hide()


class Second_Window(QtWidgets.QMainWindow, design.Ui_SecondWindow):

    def __init__(self):
        super().__init__()
        self.setupUi(self)
