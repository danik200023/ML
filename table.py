from PyQt5.QtWidgets import QTableWidgetItem
from PyQt5 import QtGui

def table(window):
    window.tableWidget.show()
    window.tableWidget.setColumnCount(5)
    window.tableWidget.setRowCount(9)
    window.tableWidget.setHorizontalHeaderLabels(
        ["Доля верной классификации", "Точность", "Полнота", "F-мера", "AUC"])
    window.tableWidget.setVerticalHeaderLabels(
        ["Логистическая регрессия", "Байесовский классификатор", "Дискриминантный анализ", "Опорные вектора",
            "Деревья решений", "Нейронная сеть", "Агрегирование по среднему", "Агрегирование по медиане",
            "Агрегирование по голосованию"])
    column = 0  # столбец
    row = 0  # строка
    font = QtGui.QFont()
    font.setBold(False)
    for i in range(9):
        window.tableWidget.verticalHeaderItem(i).setFont(font)
    try:
        significant_metrics = []
        try:
            significant_metrics.append(window.logistic_significant)
        except:
            pass
        try:
            significant_metrics.append(window.clf_significant)
        except:
            pass
        try:
            significant_metrics.append(window.disc_significant)
        except:
            pass
        try:
            significant_metrics.append(window.support_significant)
        except:
            pass
        try:
            significant_metrics.append(window.tree_significant)
        except:
            pass
        try:
            significant_metrics.append(window.neural_significant)
        except:
            pass
        best_metric = max(significant_metrics)
        best_metric_index = significant_metrics.index(best_metric)
        font = QtGui.QFont()
        font.setBold(True)
        #window.tableWidget.item(best_metric_index, 0).setFont(font)
        #window.tableWidget.item(4, 0).setFont(font)
        window.tableWidget.verticalHeaderItem(best_metric_index).setFont(font)
    except:
        pass



    try:
        #window.tableWidget.setItem(row, column, QTableWidgetItem(str(window.logistic_accuracy[
        #                                                                 max(set(window.logistic_max_values_indexes),
        #                                                                     key=window.logistic_max_values_indexes.count)])))
        #window.tableWidget.setItem(row, column + 1, QTableWidgetItem(str(window.logistic_precision[max(set(
        #    window.logistic_max_values_indexes), key=window.logistic_max_values_indexes.count)])))
        #window.tableWidget.setItem(row, column + 2, QTableWidgetItem(str(window.logistic_recall[max(set(
        #    window.logistic_max_values_indexes), key=window.logistic_max_values_indexes.count)])))
        #window.tableWidget.setItem(row, column + 3, QTableWidgetItem(str(window.logistic_f1[max(set(
        #    window.logistic_max_values_indexes), key=window.logistic_max_values_indexes.count)])))
        #window.tableWidget.setItem(row, column + 4, QTableWidgetItem(str(window.logistic_roc_auc[max(set(
        #    window.logistic_max_values_indexes), key=window.logistic_max_values_indexes.count)])))
        #print(window.logistic_accuracy[
        #                                                                 max(set(window.logistic_max_values_indexes),
        #                                                                     key=window.logistic_max_values_indexes.count)])
        window.tableWidget.setItem(row, column, QTableWidgetItem(str(window.logistic_accuracy)))
        window.tableWidget.setItem(row, column + 1, QTableWidgetItem(str(window.logistic_precision)))
        window.tableWidget.setItem(row, column + 2, QTableWidgetItem(str(window.logistic_recall)))
        window.tableWidget.setItem(row, column + 3, QTableWidgetItem(str(window.logistic_f1)))
        window.tableWidget.setItem(row, column + 4, QTableWidgetItem(str(window.logistic_roc_auc)))
        if window.logistic_f1_significant:
            window.tableWidget.item(row, column + 3).setForeground(QtGui.QColor(0, 160, 0))
        if window.logistic_roc_auc_significant:
            window.tableWidget.item(row, column + 4).setForeground(QtGui.QColor(0, 160, 0))










    except:
        pass
    row += 1
    try:
        #window.tableWidget.setItem(row, column, QTableWidgetItem(str(window.clf_accuracy[
        #                                                                 max(set(window.clf_max_values_indexes),
        #                                                                     key=window.clf_max_values_indexes.count)])))
        #window.tableWidget.setItem(row, column + 1, QTableWidgetItem(str(window.clf_precision[max(set(
        #    window.clf_max_values_indexes), key=window.clf_max_values_indexes.count)])))
        #window.tableWidget.setItem(row, column + 2, QTableWidgetItem(str(window.clf_recall[max(set(
        #    window.clf_max_values_indexes), key=window.clf_max_values_indexes.count)])))
        #window.tableWidget.setItem(row, column + 3, QTableWidgetItem(str(window.clf_f1[max(set(
        #    window.clf_max_values_indexes), key=window.clf_max_values_indexes.count)])))
        #window.tableWidget.setItem(row, column + 4, QTableWidgetItem(str(window.clf_roc_auc[max(set(
        #    window.clf_max_values_indexes), key=window.clf_max_values_indexes.count)])))
        window.tableWidget.setItem(row, column, QTableWidgetItem(str(window.clf_accuracy)))
        window.tableWidget.setItem(row, column + 1, QTableWidgetItem(str(window.clf_precision)))
        window.tableWidget.setItem(row, column + 2, QTableWidgetItem(str(window.clf_recall)))
        window.tableWidget.setItem(row, column + 3, QTableWidgetItem(str(window.clf_f1)))
        window.tableWidget.setItem(row, column + 4, QTableWidgetItem(str(window.clf_roc_auc)))
        if window.clf_f1_significant:
            window.tableWidget.item(row, column + 3).setForeground(QtGui.QColor(0, 160, 0))
        if window.clf_roc_auc_significant:
            window.tableWidget.item(row, column + 4).setForeground(QtGui.QColor(0, 160, 0))
    except:
        pass
    row += 1
    try:
        #window.tableWidget.setItem(row, column, QTableWidgetItem(str(window.disc_accuracy[
        #                                                                 max(set(window.disc_max_values_indexes),
        #                                                                     key=window.disc_max_values_indexes.count)])))
        #window.tableWidget.setItem(row, column + 1, QTableWidgetItem(str(window.disc_precision[max(set(
        #    window.disc_max_values_indexes), key=window.disc_max_values_indexes.count)])))
        #window.tableWidget.setItem(row, column + 2, QTableWidgetItem(str(window.disc_recall[max(set(
        #    window.disc_max_values_indexes), key=window.disc_max_values_indexes.count)])))
        #window.tableWidget.setItem(row, column + 3, QTableWidgetItem(str(window.disc_f1[max(set(
        #    window.disc_max_values_indexes), key=window.disc_max_values_indexes.count)])))
        #window.tableWidget.setItem(row, column + 4, QTableWidgetItem(str(window.disc_roc_auc[max(set(
        #    window.disc_max_values_indexes), key=window.disc_max_values_indexes.count)])))
        window.tableWidget.setItem(row, column, QTableWidgetItem(str(window.disc_accuracy)))
        window.tableWidget.setItem(row, column + 1, QTableWidgetItem(str(window.disc_precision)))
        window.tableWidget.setItem(row, column + 2, QTableWidgetItem(str(window.disc_recall)))
        window.tableWidget.setItem(row, column + 3, QTableWidgetItem(str(window.disc_f1)))
        window.tableWidget.setItem(row, column + 4, QTableWidgetItem(str(window.disc_roc_auc)))
        if window.disc_f1_significant:
            window.tableWidget.item(row, column + 3).setForeground(QtGui.QColor(0, 160, 0))
        if window.disc_roc_auc_significant:
            window.tableWidget.item(row, column + 4).setForeground(QtGui.QColor(0, 160, 0))
    except:
        pass
    row += 1
    try:
        #window.tableWidget.setItem(row, column, QTableWidgetItem(str(window.support_accuracy[
        #                                                                 max(set(window.support_max_values_indexes),
        #                                                                     key=window.support_max_values_indexes.count)])))
        #window.tableWidget.setItem(row, column + 1, QTableWidgetItem(str(window.support_precision[max(set(
        #    window.support_max_values_indexes), key=window.support_max_values_indexes.count)])))
        #window.tableWidget.setItem(row, column + 2, QTableWidgetItem(str(window.support_recall[max(set(
        #    window.support_max_values_indexes), key=window.support_max_values_indexes.count)])))
        #window.tableWidget.setItem(row, column + 3, QTableWidgetItem(str(window.support_f1[max(set(
        #    window.support_max_values_indexes), key=window.support_max_values_indexes.count)])))
        #window.tableWidget.setItem(row, column + 4, QTableWidgetItem(str(window.support_roc_auc[max(set(
        #    window.support_max_values_indexes), key=window.support_max_values_indexes.count)])))
        window.tableWidget.setItem(row, column, QTableWidgetItem(str(window.support_accuracy)))
        window.tableWidget.setItem(row, column + 1, QTableWidgetItem(str(window.support_precision)))
        window.tableWidget.setItem(row, column + 2, QTableWidgetItem(str(window.support_recall)))
        window.tableWidget.setItem(row, column + 3, QTableWidgetItem(str(window.support_f1)))
        window.tableWidget.setItem(row, column + 4, QTableWidgetItem(str(window.support_roc_auc)))
        if window.support_f1_significant:
            window.tableWidget.item(row, column + 3).setForeground(QtGui.QColor(0, 160, 0))
        if window.support_roc_auc_significant:
            window.tableWidget.item(row, column + 4).setForeground(QtGui.QColor(0, 160, 0))
    except:
        pass
    row += 1
    try:
        #window.tableWidget.setItem(row, column, QTableWidgetItem(str(window.tree_accuracy[
        #                                                                 max(set(window.tree_max_values_indexes),
        #                                                                     key=window.tree_max_values_indexes.count)])))
        #window.tableWidget.setItem(row, column + 1, QTableWidgetItem(str(window.tree_precision[max(set(
        #    window.tree_max_values_indexes), key=window.tree_max_values_indexes.count)])))
        #window.tableWidget.setItem(row, column + 2, QTableWidgetItem(str(window.tree_recall[max(set(
        #    window.tree_max_values_indexes), key=window.tree_max_values_indexes.count)])))
        #window.tableWidget.setItem(row, column + 3, QTableWidgetItem(str(window.tree_f1[max(set(
        #    window.tree_max_values_indexes), key=window.tree_max_values_indexes.count)])))
        #window.tableWidget.setItem(row, column + 4, QTableWidgetItem(str(window.tree_roc_auc[max(set(
        #    window.tree_max_values_indexes), key=window.tree_max_values_indexes.count)])))
        window.tableWidget.setItem(row, column, QTableWidgetItem(str(window.tree_accuracy)))
        window.tableWidget.setItem(row, column + 1, QTableWidgetItem(str(window.tree_precision)))
        window.tableWidget.setItem(row, column + 2, QTableWidgetItem(str(window.tree_recall)))
        window.tableWidget.setItem(row, column + 3, QTableWidgetItem(str(window.tree_f1)))
        window.tableWidget.setItem(row, column + 4, QTableWidgetItem(str(window.tree_roc_auc)))
        if window.tree_f1_significant:
            window.tableWidget.item(row, column + 3).setForeground(QtGui.QColor(0, 160, 0))
        if window.tree_roc_auc_significant:
            window.tableWidget.item(row, column + 4).setForeground(QtGui.QColor(0, 160, 0))
        
    except:
        pass
    row += 1
    try:
        #window.tableWidget.setItem(row, column, QTableWidgetItem(str(window.neural_accuracy[
        #                                                                 max(set(window.neural_max_values_indexes),
        #                                                                     key=window.neural_max_values_indexes.count)])))
        #window.tableWidget.setItem(row, column + 1, QTableWidgetItem(str(window.neural_precision[max(set(
        #    window.neural_max_values_indexes), key=window.neural_max_values_indexes.count)])))
        #window.tableWidget.setItem(row, column + 2, QTableWidgetItem(str(window.neural_recall[max(set(
        #    window.neural_max_values_indexes), key=window.neural_max_values_indexes.count)])))
        #window.tableWidget.setItem(row, column + 3, QTableWidgetItem(str(window.neural_f1[max(set(
        #    window.neural_max_values_indexes), key=window.neural_max_values_indexes.count)])))
        #window.tableWidget.setItem(row, column + 4, QTableWidgetItem(str(window.neural_roc_auc[max(set(
        #    window.neural_max_values_indexes), key=window.neural_max_values_indexes.count)])))
        window.tableWidget.setItem(row, column, QTableWidgetItem(str(window.neural_accuracy)))
        window.tableWidget.setItem(row, column + 1, QTableWidgetItem(str(window.neural_precision)))
        window.tableWidget.setItem(row, column + 2, QTableWidgetItem(str(window.neural_recall)))
        window.tableWidget.setItem(row, column + 3, QTableWidgetItem(str(window.neural_f1)))
        window.tableWidget.setItem(row, column + 4, QTableWidgetItem(str(window.neural_roc_auc)))
        #font = QtGui.QFont()
        #font.setBold(True)
        #window.tableWidget.item(3, 0).setFont(font)
        if window.neural_f1_significant:
            window.tableWidget.item(row, column + 3).setForeground(QtGui.QColor(0, 160, 0))
        if window.neural_roc_auc_significant:
            window.tableWidget.item(row, column + 4).setForeground(QtGui.QColor(0, 160, 0))

    except:
        pass
        row += 1
    try:
        window.tableWidget.setItem(row, column, QTableWidgetItem(window.accuracy_score_agregation_mean))
        window.tableWidget.setItem(row, column + 1, QTableWidgetItem(window.precision_score_agregation_mean))
        window.tableWidget.setItem(row, column + 2, QTableWidgetItem(window.recall_score_agregation_mean))
        window.tableWidget.setItem(row, column + 3, QTableWidgetItem(window.f1_score_agregation_mean))
        window.tableWidget.setItem(row, column + 4, QTableWidgetItem(window.auc_score_agregation_mean))
        
    except:
        pass
    row += 1
    try:
        window.tableWidget.setItem(row, column, QTableWidgetItem(window.accuracy_score_agregation_median))
        window.tableWidget.setItem(row, column + 1, QTableWidgetItem(window.precision_score_agregation_median))
        window.tableWidget.setItem(row, column + 2, QTableWidgetItem(window.recall_score_agregation_median))
        window.tableWidget.setItem(row, column + 3, QTableWidgetItem(window.f1_score_agregation_median))
        window.tableWidget.setItem(row, column + 4, QTableWidgetItem(window.auc_score_agregation_median))
        
    except:
        pass
    row += 1
    try:
        window.tableWidget.setItem(row, column, QTableWidgetItem(window.accuracy_score_agregation_voting))
        window.tableWidget.setItem(row, column + 1, QTableWidgetItem(window.precision_score_agregation_voting))
        window.tableWidget.setItem(row, column + 2, QTableWidgetItem(window.recall_score_agregation_voting))
        window.tableWidget.setItem(row, column + 3, QTableWidgetItem(window.f1_score_agregation_voting))
        window.tableWidget.setItem(row, column + 4, QTableWidgetItem(window.auc_score_agregation_voting))
        
    except:
        pass
    row += 1

def cleaning(window):
    try:
        del window.tree_accuracy
        del window.tree_precision
        del window.tree_recall
        del window.tree_f1
        del window.tree_roc_auc
    except:
        pass
    try:
        del window.clf_accuracy
        del window.clf_precision
        del window.clf_recall
        del window.clf_f1
        del window.clf_roc_auc
    except:
        pass
    try:
        del window.disc_accuracy
        del window.disc_precision
        del window.disc_recall
        del window.disc_f1
        del window.disc_roc_auc
    except:
        pass
    try:
        del window.support_accuracy
        del window.support_precision
        del window.support_recall
        del window.support_f1
        del window.support_roc_auc
    except:
        pass
    try:
        del window.tree_accuracy
        del window.tree_precision
        del window.tree_recall
        del window.tree_f1
        del window.tree_roc_auc
    except:
        pass
    try:
        del window.neural_accuracy
        del window.neural_precision
        del window.neural_recall
        del window.neural_f1
        del window.neural_roc_auc
    except:
        pass
    """try:
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
        pass"""