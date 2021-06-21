import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

TN = 997990
FP = 1910
FN = 10
TP = 90
#[[ 2 10]
#[ 4 89]]cf_matrix
cf_matrix = np.array([[TP, FP], [TN, FN]])
print(cf_matrix)
group_names = ["Верно-исправные (TP)", "Ложно-исправные (FP)", "Верно-неисправные (TN)", "Ложно-неисправные (FN)"]
group_counts = ["{0: 0.0f}".format(value) for value in cf_matrix.flatten()]
labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_names, group_counts)]
labels = np.asarray(labels).reshape(2, 2)
sns.heatmap(cf_matrix, annot=labels, fmt="", cmap="Blues", cbar=True, xticklabels=False,
                yticklabels=False,)
plt.title("Модель 2")
plt.show()