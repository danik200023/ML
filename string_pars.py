from math import sqrt
from numpy import mean
from scipy.stats import t, sem
import pandas as pd
import os


#string = "pisoska ot anato@liyta@"
#print(string[4:8:1])
#print(string.find("ana"))
#print(string[11:len(string):1])
#print(string.find("@"))
#print(string[16:len(string):1])



def independent_ttest(data1, data2, alpha):
	# calculate means
	mean1, mean2 = mean(data1), mean(data2)
	# calculate standard errors
	se1, se2 = sem(data1), sem(data2)
	# standard error on the difference between the samples
	sed = sqrt(se1**2.0 + se2**2.0)
	# calculate the t statistic
	t_stat = (mean1 - mean2) / sed
	# degrees of freedom
	df = len(data1) + len(data2) - 2
	# calculate the critical value
	cv = t.ppf(1.0 - alpha, df)
	# calculate the p-value
	p = (1.0 - t.cdf(abs(t_stat), df)) * 2.0
	# return everything
	return t_stat, df, cv, p

data = pd.read_csv(os.path.normpath(os.getcwd() + os.sep + os.pardir + os.sep + os.pardir) + "\\Универ\\6 Семестр\\ML\\data\\vodoochistka.csv")
alpha = 0.05
print(str(len(data.columns)))
print(str(data[data.columns[0]].name))
#print(data[[str(1), '2', '3']])
true_t_metrics = []
for i in range(len(data.columns) - 1):
	print("Т-критерий для " + str(data[data.columns[i]].name))
	data1 = (data[data[str(len(data.columns))] == int(0)])[(data[data[str(len(data.columns))] == int(0)]).columns[i]].to_numpy()
	data2 = (data[data[str(len(data.columns))] == int(1)])[(data[data[str(len(data.columns))] == int(1)]).columns[i]].to_numpy()
	t_stat, df, cv, p = independent_ttest(data1, data2, alpha)
	print('t=%.3f, df=%d, cv=%.3f, p=%.3f' % (t_stat, df, cv, p))
	# interpret via critical value
	if abs(t_stat) <= cv:
		print("Принимается нулевая гипотеза о том, что средние равны (по критическому значению)")#print('Accept null hypothesis that the means are equal.')
	else:
		print("Отвергается нулевая гипотеза о том, что средние равны (по критическому значению")#print('Reject the null hypothesis that the means are equal.')
	# interpret via p-value
	if p > alpha:
		print("Принимается нулевая гипотеза о том, что средние равны (по p - значению")#print('Accept null hypothesis that the means are equal.')
	else:
		print("Отвергается нулевая гипотеза о том, что средние равны (по p - значению")#print('Reject the null hypothesis that the means are equal.')"""
	if abs(t_stat) <= cv and p > alpha:
		true_t_metrics.append(str(data[data.columns[i]].name))
print(true_t_metrics)
true_t_metrics.append(data[data.columns[len(data.columns) - 1]].name)
data = data[true_t_metrics]# + str(last_column)]
print(data)