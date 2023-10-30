import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys 
sys.path.append('../..')

from linear_regression import LinearRegression

data = pd.read_csv('../data/airfoil_noise_samples.csv')

# 将数据集分为训练集和测试集
train_data = data.sample(frac = 0.8)
test_data = data.drop(train_data.index)

input_param_name = 'Displacement'
output_param_name = 'Sound Pressure'

x_train = train_data[input_param_name].values
y_train = train_data[output_param_name].values

x_test = test_data[input_param_name].values
y_test = test_data[output_param_name].values



plt.scatter(x_train, y_train, label= 'Train_data')
plt.scatter(x_test, y_test, label= 'Test_data')
plt.xlabel(input_param_name)
plt.ylabel(output_param_name)
plt.title('xtj')
plt.legend()
plt.show()


# num_iterations = 500
# learing_rate = 0.01

# linear_regression = LinearRegression(x_train,y_train)
# (theta, cost_history) = linear_regression.train(learing_rate, num_iterations)

# print('开始的损失：',cost_history[0])
# print('训练后的损失',cost_history[-1])