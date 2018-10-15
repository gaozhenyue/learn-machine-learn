import numpy as np
from numpy import mat

x0 = mat([-1,-1,0,1,1,1]).reshape(6,1)
x1 = mat([-2,-1,0,2,3,4]).reshape(6,1)

def sigmoid(z):
    return 1/(1 + np.exp(-z))

z = 2*x0 + 3*x1
y = sigmoid(z)
# print(Z)

#逻辑回归拟合数据
#梯度下降算法
theta0 = 1.
theta1 = 1.
alpha = 0.2

for i in range(10000):
    hx = sigmoid(theta0 * x0 + theta1 * x1)

    temp0 = theta0 - alpha * x0.T * (hx - y) /6
    temp1 = theta1 - alpha * x1.T * (hx - y) /6

    theta0 = temp0[0,0]
    theta1 = temp1[0,0]

# ws1 = mat([theta0,theta1]).reshape(2,1)
# print("逻辑回归梯度下降法的输出\n",ws1)

print(theta0)
print(theta1)