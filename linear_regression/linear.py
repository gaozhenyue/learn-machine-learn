import numpy as np
from numpy.linalg import inv
from numpy import dot
from numpy import mat

#y = 2*x
x1 = mat([1,2,3,4]).reshape(4,1)
# print(x1)
x0 = mat([1,1,1,1]).reshape(4,1)
# print(x0)
y = 2*x0 + 3*x1
# print(y)
x = np.column_stack((x0,x1))
# print(x)
#最小二乘法
xTx = x.T * x
ws = xTx.I * (x.T * y)
print("最小二乘法的输出\n",ws)

#梯度下降法
theta0 = 1.
theta1 = 1.
alpha = 0.2


for i in range(1000):
    hx = theta0 * x0 + theta1 * x1
    temp0 = theta0 - alpha * x0.reshape(1,4) * (hx - y)/4
    temp1 = theta1 - alpha * x1.reshape(1,4) * (hx - y)/4
    theta0 = temp0[0,0]
    theta1 = temp1[0,0]

ws1 = mat([theta0,theta1]).reshape(2,1)
print("梯度下降法的输出\n",ws1)







