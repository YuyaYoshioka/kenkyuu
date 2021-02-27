import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import dot, multiply, diag, power
from numpy import pi, exp, sin, cos, cosh, tanh, real, imag
from numpy.linalg import inv, eig, pinv
from scipy.linalg import svd, svdvals
from scipy.integrate import odeint, ode, complex_ode
from warnings import warn
from scipy.linalg import eigh
import glob
import math
from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_EVEN

file_list = glob.glob('*.txt')

u=[]

for filename in file_list:
    with open(filename) as input:
        input_=input.read()
        u.append(input_.split())

num=2
y_num=1
print(u)

y=set()

U=[]
n_u=[]
while num<len(u[0]):
    n_u.append(float(u[0][num]))
    y.add(float(u[0][y_num]))
    if len(n_u)==80:
        U.append(n_u)
        n_u=[]
    num+=3
    y_num+=3


x=np.arange(-4.938,4.938,0.125)
y=np.array(list(y))
X, Y = np.meshgrid(x, y)
print(X,Y)

plt.pcolormesh(X, Y, U, cmap='hsv') # 等高線図の生成。cmapで色付けの規則を指定する。
#plt.pcolor(X, Y, Z, cmap='hsv') # 等高線図の生成。cmapで色付けの規則を指定する。

pp=plt.colorbar (orientation="vertical") # カラーバーの表示
pp.set_label("Label", fontname="Arial", fontsize=24) #カラーバーのラベル

plt.xlabel('X', fontsize=24)
plt.ylabel('Y', fontsize=24)

plt.show()