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

times=[]
for n in range(1,34):
    time=[]
    while n<len(u[0]):
        time.append((math.log(float(u[0][n]))))
        n+=34
    times.append(time)
t=np.linspace(0,85,851)
ganma=[]

fig = plt.figure()

for n in range(33):
    plt.xlabel('t')
    plt.plot(t,times[n],label=n)
    plt.legend()
    if n%10==0:
        plt.show()
        fig = plt.figure()

    print(len(times))
for n in range(1,33):
    ganma.append((times[n][800]-times[n][500])/300)
n=np.linspace(0,32,32)

fig = plt.figure()

plt.plot(n,ganma)
plt.show()
