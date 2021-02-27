import numpy as np
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import math
from numpy import pi, exp, sin, cos, cosh, tanh, real, imag


t = np.linspace(0, 4*pi, 200) #時間軸
fre=np.linspace(0,100,200)
y=sin(t)*exp(t)

plt.plot(t,y)
plt.show()


def dft(f):
    n = len(f)
    Yr = []
    Yi=[]
    for x in range(n):
        y = 0j
        for t in range(n):
            a = 2 * pi * t * x / n
            y += f[t] * exp(-1j * a)
        Yr.append(abs(y.real))
        Yi.append(abs(y.imag))
    return Yr,Yi

yf_r,yf_i=dft(y)

print(len(yf_r))

plt.plot(fre,yf_r)
plt.show()

plt.plot(fre,yf_i)
plt.show()