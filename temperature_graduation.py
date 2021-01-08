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
import os

a_0 = 150
delta_r = 0.2*a_0
ratio_a  = 0.36
R_0     = a_0/ratio_a
ratio_Ti = 5.55
ratio_Te = 6.92
L_Ti = R_0 / ratio_Ti
L_Te = R_0 / ratio_Te
L_x = a_0

r=np.arange(0,150,1)

Ti = exp(-delta_r / L_Ti * tanh((r - 0.7* L_x) / delta_r))
Te = exp(-delta_r / L_Te * tanh((r - 0.3 * L_x) / delta_r))

plt.rcParams["font.size"] = 14
plt.tight_layout()
plt.title('イオン温度勾配がa_0*0.7,電子温度勾配がa_0*0.3で急峻', fontname="MS Gothic")
plt.xlabel('r')
plt.plot(r,Ti)
plt.plot(r,Te)
plt.legend(["イオン温度勾配","電子温度勾配"], prop={"family":"MS Gothic"})
plt.show()
