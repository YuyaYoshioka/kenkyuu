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
import math


# define time and space domains
x = np.linspace(-pi, pi, 80) #x軸
y = np.linspace(-pi,pi,80) #y軸
t = np.linspace(0, 3*pi, 150) #時間軸
time = np.linspace(0, 2*pi, 100) #時間軸
dt=t[2]-t[1]
f=[]
for t_ in t:
    f_=[]
    for x_ in x:
        for y_ in y:
            f1=(x_+y_)*exp(1j*t_)
            f2=tanh(x_+y_)*exp(2j*t_)
            f_.append(f1+f2)
    f.append(f_)
f=np.array(f)
print(f.shape)
f_r=f.real
print(f.shape)
n_f=f.T
print(n_f.shape)
X0=n_f[:,:-1]
X1=n_f[:,1:]

print(X0.shape)

# X0の特異値分解
mode=np.arange(1,21,1)
U,Sig,Vh2 = svd(X0, False)
plt.xlabel('mode')
plt.ylabel('eigenvalues')
plt.plot(mode,Sig[:20],linestyle='None',marker='.')
plt.show()



# n_Aの計算
r=2 # モード数
Sig_r=np.eye(r)
V = Vh2.conj().T[:,:r]
for a in range(r):
    Sig_r[a][a]=1/Sig[a]
U_r=U[:,:r]
n_A=dot(dot(dot(np.conjugate(U_r.T),X1),V[:,:r]),Sig_r)


# n_Aの固有値、固有ベクトルの計算
lam,W=np.linalg.eig(n_A)

# 固有モードの計算
phi=dot(dot(dot(X1,V[:,:r]),Sig_r),W)
# compute time evolution
b = dot(pinv(phi), X0[:,0])
Psi = np.zeros([r, len(t)], dtype='complex')
for i,_t in enumerate(t):
    Psi[:,i] = multiply(power(lam, _t/dt), b)

n_f = dot(phi, Psi)
f=f.T

plt.xlabel('t')
plt.plot(time,Psi[0].real[:100],label='mode1')
plt.plot(time,Psi[1].real[:100],label='mode2')
plt.show()

plt.xlabel('t')
plt.plot(time,Psi[0].real[:100],label='mode1_Re')
plt.plot(time,Psi[0].imag[:100],label='mode1_Im')
plt.legend()
plt.show()

plt.xlabel('t')
plt.plot(time,Psi[1].real[:100],label='mode2_Re')
plt.plot(time,Psi[1].imag[:100],label='mode2_Im')
plt.legend()
plt.show()

#e^(rt)の計算
ert1=[]
ert2=[]
for n in range(len(Psi[0])):
    ert1.append(((Psi[0][n].real)**2+(Psi[0][n].imag)**2)**0.5)
    ert2.append(((Psi[1][n].real)**2+(Psi[1][n].imag)**2)**0.5)
#cos(ωt),sin(ωt)の計算
coss1=[]
sins1=[]
coss2=[]
sins2=[]
for n in range(len(Psi[0])):
    coss1.append(Psi[0][n].real/ert1[n])
    sins1.append(Psi[0][n].imag/ert1[n])
    coss2.append(Psi[1][n].real / ert2[n])
    sins2.append(Psi[1][n].imag / ert2[n])

#tan(ωt)の計算
atans1=[]
atans2=[]
for n in range(len(Psi[0])):
    atans1.append(math.atan(sins1[n]/coss1[n]))
    atans2.append(math.atan(sins2[n]/coss2[n]))

plt.xlabel('t')
plt.plot(time,atans1,label='mode1')
plt.legend()
plt.show()

plt.xlabel('t')
plt.plot(time,atans2,label='mode2')
plt.legend()
plt.show()

plt.xlabel('t')
plt.plot(time,ert1,label='e^(rt)')
plt.plot(time,coss1,label='cos(ωt)')
plt.plot(time,sins1,label='sin(ωt)')
plt.legend()
plt.show()

plt.xlabel('t')
plt.plot(time,ert2,label='e^(rt)')
plt.plot(time,coss2,label='cos(ωt)')
plt.plot(time,sins2,label='sin(ωt)')
plt.legend()
plt.show()

plt.xlabel('x')
plt.ylabel('mode')
plt.plot(x,phi[:,0].real,label='mode1')
plt.plot(x,phi[:,1].real,label='mode2')
plt.legend()
plt.show()


#グラフの描画
fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlabel("x")
ax.set_ylabel("t")
ax.set_zlabel("f(x,t)")
ax.plot_wireframe(X, T, f.T)
plt.show()

fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlabel("x")
ax.set_ylabel("t")
ax.set_zlabel("f(x,t)")
ax.plot_wireframe(X, T, n_f.T)
plt.show()
