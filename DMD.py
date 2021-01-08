from numpy import dot, multiply, diag, power
from numpy import pi, exp, sin, cos, cosh, tanh, real, imag
from numpy.linalg import inv, eig, pinv
from scipy.linalg import svd, svdvals
from scipy.integrate import odeint, ode, complex_ode
from warnings import warn
from scipy.linalg import eigh

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import pi,exp,cos,tanh

# 定義
x = np.linspace(-pi, pi, 100) #空間軸
t = np.linspace(0, 6*pi, 300) #時間軸
dt=t[2]-t[1] # 微小時間dtの定義
X, T = np.meshgrid(x, t) # グラフ描画のためにX,Tをメッシュ化
f1=(X)*exp(1j*T) # x*exp(it)の定義、虚数iがjであることに注意！
f2=cos(X)*exp(2j*T) # cos(x)*exp(2it)の定義
f3=tanh(X)*exp(3j*T) # tanh(x)*exp(3jt)の定義
f=f1+f2+f3 #元の関数
f_r=f.real # 元の関数の実部

#グラフの描画（ｆ）
fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlabel("x")
ax.set_ylabel("t")
ax.set_zlabel("f(x,t)")
ax.plot_wireframe(X, T, f)
plt.show()

from scipy.linalg import svd

n_f=f.T #fの転置をとる
X0=n_f[:,:-1] # 式(3)のX
X1=n_f[:,1:] # 式(4)のXダッシュ

# X0の特異値分解
mode=np.arange(1,21,1) # モード数の定義
U,Sig,Vh2 = svd(X0, False) # 式(5)の計算、UがU、SigがΣ、Vh2がV^*

#グラフの描画（固有値）
plt.title('固有値', fontname="MS Gothic")
plt.xlabel('mode')
plt.ylabel('eigenvalues')
plt.plot(mode,Sig[:20],linestyle='None',marker='.')
plt.show()

# 累積寄与率の計算
values=[] # 累積寄与率の定義
whole=sum(Sig) # 固有値の合計を計算
for n in range(20): 
  value=Sig[:n+1]
  values.append(sum(value)/whole)

# グラフの描画（累積寄与率）
plt.title('累積寄与率', fontname="MS Gothic")
plt.xlabel('mode')
plt.ylabel('cumulative contribution rate')
plt.plot(mode,values,linestyle='None',marker='.')
plt.show()

from numpy import dot,eye
from numpy.linalg import eig

# 式(12)の計算
r = 3 # モード数
Sig_r = eye(r) # r×rの単位行列を作成
V = Vh2.conj().T # Vh2の随伴行列、Vが式(6)におけるVとなる
V_r = V[:,:r] # 式(12)におけるVr(式(12)においてAをr×rの行列に
              # するために次元削減)
for a in range(r):
    Sig_r[a][a] = 1/Sig[a] # 式(12)におけるΣr^(-1)
U_n = U[:,:r] # 式(12)におけるUr(式(12)においてAをr×rの行列に
                # するために次元削減)
U_r = U_n.conj().T # U_nの随伴行列、U_rが式(12)におけるUr^*
n_A=dot(dot(dot(U_r,X1),V_r),Sig_r) # 式(12)の計算

# n_Aの固有値、固有ベクトルの計算
lam,W=eig(n_A) # 式(14)におけるΛがlam、WがW

# 式(22)の計算
phi=dot(dot(dot(X1,V_r),Sig_r),W)

# グラフの描画（固有モード）
plt.title('固有モード', fontname="MS Gothic")
plt.xlabel('x')
plt.ylabel('eigenmode')
plt.plot(x,phi[:,0].real,label='mode1')
plt.plot(x,phi[:,1].real,label='mode2')
plt.plot(x,phi[:,2].real,label='mode3')
plt.legend()
plt.show()

from numpy.linalg import pinv
from numpy import power

# 式(31)の計算
rev_phi = pinv(phi) # Φの擬似逆行列の計算
b = dot(rev_phi, X0[:,0]) # 式(31)のΦ^†x_1の計算
Psi = np.zeros([r, len(t)], dtype='complex') # 式(31)のTを定義
                                             # モード数r×時間方向メッシュ数
for i,_t in enumerate(t):
    n_lam = power(lam, _t/dt) # 式(31)のΛ^(t/Δt)の計算
    Sig_r = eye(r) # r×rの単位行列を作成
    for r_ in range(r):
        Sig_r[r_][r_] = n_lam[r_] # 式(30)のΛ^(t/Δt)を定義
    Psi[:,i] = dot(Sig_r, b) #　式(31)の計算

# グラフの描画（時間発展関数）
plt.title('時間発展関数',fontname="MS Gothic")
plt.xlabel('t')
time=np.linspace(0,2*pi,100)
plt.plot(time,Psi[0][:100].real,label='mode1')
plt.plot(time,Psi[1][:100].real,label='mode2')
plt.plot(time,Psi[2][:100].real,label='mode3')
plt.legend()
plt.show()

# 式(32)の計算
n_f = dot(phi, Psi)

#グラフの描画
fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlabel("x")
ax.set_ylabel("t")
ax.set_zlabel("f(x,t)")
ax.plot_wireframe(X, T, f)
plt.title('元のデータ',fontname="MS Gothic")
plt.show()

fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlabel("x")
ax.set_ylabel("t")
ax.set_zlabel("f(x,t)")
ax.plot_wireframe(X, T, n_f.T)
plt.title('動的モード分解',fontname="MS Gothic")
plt.show()
