import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy import dot
from numpy import pi, cos, sqrt,exp,sin
from scipy.linalg import eigh
from scipy.integrate import solve_ivp

# データの生成
L = 2.0*pi # 空間の長さ
T = 10 # 計算時間
N = 256 # 空間分割数
dt =1.0e-2 #　時間刻み
x = np.linspace(0, L, N) #空間
t = np.arange(0, T, dt) # 時間

# パラメータと初期条件
dd = 1
mu = 1.0
sigma = 0.2
u0 = sin(x)

# 線形拡散方程式
def lde(t,u) :
  ux = np.gradient(u,x)
  return dd*ux

# 数値積分の実行
sol = solve_ivp(lde,[0,T], u0, method='Radau', t_eval=t)
u = sol.y

plt.xlabel('x',fontsize=25)
plt.ylabel('u',fontsize=25)
plt.plot(x,u[:,0],label='t=0')
plt.plot(x,u[:,199],label='t=2')
plt.plot(x,u[:,399],label='t=4')
plt.plot(x,u[:,599],label='t=6')
plt.plot(x,u[:,799],label='t=8')
plt.plot(x,u[:,-1],label='t=10')
plt.legend(fontsize=25)
plt.tick_params(labelsize=25)
plt.show()

# データ行列
u_ave = np.average(u, axis=1) # 列方向の平均
D = u - u_ave.reshape(len(u_ave), 1) # 時間平均を差し引く

# 固有値問題
R = (D.T).dot(D)
val, vec = eigh(R) # R is symmetric

# eighの戻り値は昇順なので逆順にして降順にする
val = val[::-1]
vec = vec[:, ::-1]

plt.xlabel('mode',fontsize=12.5)
plt.ylabel('eigenvalues',fontsize=12.5)
mode=np.arange(1,21,1)
plt.plot(mode,val[:20],linestyle='None',marker='.')
plt.tick_params(labelsize=12.5)
ax = plt.gca()
ax.set_yscale('log')
plt.show()

values=[]
whole=sum(val)
for n in range(20):
    values.append(sum(val[:n+1])/whole)

plt.xlabel('mode',fontsize=12.5)
plt.ylabel('eigenvalues',fontsize=12.5)
mode=np.arange(1,21,1)
plt.plot(mode,values,linestyle='None',marker='.')
plt.tick_params(labelsize=12.5)
plt.show()

# 固有モード
r = 3
vn = vec[:,:r]/sqrt(val[:r])
phi = D.dot(vn)


plt.xlabel('x',fontsize=12)
plt.ylabel('eigenmodes',fontsize=12)
plt.plot(x,phi[:,0],label='mode1')
plt.plot(x,phi[:,1],label='mode2')
plt.plot(x,phi[:,2],label='mode3')
plt.tick_params(labelsize=12)
plt.legend(fontsize=12)
plt.show()