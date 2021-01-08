import math
import numpy as np
from numpy import dot, exp, sin
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.linalg import eigh
from scipy.integrate import solve_ivp

# 各種定数の設定
dt = 0.0008# 時間刻み
xmin = 0.0
xmax = 2.0*math.pi
N=40
dx = (xmax - xmin)/N # 空間刻み
c = 1.0 # 移流速度
ν = 0 # 拡散係数
T=4
M = int(T/dt)
a=-c*dt/2/dx


x=np.arange(xmin,xmax,dx)
t = np.arange(0, T, dt) # 時間



U=[]

#　初期値
u0 = np.zeros(N)
u0[int(2/dx):int(4/dx)]=2
U.append(u0)

plt.plot(x,u0)
plt.show()

# 差分法
for _ in range(int(M)+1):
  u=[0 for _ in range(N)]
  u[0] = U[-1][0]+a*(2*U[-1][1]+3*U[-1][0]-6*U[-1][-1]+U[-1][-2])
  u[1] = U[-1][1]+a*(2*U[-1][2]+3*U[-1][1]-6*U[-1][0]+U[-1][-1])
  for j in range(2,N-1):
    u[j] = U[-1][j] + a * (2 * U[-1][j+1] + 3 * U[-1][j] - 6 * U[-1][j-1] + U[-1][j-2])
  u[N-1] = U[-1][N-1]+a*(2*U[-1][0]+3*U[-1][N-1]-6*U[-1][N-2]+U[-1][N-3])
  U.append(u)

plt.plot(x,U[int(M/8)])
plt.show()

plt.plot(x,U[-1])
plt.show()

fig = plt.figure(figsize=(6,4))
ax = fig.add_subplot(1,1,1)

# アニメ更新用の関数
def update_func(i):
  # 前のフレームで描画されたグラフを消去
  ax.clear()

  ax.plot(x, U[i], color='blue')
  ax.scatter(x, U[i], color='blue')
  # 軸の設定
  ax.set_ylim(-1, 7)
  # 軸ラベルの設定
  ax.set_xlabel('x', fontsize=12)
  ax.set_ylabel('u', fontsize=12)
  # サブプロットタイトルの設定
  ax.set_title('Time: ' + '{:.2f}'.format(dt*i))

ani = animation.FuncAnimation(fig, update_func,  interval=1, repeat=True, save_count=int(M))
# アニメーションの保存
# ani.save('test7.gif', writer="imagemagick")

# 表示
plt.show()


V=np.array(U)
W=V.T

# データ行列
u_ave = np.average(W, axis=1) # 列方向の平均
print(u_ave.shape)
D = W - u_ave.reshape(len(u_ave), 1) # 時間平均を差し引く

# 固有値問題
R = (D.T).dot(D)
val, vec = eigh(R) # R is symmetric
# eighの戻り値は昇順なので逆順にして降順にする
val = val[::-1]
vec = vec[:, ::-1]
print(min(val))

print(val[:100])
# 累積寄与率
values=[]

whole=sum(val)
for n in range(40):
  value=val[:n+1]
  values.append(sum(value)/whole)

mode=np.arange(1,41,1)
plt.xlabel('mode')
plt.ylabel('eigenvalues')
plt.plot(mode,val[:40],linestyle='None',marker='.')
ax = plt.gca()
ax.set_yscale('log')  # メイン: y軸をlogスケールで描く
plt.show()

plt.xlabel('mode')
plt.ylabel('cumulative contribution rate')
plt.plot(mode,values,linestyle='None',marker='.')
plt.show()


# 固有モード
r=20
vn = vec[:,:r]/np.sqrt(val[:r])
phi = D.dot(vn)



# ROMシミュレーション

# 初期値
a0 = (u0 - u_ave).dot(phi)
print(a0.shape)
# 平均値の勾配と分散
uax = np.gradient(u_ave,x)
uaxx = np.gradient(uax,x)

# 固有モードの勾配と分散
phix = np.gradient(phi,x, axis=0)
phixx = np.gradient(phix,x, axis=0)

print(phix.shape)

def lde_rom(t,a) :
  rhs1 = dot(phi.T, uax)
  rhs2 = dot((phi.T).dot(phix), a)
  n_rhs1 = dot(phi.T, uaxx)
  n_rhs2 = dot((phi.T).dot(phixx), a)
  return -c*(rhs1 + rhs2)+ν*(n_rhs1 + n_rhs2)


sol_a = solve_ivp(lde_rom,[0,T], a0, method='Radau', t_eval=t)
a =sol_a.y
u_rom = u_ave.reshape(len(u_ave),1) + np.dot(phi, a)



u_rom=u_rom.T


plt.xlabel('x')
plt.ylabel('eigenmode')
plt.plot(x,phi[:,0],label='mode1')
plt.plot(x,phi[:,1],label='mode2')
plt.legend()
plt.show()



plt.xlabel('x')
plt.ylabel('eigenmode')
plt.plot(x,phi[:,2],label='mode3')
plt.plot(x,phi[:,3],label='mode4')
plt.legend()
plt.show()


plt.xlabel('x')
plt.ylabel('eigenmode')
plt.plot(x,phi[:,4],label='mode5')
plt.plot(x,phi[:,5],label='mode6')
plt.legend()
plt.show()

print(len(U))
print(len(u_rom))

plt.xlabel('x')
plt.ylabel('u')
plt.plot(x,U[0],label='original(t=0)',linestyle='-.')
plt.plot(x,u_rom[0],label='u_rom(t=0)',linestyle=':')
plt.legend()
plt.show()

plt.xlabel('x')
plt.ylabel('u')
plt.plot(x,U[int(M/2)],label='original(t=2)')
plt.plot(x,u_rom[int(M/2)],label='u_rom(t=2)')
plt.legend()
plt.show()

plt.xlabel('x')
plt.ylabel('u')
plt.plot(x,U[-1],label='original(t=4)')
plt.plot(x,u_rom[-1],label='u_rom(t=4)')
plt.legend()
plt.show()


plt.xlabel('t')
plt.ylabel('a(t)')
plt.plot(t,a[0],label='a1')
plt.plot(t,a[1],label='a2')
plt.legend()
plt.show()




plt.xlabel('t')
plt.ylabel('a(t)')
plt.plot(t,a[2],label='a3')
plt.plot(t,a[3],label='a4')
plt.legend()
plt.show()





plt.xlabel('t')
plt.ylabel('a(t)')
plt.plot(t,a[4],label='a5')
plt.plot(t,a[5],label='a6')
plt.legend()
plt.show()

fig = plt.figure(figsize=(6,4))
ax = fig.add_subplot(1,1,1)

# アニメ更新用の関数
def update_func(i):
  # 前のフレームで描画されたグラフを消去
  ax.clear()

  ax.plot(x, U[i], color='blue')
  ax.scatter(x, U[i], color='blue')
  # 軸の設定
  ax.set_ylim(-1.1, 1.1)
  # 軸ラベルの設定
  ax.set_xlabel('x', fontsize=12)
  ax.set_ylabel('u', fontsize=12)
  # サブプロットタイトルの設定
  ax.set_title('Time: ' + '{:.2f}'.format(dt*i))

ani = animation.FuncAnimation(fig, update_func,  interval=1, repeat=True, save_count=int(M))
# アニメーションの保存
# ani.save('test7.gif', writer="imagemagick")

# 表示
plt.show()




