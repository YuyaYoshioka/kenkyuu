import math
import numpy as np
from numpy import dot, exp, pi, cos, sin
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.linalg import eigh
from scipy.integrate import solve_ivp
from scipy import integrate


# 各種定数の設定
dt = 0.0008 # 時間刻み
xmin = 0.0
xmax = 2.0*math.pi
N =600
dx = (xmax - xmin)/N # 空間刻み
c = 1.0 # 移流速度
ν = 0 # 拡散係数
T=4
M = int(T/dt)
α=c*dt/dx
β=ν*dt/dx/dx


t = np.arange(0, T, dt) # 時間


mu = 1.0
sigma = 0.2


# 関数を定義
y = lambda x: 0 if x==0 or x==round(dx*(N-1),8) else exp(-(x-mu)**2 / sigma**2)

# yを0からpiまで数値積分
a0, er = integrate.quad(y, dx, dx*(N-2))
a0/=pi*2
r=23
an=[]
bn=[]

for n in range(1,r+1):
    ya= lambda x: 0 if x==0 or x==round(dx*(N-1),8) else exp(-(x-mu)**2 / sigma**2)*cos(n*x)
    yb= lambda x: 0 if x==0 or x==round(dx*(N-1),8) else exp(-(x-mu)**2 / sigma**2)*sin(n*x)
    a, er = integrate.quad(ya, dx, dx*(N-2))
    b, er = integrate.quad(yb, dx,dx*(N-2))
    an.append(a/pi)
    bn.append(b/pi)

x=np.arange(xmin,xmax,dx)
print(x)
print(max(x))
print(round(dx*(N-1),8))
ff=np.full(len(x),a0)
n=1
for a in an:
    ff+=a*cos(n*x)
    n+=1
n=1
for b in bn:
    ff+=b*sin(n*x)
    n+=1
plt.plot(x,ff)
plt.show()


# 各種定数の設定
dt = 0.0008 # 時間刻み
xmin = 0.0
xmax = 2.0*math.pi
dx = (xmax - xmin)/N # 空間刻み
c = 1.0 # 移流速度
ν = 0 # 拡散係数
T=4
M = int(T/dt)
α=c*dt/dx
β=ν*dt/dx/dx

print(ν*dt/dx/dx)

x=np.arange(xmin,xmax,dx)
t = np.arange(0, T, dt) # 時間



U=[]

#　初期値
u0 = exp(-(x-mu)**2 / sigma**2)
u0[0]=0
u0[-1]=0
U.append(u0)
a=-2*c*dt/3/dx
b=c*dt/12/dx
plt.plot(x,u0)
plt.show()

# 差分法
for _ in range(int(M)+1):
    u = [0 for _ in range(N)]
    u[0] = 0
    u[1] = U[-1][1] + a * (U[-1][2] - U[-1][0]) + b * (U[-1][3] - U[-1][-1])
    for j in range(2, N - 2):
        u[j] = U[-1][j] - (U[-1][j + 1] - U[-1][j - 1]) * c * dt / 2 / dx + dt * ν * (
                    U[-1][j - 1] - 2 * U[-1][j] + U[-1][j + 1]) / dx / dx
    u[N - 2] = U[-1][N - 2] + a * (U[-1][N - 1] - U[-1][N - 3]) + b * (U[-1][0] - U[-1][N - 4])
    u[N - 1] = 0
    U.append(u)

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
plt.plot(values,linestyle='None',marker='.')
plt.show()


# 固有モード
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

error=0
for n in range(N):
    error+=abs(u0[n]-ff[n])
print('フーリエと理論解')
print(error/N)
error=0
for n in range(N):
    error+=abs(u0[n]-u_rom[0][n])
print('ROMと理論解')
print(error/N)

plt.xlabel('x')
plt.ylabel('u')
plt.plot(x,ff,label='fourie',linestyle='-.')
plt.plot(x,u_rom[0],label='u_rom',linestyle=':')
plt.plot(x,u0,label='theoretical',linestyle='-.')
plt.legend()
plt.show()

