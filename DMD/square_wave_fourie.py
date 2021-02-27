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
N =300
dx = (xmax - xmin)/N # 空間刻み
c = 1.0 # 移流速度
ν = 0 # 拡散係数
T=4
M = int(T/dt)
α=c*dt/dx
β=ν*dt/dx/dx

t = np.arange(0, T, dt) # 時間

# 関数を定義
y = lambda x: 2 if x>=2 and x<=4 else 0

# yを0からpiまで数値積分
a0, er = integrate.quad(y, 0, 2*pi)
a0/=pi*2
r=57
an=[]
bn=[]

for n in range(1,r+1):
    ya= lambda x: cos(n*x)*2 if x>=2 and x<=4 else 0
    yb= lambda x: sin(n*x)*2 if x>=2 and x<=4 else 0
    a, er = integrate.quad(ya, 0, 2 * pi)
    b, er = integrate.quad(yb, 0, 2 * pi)
    an.append(a/pi)
    bn.append(b/pi)

x=np.arange(xmin,xmax,dx)
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
a=-c*dt/2/dx

print(ν*dt/dx/dx)

x=np.arange(xmin,xmax,dx)
t = np.arange(0, T, dt) # 時間



U=[]

#　初期値
u0 = np.zeros(N)
u0[int(2/dx+1):int(4/dx+1)]=2
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

# 累積寄与率
values=[]

whole=sum(val)
for n in range(100):
  value=val[:n+1]
  values.append(sum(value)/whole)

mode=np.arange(1,61,1)
plt.xlabel('mode')
plt.ylabel('eigenvalues')
plt.plot(mode,val[:60],linestyle='None',marker='.')
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
for n in range(int(2/dx+1)):
    error+=abs(ff[n])
for n in range(int(2/dx+1),int(4/dx+1)):
    error+=abs(2-ff[n])
for n in range(int(4/dx+1),N):
    error+=abs(ff[n])
print('フーリエと理論解')
print(error/N)
error=0
for n in range(int(2/dx+1)):
    error+=abs(u_rom[0][n])
for n in range(int(2/dx+1),int(4/dx+1)):
    error+=abs(2-u_rom[0][n])
for n in range(int(4/dx+1),N):
    error+=abs(u_rom[0][n])
print('ROMと理論解')
print(error/N)

plt.xlabel('x')
plt.ylabel('u')
plt.plot(x,ff,label='fourie(t=0)',linestyle='-.')
plt.plot(x,u_rom[0],label='u_rom(t=0)',linestyle=':')
plt.plot(x,u0)
plt.legend()
plt.show()

