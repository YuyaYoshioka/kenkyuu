import math
import numpy as np
from numpy import dot, pi, sin, exp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.linalg import eigh
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm




# 各種定数の設定
xmin = 0.0
xmax = 2.0*pi
ymin = 0.0
ymax = 2.0*pi
nx = 50
ny = 50
dx = (xmax - xmin)/nx # x空間刻み
dy = (ymax - ymin)/ny # y空間刻み
cx = 1.0 # x移流速度
cy = 1.0 # y移流速度
T= 10
M = int(T/0.003)
t = np.linspace(0, T, M) # 時間
dt = t[2]-t[1] # 時間刻み
print(dt)
ax=cx*dt/2/dx
ay=cy*dt/2/dy
x=np.arange(xmin,xmax,dx)
y=np.arange(ymin,ymax,dy)
X, Y = np.meshgrid(x, y)

U=[]
# 初期値
u0 = np.zeros((ny, nx))
for ny_ in range(ny):
    for nx_ in range(nx):
        u0[ny_][nx_]=sin(ny_*dy+nx_*dx)
U.append(u0)

# 理論値
u_a = np.ones((ny,nx))
for ny_ in range(ny):
    for nx_ in range(nx):
        u_a[ny_][nx_]=sin(ny_*dy+nx_*dx-(cx+cy)*T)

# 中心差分法
for _ in range(M):
    u=np.zeros((ny, nx))
    u[1:-1,1:-1]=U[-1][1:-1,1:-1]-ax*(U[-1][1:-1,2:]-U[-1][1:-1,:-2])-ay*(U[-1][2:,1:-1]-U[-1][:-2,1:-1])
    u[0,0]=U[-1][0,0]-ax*(U[-1][0,1]-U[-1][0,-1])-ay*(U[-1][1,0]-U[-1][-1,0])
    u[0,-1]=U[-1][0,-1]-ax*(U[-1][0,0]-U[-1][0,-2])-ay*(U[-1][1,-1]-U[-1][-1,-1])
    u[-1,0]=U[-1][-1,0]-ax*(U[-1][-1,1]-U[-1][-1,-1])-ay*(U[-1][0,0]-U[-1][-2,0])
    u[-1,-1]=U[-1][-1,-1]-ax*(U[-1][-1,0]-U[-1][-1,-2])-ay*(U[-1][0,-1]-U[-1][-2,-1])
    u[0,1:-1]=U[-1][0,1:-1]-ax*(U[-1][0,2:]-U[-1][0,:-2])-ay*(U[-1][1,1:-1]-U[-1][-1,1:-1])
    u[-1,1:-1]=U[-1][-1,1:-1]-ax*(U[-1][-1,2:]-U[-1][-1,:-2])-ay*(U[-1][0,1:-1]-U[-1][-2,1:-1])
    u[1:-1,0]=U[-1][1:-1,0]-ax*(U[-1][1:-1,1]-U[-1][1:-1,-1])-ay*(U[-1][2:,0]-U[-1][:-2,0])
    u[1:-1,-1]=U[-1][1:-1,-1]-ax*(U[-1][1:-1,0]-U[-1][1:-1,-2])-ay*(U[-1][2:,-1]-U[-1][:-2,-1])
    U.append(u)





# データの平均を引く
U_n=[]
for u in U:
    V = []
    for v in u:
        for v_ in v:
            V.append(v_)
    U_n.append(V)
U_n=np.array(U_n)
U_n_T=U_n.T
u_ave = np.average(U_n_T, axis=1) # 列方向の平均
D = U_n_T - u_ave.reshape(len(u_ave), 1) # 時間平均を差し引く

# 固有値問題
R = (D.T).dot(D)
val, vec = eigh(R) # R is symmetric
# eighの戻り値は昇順なので逆順にして降順にする
val = val[::-1]
vec = vec[:, ::-1]

mode=np.arange(1,21,1)
plt.xlabel('mode')
plt.ylabel('eigenvalue')
ax = plt.gca()
ax.set_yscale('log')  # メイン:# y軸をlogスケールで描く
plt.plot(mode,val[:20],marker='.',linestyle='None')
plt.show()

# 累積寄与率
values=[]
whole=sum(val)
for n in range(20):
  value=val[:n+1]
  values.append(sum(value)/whole)
plt.xlabel('mode')
plt.ylabel('cumulative contribution rate')
plt.plot(mode, values, linestyle='None', marker='.')
#plt.show()

# 固有モード
r=2
vn = vec[:,:r]/np.sqrt(val[:r])
phi = D.dot(vn)

phi1=phi[:,0]
Phi1=phi1.reshape(ny,nx)
phi2=phi[:,1]
Phi2=phi2.reshape(ny,nx)



fig = plt.figure(figsize=(11, 7), dpi=100)
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Phi1, color='Blue', label='mode1')
ax.plot_surface(X, Y, Phi2, color='Red', label='mode2')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("eigenmode")
# plt.show()

# ROMシミュレーション
# 初期値
U0=u0.reshape(nx*ny,1)
a0 = dot(phi.T,U0 - u_ave.reshape(nx*ny,1))
A0=a0.reshape(r)
print(A0.shape)
# 平均値の勾配と分散
u_ave=u_ave.reshape(ny,nx)
uay,uax = np.gradient(u_ave,y,x)
uax=uax.reshape(nx*ny)
uay=uay.reshape(nx*ny)
# 固有モードの勾配と分散
phiy1,phix1 = np.gradient(Phi1,y,x)
phiy2,phix2 = np.gradient(Phi2,y,x)
Phix1=phix1.reshape(ny*nx)
Phiy1=phiy1.reshape(ny*nx)
Phix2=phix2.reshape(ny*nx)
Phiy2=phiy2.reshape(ny*nx)
phix=[]
phix.append(list(Phix1))
phix.append(list(Phix2))
phix=np.array(phix)
Phix=phix.T
phiy=[]
phiy.append(list(Phiy1))
phiy.append(list(Phiy2))
phiy=np.array(phiy)
Phiy=phiy.T
def lde_rom(t,a) :
  x_rhs1 = dot(phi.T, uax)
  x_rhs2 = dot((phi.T).dot(Phix), a)
  y_rhs1 = dot(phi.T, uay)
  y_rhs2 = dot((phi.T).dot(Phiy), a)
  return -cx*(x_rhs1 + x_rhs2)-cy*(y_rhs1 + y_rhs2)
sol_a = solve_ivp(lde_rom,[0,T], A0, method='Radau', t_eval=t)
a =sol_a.y
u_rom = u_ave.reshape(ny*nx,1) + np.dot(phi, a)
u_rom=u_rom.T

a1=a[0][:int(M/4)]
print(max(a1))
print(min(a1))
average1=(max(a1)+min(a1))/2
print(average1)
print(max(a1)-average1)
print(min(a1)-average1)
print(np.where(a1==max(a1)))
a2=a[1][:int(M/4)]
average2=(max(a2)+min(a2))/2
print(average2)
print(max(a2)-average2)
print(min(a2)-average2)
print(max(a2))
print(min(a2))

plt.xlabel('t')
plt.ylabel('a(t)')
plt.plot(t,a[0],label='a1')
plt.plot(t,a[1],label='a2')
# plt.legend()
# plt.show()

U_rom=[]
for u in u_rom:
    UU=u.reshape(ny,nx)
    U_rom.append(UU)


# 誤差
errors1=u_a-U_rom[-1]
error1=0
for errors_ in errors1:
    for errors__ in errors_:
        error1+=abs(errors__)
error1/=(nx*ny)
print('ROMと理論解')
print(error1)
errors2=U[-1]-U_rom[-1]
error2=0
for errors_ in errors2:
    for errors__ in errors_:
        error2+=abs(errors__)
error2/=(nx*ny)
print('ROMと差分法')
print(error2)
errors3=u_a-U[-1]
error3=0
for errors_ in errors3:
    for errors__ in errors_:
        error3+=abs(errors__)
error3/=(nx*ny)
print('差分法と理論解')
print(error3)


fig = plt.figure(figsize=(11, 7), dpi=100)
ax = fig.gca(projection='3d')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("u(x, y)")
ax.plot_surface(X, Y, U_rom[-1], cmap=cm.viridis)
# plt.show()

# アニメ更新用の関数
def update_func(i):
  # 前のフレームで描画されたグラフを消去
  ax.clear()
  ax.set_xlabel("x")
  ax.set_ylabel("y")
  ax.set_zlabel("u(x, y)")
  ax.plot_surface(X, Y, U[i], cmap=cm.viridis)


ani = animation.FuncAnimation(fig, update_func,  interval=1, repeat=True, save_count=int(M))
# アニメーションの保存
#ani.save('test9.gif', writer="imagemagick")

# 表示
#plt.show()
