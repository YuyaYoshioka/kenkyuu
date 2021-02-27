import math
import numpy as np
from numpy import dot, exp, sin
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.integrate import solve_ivp

import numpy as np
import matplotlib.pyplot as plt

# データの生成
L = 2.0 # 空間の長さ
T = 10 # 計算時間
N = 250 # 空間分割数
dt = 0.01 #　時間刻み
N_T = int(T / dt) # 時間分割数
dx = L / N # 空間刻み
x = np.linspace(0, L, N) #空間
t = np.arange(0, T, dt) # 時間

# パラメータと初期条件
dd = 0.003 # 拡散係数を定義
mu = 1.0 # 初期条件のガウス分布がx=1で極大となる
sigma = 0.2 # ガウス分布の分散
u0 = np.exp(-(x-mu)**2 / sigma**2) # 初期条件のガウス分布を定義

# 差分法による熱伝導方程式の解法
U=[u0] # 差分法の解の配列、初期条件にガウス分布
for _ in range(N_T-1): # 時間分割数の範囲のfor文
    u = U[-1] # Uの最後の値を取得
    new_u = [] # 一つ先の時間のuを定義
    new_u.append(u[0] + (u[1] -2*u[0] +u[-1])*dt*dd/dx**2) # x0を定義、周期境界条件より、
                                                           # u[-1]=u[0]であることに注意
    for n in range(1,N-1): # 空間分割数の範囲のfor文、始点と終点を除いていることに注意
        new_u.append(u[n] + (u[n+1] -2*u[n] +u[n-1])*dt*dd/dx**2) # 式(12)の計算
    new_u.append(u[-1] + (u[0] -2*u[-1] +u[-2])*dt*dd/dx**2) # xNを定義、周期境界条件より、
                                                             # u[0]=u[-1]であることに注意
    U.append(new_u) # 差分法の解の配列に新たな解を加えていく

# グラフの描画
plt.plot(x,U[0])
plt.plot(x,U[N_T//5])
plt.plot(x,U[N_T*2//5])
plt.plot(x,U[N_T*3//5])
plt.plot(x,U[N_T*4//5])
plt.plot(x,U[-1],label='t=10')
plt.title('熱伝導方程式', fontname="MS Gothic") # グラフタイトル
plt.legend(["理論解","差分法"], prop={"family":"MS Gothic"})
plt.legend()
plt.show()


from scipy.linalg import eigh

# 行列データの作成
np_U = np.array(U) # Uをnumpyの配列に変換
U = np_U.T # 今回Uは時間方向のメッシュ数×空間方向のメッシュ数形の行列
           # であるため、Uの転置をとる
U_ave = np.average(U, axis=1) # 列方向の平均
D = U - U_ave.reshape(len(U_ave), 1) # 時間平均を差し引く

# 固有値問題
R = np.dot(D.T,D) # X^TXの計算
val, vec = eigh(R) # eighを利用して、Rの固有値および固有値ベクトルを計算
                   # 式(14)におけるVがvec、Λがval

# eighの戻り値は昇順なので逆順にして降順にする
val = val[::-1]
vec = vec[:, ::-1]

# 累積寄与率の計算
values=[] # 累積寄与率の配列
whole=sum(val) # 固有値の和
for n in range(40):
  value=val[:n+1]
  values.append(sum(value)/whole)

# 固有値Λのグラフの描画
mode=np.arange(1,41,1)
plt.xlabel('mode')
plt.ylabel('eigenvalues')
plt.plot(mode,val[:40],linestyle='None',marker='.')
ax = plt.gca()
ax.set_yscale('log')  # y軸をlogスケールで描く
plt.title('固有値', fontname="MS Gothic")
plt.show()

# 累積寄与率のグラフの描画
plt.xlabel('mode')
plt.ylabel('cumulative contribution rate')
plt.plot(mode,values,linestyle='None',marker='.')
plt.title('累積寄与率', fontname="MS Gothic")
plt.show()

# 固有モード
m=3 # 採用するモード数、今回はm=3
vn = vec[:,:m]/np.sqrt(val[:m]) # 式(14)のVΛ^(-1/2)を計算
phi = np.dot(D,vn) # 式(14)の固有モードUを計算

#固有モードのグラフの描画
plt.title('固有モード',fontname="MS Gothic")
plt.plot(x,phi[:,0],label='mode1')
plt.plot(x,phi[:,1],label='mode2')
plt.plot(x,phi[:,2],label='mode3')
plt.legend()
plt.show()

# 時間に関するモードの計算
A = np.dot(phi.T,D) # 式(20)の計算、Dはすでに時間平均を
                    # 引いてある事に注意

# 時間に関するモードのグラフの描画
plt.title('時間モード',fontname="MS Gothic")
plt.plot(t,A[0],label='mode1')
plt.plot(t,A[1],label='mode2')
plt.plot(t,A[2],label='mode3')
plt.xlabel('t')
plt.ylabel('A')
plt.legend()
plt.show()

# 式(15)の計算
u = U_ave.reshape(len(U_ave),1) + np.dot(phi, A)
u_T = u.T # uの転置

# 元のデータと主成分解析によるデータのグラフの描画
# 元のデータの描画
plt.plot(x,U.T[0],linestyle='--')
plt.plot(x,U.T[N_T//5],linestyle='--')
plt.plot(x,U.T[N_T*2//5],linestyle='--')
plt.plot(x,U.T[N_T*3//5],linestyle='--')
plt.plot(x,U.T[N_T*4//5],linestyle='--')
plt.plot(x,U.T[-1],linestyle='--')
# 主成分解析によるデータの描画
plt.plot(x,u_T[0],linestyle=':')
plt.plot(x,u_T[N_T//5],linestyle=':')
plt.plot(x,u_T[N_T*2//5],linestyle=':')
plt.plot(x,u_T[N_T*3//5],linestyle=':')
plt.plot(x,u_T[N_T*4//5],linestyle=':')
plt.plot(x,u_T[-1],linestyle=':')
plt.title('元のデータおよび主成分分析によるデータ', fontname="MS Gothic")
plt.legend(['t=0_ori','t=2_ori','t=4_ori','t=6_ori','t=8_ori','t=10_ori','t=0_pca','t=2_pca','t=4_pca','t=6_pca','t=8_pca','t=10_pca'], prop={"family":"MS Gothic"})
plt.show()
