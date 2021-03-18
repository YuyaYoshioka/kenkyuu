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

# 動的モード分解するファイルを指定
path = os.getcwd()
file_list = glob.glob('04P????.txt')

u=[]

for filename in file_list:
    with open(filename) as input:
        input_=input.read()
        u.append(input_.split())

UU1=u[0]

R=[]
U=[]
for u_ in u:
    number=1
    suuji=2
    n_u=[]
    while number < len(u_):
        n_u.append(float(u_[number]))
        R.append(float(u_[suuji]))
        suuji+=3
        number+=3
    U.append(n_u)


U_n=np.array(U)
U_n_T=U_n.T
X0=U_n_T[:,:-1]
X1=U_n_T[:,1:]


mode=np.arange(1,21,1)
U,Sig,Vh2 = svd(X0, False)

# 固有値を計算
plt.xlabel('mode')
plt.ylabel('eigenvalues')
plt.plot(mode,Sig[:20],linestyle='None',marker='.')
# plt.show()

values=[]

# 累積寄与率を計算
whole=sum(Sig)
for n in range(20):
  value=Sig[:n+1]
  values.append(sum(value)/whole)

plt.xlabel('mode')
plt.ylabel('cumulative contribution rate')
plt.plot(mode,values,linestyle='None',marker='.')
# plt.show()


# モード数を指定
r=10


Sig_r=np.eye(r)
V = Vh2.conj().T[:,:r]
for a in range(r):
    Sig_r[a][a]=1/Sig[a]
U_r=U[:,:r]
n_A=dot(dot(dot(np.conjugate(U_r.T),X1),V[:,:r]),Sig_r)

lam,W=np.linalg.eig(n_A)

phi=dot(dot(dot(X1,V[:,:r]),Sig_r),W)

t = np.linspace(0, 39, 40)
dt=1

b = dot(pinv(phi), X0[:,0])
Psi = np.zeros([r, len(t)], dtype='complex')
for i,_t in enumerate(t):
    Psi[:,i] = multiply(power(lam, _t/dt), b)

n_f = dot(phi, Psi)
f=U_n_T


#　成長率及び周波数を計算
tans=[]
for r_ in range(r):
    tan=[]
    num=0
    for n in range(len(Psi[0])-1):
        if r_ % 2 == 0:
            if math.atan(Psi[r_][n - 1].imag / Psi[r_][n - 1].real) > 0 and math.atan(
                    Psi[r_][n].imag / Psi[r_][n].real) < 0:
                num += pi
        else:
            if math.atan(Psi[r_][n - 1].imag / Psi[r_][n - 1].real) < 0 and math.atan(
                    Psi[r_][n].imag / Psi[r_][n].real) > 0:
                num -= pi
        tan.append(math.atan(Psi[r_][n].imag / Psi[r_][n].real)+num)
    tans.append(tan)
for n in range(0,len(tans),2):
    fig=plt.figure()
    plt.xlabel('t')
    omega=Decimal(str(abs(tans[n][20]-tans[n][0])/19)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    plt.plot(t[:20],tans[n][:20],label='mode'+str(n+1))
    plt.plot(t[:20],tans[n+1][:20],label='mode'+str(n+2))
    plt.text(12,0,'omega='+str(omega),fontsize=14)
    plt.legend()
    fig.savefig("img"+str(n+1)+".png")

ganmas=[]
for r_ in range(r):
    ganma=[]
    for n in range(len(Psi[0])):
        if Psi[r_][n].real**2+Psi[r_][n].imag**2 == 0:
            break
        ganma.append(math.log(Psi[r_][n].real**2+Psi[r_][n].imag**2)/2)
    ganmas.append(ganma)
for n in range(0,len(ganmas),2):
    fig=plt.figure()
    ganma=Decimal(str((ganmas[n][20]-ganmas[n][0])/19)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    plt.xlabel('t')
    plt.plot(t[:20],ganmas[n][:20],label='mode'+str(n+1))
    plt.plot(t[:20],ganmas[n+1][:20],label='mode'+str(n+2))
    plt.text(12,ganmas[n][10],'ganma='+str(ganma),fontsize=14)
    plt.legend()
    fig.savefig("img"+str(n+20)+".png")

# plt.show()

# 動的モード分解後のモードのファイルを出力

for r_ in range(0,r):
    path_w = path+'/'+str(r_)+'.txt'
    number=1
    suuji=0
    while number < len(UU1):
        UU1[number]=phi[:,r_][suuji]
        suuji+=1
        number+=3

    Phi=[]
    number=0
    for u_ in UU1:
        if number % 3 ==1:
            a='{:E}'.format(float(u_))
            b=str(a)
            c=b.replace('.','')
            Phi.append(c)
        else:
            a=str(u_)
            Phi.append(a)
        number+=1

    Out=[]
    number=0
    while number<len(Phi):
        if Phi[number + 1][0]== '-':
            if Phi[number + 2]== '149.063':
                Out.append(
                    ' ' + str(Phi[number]) + ' ' + '-0.' + str(Phi[number + 1][1:4]) + str(Phi[number + 1][-4:-2]) + '0'+str(
                        int(str(Phi[number + 1][-2:])) - 1) + '  ' + str(Phi[number + 2]))
                Out.append(' ')
            else:
                Out.append(' ' + str(Phi[number]) + ' ' + '-0.' + str(Phi[number + 1][1:4]) + str(Phi[number + 1][-4:-2]) + '0'+str(int(str(Phi[number + 1][-2:])) - 1) + '  ' + str(Phi[number + 2]))
        else:
            if Phi[number + 2]== '149.063':
                Out.append(' ' + str(Phi[number]) + '  ' + '0.' + str(Phi[number + 1][:3]) + str(Phi[number + 1][-4:-2]) +'0' +str(int(str(Phi[number + 1][-2:])) - 1) + '  ' + str(Phi[number + 2]))
                Out.append(' ')
            else:
                Out.append(' ' + str(Phi[number]) + '  ' + '0.' + str(Phi[number + 1][:3]) + str(Phi[number + 1][-4:-2]) +'0' +str(int(str(Phi[number + 1][-2:])) - 1) + '  ' + str(Phi[number + 2]))
        number+=3
    with open(path_w, mode='w') as f:
        f.write('\n'.join(Out))
