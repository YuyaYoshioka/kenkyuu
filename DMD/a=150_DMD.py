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

print(R[:100])

U_n=np.array(U)
U_n_T=U_n.T
X0=U_n_T[:,:-1]
X1=U_n_T[:,1:]


mode=np.arange(1,21,1)
U,Sig,Vh2 = svd(X0, False)

print(Sig)
plt.xlabel('mode')
plt.ylabel('eigenvalues')
plt.plot(mode,Sig[:20],linestyle='None',marker='.')
# plt.show()

values=[]

whole=sum(Sig)
for n in range(20):
  value=Sig[:n+1]
  values.append(sum(value)/whole)

plt.xlabel('mode')
plt.ylabel('cumulative contribution rate')
plt.plot(mode,values,linestyle='None',marker='.')
# plt.show()



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

R=np.array([0.781, 2.344, 3.906, 5.469, 7.031, 8.594, 10.156, 11.719, 13.281, 14.844, 16.406, 17.969, 19.531, 21.094, 22.656, 24.219, 25.781, 27.344, 28.906, 30.469, 32.031, 33.594, 35.156, 36.719, 38.281, 39.844, 41.406, 42.969, 44.531, 46.094, 47.656, 49.219, 50.781, 52.344, 53.906, 55.469, 57.031, 58.594, 60.156, 61.719, 63.281, 64.844, 66.406, 67.969, 69.531, 71.094, 72.656, 74.219, 75.781, 77.344, 78.906, 80.469, 82.031, 83.594, 85.156, 86.719, 88.281, 89.844, 91.406, 92.969, 94.531, 96.094, 97.656, 99.219, 100.781, 102.344, 103.906, 105.469, 107.031, 108.594, 110.156, 111.719, 113.281, 114.844, 116.406, 117.969, 119.531, 121.094, 122.656, 124.219, 125.781, 127.344, 128.906, 130.469, 132.031, 133.594, 135.156, 136.719, 138.281, 139.844, 141.406, 142.969, 144.531, 146.094, 147.656, 149.219])


fig = plt.figure()
for r_ in range(0,6,2):
    Rs = []
    for n in range(len(R)):
        rs=0
        for n_ in range(n,len(phi[:,r_]),len(R)):
            rs+=abs(phi[:,r_][n_])
        Rs.append(rs/(len(phi[:,r_])//len(R)))
    plt.rcParams["font.size"] = 14
    plt.tight_layout()
    plt.xlabel('r')
    plt.plot(R,Rs,label='m='+str(r_+1)+','+str(r_+2))
    plt.legend()
fig.savefig("R.png")
