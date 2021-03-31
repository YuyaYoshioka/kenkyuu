import numpy as np
import matplotlib.pyplot as plt
X_len=100
T_len=10000
X=np.linspace(0,10,X_len)
T=np.linspace(0,100,T_len)

c=-0.1
nu=0.01
dx=max(X)/X_len
dt=max(T)/T_len

a=2*nu*dt/dx**2
b=c*dt/2/dx
print(a)
print(b)
print(dx,dt)

mu = 5
sigma = 0.5
first_u = np.exp(-(X-mu)**2 / sigma**2)
U=[first_u]

for _ in range(T_len):
    last_u=U[-1]
    new_u=[]
    for x in range(X_len):
        if x==0:
            u=last_u[0]*(1-a)+last_u[1]*(a-b)+last_u[0]*(a+b)
        elif x==X_len-1:
            u=last_u[0]*(1-a)+last_u[0]*(a-b)+last_u[-2]*(a+b)
        else:
            u=last_u[x]*(1-a)+last_u[x+1]*(a-b)+last_u[x-1]*(a+b)
        new_u.append(u)
    U.append(new_u)

plt.plot(X,U[-1])
plt.show()
