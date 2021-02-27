import glob

u=[]
r=[]
theta=[]
v=[]
mu=[]

file_list_u = glob.glob('0V0000.txt')
for filename in file_list_u:
    with open(filename) as input:
        input_=input.read().split('\n')
        for input__ in input_:
            if input__ == '':
                continue
            u_ = input__.split()
            u.append(u_)
file_list_r = glob.glob('r.txt')
for filename in file_list_r:
    with open(filename) as input:
        input_=input.read().split('\n')
        for input__ in input_:
            if input__ == '':
                continue
            r.append(input__)
file_list_theta = glob.glob('theta.txt')
for filename in file_list_theta:
    with open(filename) as input:
        input_=input.read().split('\n')
        for input__ in input_:
            if input__ == '':
                continue
            theta.append(input__)
file_list_v = glob.glob('v.txt')
for filename in file_list_v:
    with open(filename) as input:
        input_=input.read().split('\n')
        for input__ in input_:
            if input__ == '':
                continue
            v.append(input__)
file_list_mu = glob.glob('mu.txt')
for filename in file_list_mu:
    with open(filename) as input:
        input_=input.read().split('\n')
        for input__ in input_:
            if input__ == '':
                continue
            mu.append(input__)

