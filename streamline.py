import numpy as np
from tqdm import tqdm
import random
import functions as fn
import matplotlib.pyplot as plt
import inputs

#-----------setting inputs--------------------------------------------------
f, n, a, neq, nmc, dx, n_p, nc, kp, N_inst, tcore, score, dz, seed = inputs.iilm()
#random.seed(seed)
#'''----------Definitions-------------------------------------------------------'''
tcore = tcore/f
tmax  = n*a
s0    = 4.0/3.0*f**3
score = score*s0
x = np.zeros(n+1)
#   plot S_IA                                                              
#------------------------------------------------------------------------------
fn.directory('streamline')
sum_ansatz = open('Data/streamline/sum_ansatz.dat', 'w')
zero_crossing = open('Data/streamline/zero_crossing.dat', 'w')
fn.directory('iilm')
sia  = open('Data/iilm/sia.dat', 'w')

ni = n//4

action_int_ansatz = np.zeros(ni)
tau_ia_ansatz = np.zeros(ni)

#-----------------sum ansatz----------------------------------------------------
for na in range(ni, ni*2):
    z = np.array([ni*a, na*a])
    x = fn.new_config(x, n, 2, z, f, a)
    s = fn.action(x, n, a, f)
    s /= s0
    s -= 2
    tau_ia_ansatz[na - ni] = z[1] - z[0] 
    sum_ansatz.write('{:.2f}\t{:.4f}\n'.format(tau_ia_ansatz[na-ni], s))
    #sia.write('{:.4f}\t{:.4f}\n'.format((na-ni)*a, S/s0-2.0))
sum_ansatz.close()

#------------sum ansatz zero crossing-----------------------------------------------------
tau_ia_zcr, action_int_zcr = fn.zero_crossing(x)
for k in range(len(tau_ia_zcr)):
    zero_crossing.write('{}\t{}\n'.format(tau_ia_zcr[k], action_int_zcr[k]))
zero_crossing.close()

#-----------interactive model---------------------------------------------------
sia.write('(na-ni)*a\t S/s0 - 2\n')
for na in range(ni, ni*2):
    z = np.array([ni*a, na*a])
    x = fn.new_config(x, n, 2, z, f, a)
    T = fn.kinetic(x, n, a)
    V = fn.potential(x, n, a, f)
    S = T+V
    S  += fn.hardcore_interaction(z, 2, tcore, score, tmax, s0)
    sia.write('{:.4f}\t{:.4f}\n'.format((na-ni)*a, S/s0-2.0))
sia.close()
#------------stream line--------------------------------------------------------
fn.streamline_equation(50, 1.8, 70001, 0.05)
#------------plotting------------------------------------------------------------
with open('Data/streamline/sum_ansatz.dat', 'r') as file:
    lines = file.readlines()

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]

x     = np.array(column1)
y     = np.array(column2)

plt.plot(x, y, color = 'black', label= 'sum ansatz')

with open('Data/iilm/sia.dat', 'r') as file:
    lines = file.readlines()[1:]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]

x     = np.array(column1)
y     = np.array(column2)

plt.errorbar(x, y, fmt='v', markeredgecolor = 'blue',  label= 'interactive model')

with open('Data/streamline/zero_crossing.dat', 'r') as file:
    lines = file.readlines()

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]

x     = np.array(column1)
y     = np.array(column2)

plt.errorbar(x, y, fmt='s', markeredgecolor = 'blue',  label= 'sum ansatz zero crossing')

with open('Data/streamline/zero_crossing.dat', 'r') as file:
    lines = file.readlines()

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]

x     = np.array(column1)
y     = np.array(column2)

plt.errorbar(x, y,color = 'black', linestyle = '--', label= 'streamline')

with open('Data/zero cross cooling/zia.dat', 'r') as file:
    lines = file.readlines()

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]

x     = np.array(column1)
y     = np.array(column2)

plt.errorbar(x, y, fmt='o', markeredgecolor = 'red',  label= 'cooling montecarlo')

plt.xlim(0,2)
plt.ylim(-2,1.1)
plt.xlabel('t_IA')
plt.ylabel('S/S0 - 2')
plt.legend()
plt.savefig('Data/streamline/streamline.pdf')
plt.savefig('Data/streamline/streamline.png')
plt.show()

files = ['conf.dat', 'conf1.dat', 'conf2.dat', 'conf3.dat', 'conf4.dat',\
         'conf5.dat', 'conf6.dat', 'conf7.dat', \
             'conf8.dat' ,'conf9.dat', 'conf10.dat']

for fil in tqdm(files):
    with open('Data/streamline/'+ fil, 'r') as file:
        lines = file.readlines()[0:100]

    column1 = [float(line.split()[0]) for line in lines]
    column2 = [float(line.split()[1]) for line in lines]

    x     = np.array(column1)
    y     = np.array(column2)

    plt.plot(x-2.5, y, color = 'black', linewidth = 0.5, label= 'sum ansatz')
plt.xlabel('t')
plt.ylabel('x(t)')
plt.savefig('Data/streamline/config.pdf')
plt.savefig('Data/streamline/config.png')
plt.show()
#--------------------------------------------------------------------------------
files = ['action.dat', 'action1.dat', 'action2.dat', 'action3.dat', 'action4.dat',\
         'action5.dat', 'action6.dat', 'action7.dat', \
             'action8.dat' ,'action9.dat', 'action10.dat']

for fil in tqdm(files):
    with open('Data/streamline/'+ fil, 'r') as file:
        lines = file.readlines()[0:100]

    column1 = [float(line.split()[0]) for line in lines]
    column2 = [float(line.split()[1]) for line in lines]

    x     = np.array(column1)
    y     = np.array(column2)

    plt.plot(x-2.5, y, color = 'black',linewidth = 0.5,)
plt.xlabel('t')
plt.ylabel('s(t)')
plt.savefig('Data/streamline/densaction.pdf')
plt.savefig('Data/streamline/densaction.png')
plt.show()
