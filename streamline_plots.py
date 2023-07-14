import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

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

plt.errorbar(x, y, fmt='v', markeredgecolor = 'blue',  label= 'iilm')

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

plt.xlim(0,2)
plt.ylim(-2,1.1)
plt.xlabel('t_IA')
plt.ylabel('S/S0 - 2')
plt.legend()
plt.savefig('Data/streamline/streamline.pdf')
plt.savefig('Data/streamline/streamline.png')
plt.show()