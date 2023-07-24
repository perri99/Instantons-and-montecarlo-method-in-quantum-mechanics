import functions as fn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import inputs

f, n, a, neq, nmc, dx, n_p, nc, kp, mode, seed, kp2, ncool = inputs.qmcool()
pi  = np.pi
s0  = 4.0/3.0*f**3     #instanton action
de  = 8*np.sqrt(2.0/pi)*f**2.5*np.exp(-s0)
de2 = de*(1.0-71.0/72.0/s0)
tmax = n*a
histz = open('Data/qmcool/hist_z.dat', 'w')
histz.write('xx iz[i]\n')
nzhist = 40
stzhist = 4.0 / float(nzhist)
iz = np.zeros(nzhist)
action_array = np.empty(0)
x = fn.periodic_starting_conf(n, f, mode)

for i in tqdm(range(350000)):
    if i == 100:
        iz = np.zeros(nzhist)
    x = fn.periodic_update(x,n,a,f, dx)  #metropolis algorithm implementation with periodic boundary conditions
    xs = np.copy(x)
    if i % kp2 == 0:
        S = 0
        for icool in range(1,ncool+1):                     
           xs = fn.cooling_update(xs, n, a, f, dx)
           
           ni, na, z     = fn.instantons(a, n, xs)
           nin = na + ni
           zero_crossing_histogram = \
            fn.instanton_distribution(z, nin, tmax, nzhist)
           iz = np.add(iz, zero_crossing_histogram)
           
tau = np.zeros(nzhist)       
for i in range(nzhist): 
    tau[i] = (i+0.5)*stzhist    

for i in range(nzhist): 
    xx = (i+0.5)*stzhist
    histz.write("{:.4f}\t{}\n".format(xx, iz[i]))        
plt.plot(tau, iz, drawstyle = 'steps')        
plt.show()    
histz.close()
#------------------------------------------------------------------------------
#   Fig.16 Distribution of instanton-anti-instanton separations
#------------------------------------------------------------------------------
with open('Data/rilm/hist_z_rilm.dat', 'r') as file:
    lines = file.readlines()[1:]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]

x     = np.array(column1)
y     = np.array(column2)

plt.plot(x, y, color = 'red', label= 'random sum ansatz')

with open('Data/iilm/zdist.dat', 'r') as file:
    lines = file.readlines()[1:]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]

x     = np.array(column1)
y     = np.array(column2) 

plt.plot(x, y, color = 'black', drawstyle = 'steps', label = 'interactive ensemble')

with open('Data/qmcool/hist_z.dat', 'r') as file:
    lines = file.readlines()[3:]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]

x     = np.array(column1)
y     = np.array(column2) 

plt.plot(x, y, color = 'green', drawstyle = 'steps', label = 'cooling montecarlo')
'''
zcr = np.loadtxt('C:/Users/perri/OneDrive/Desktop/instantons-main/output_data/output_cooled_monte_carlo/zero_crossing/zcr_cooling.txt', float,
                 delimiter=' ')

plt.hist(zcr, 40, (0., 4.), 
        label='Monte carlo cooling', histtype='step',
        color='blue', linewidth=1.8)
'''
plt.xlabel('τ_z')
plt.ylabel('n_IA(τ_z)')
plt.title('Istanton-anti-istanton separation distribution')
plt.xlim(0,3.85)
plt.ylim(0,50000)
plt.legend()
plt.savefig('Data/iilm/Istanton-anti-istanton separation distribution.pdf')
plt.savefig('Data/iilm/Istanton-anti-istanton separation distribution.png')
plt.show()