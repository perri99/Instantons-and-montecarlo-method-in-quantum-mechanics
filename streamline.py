import numpy as np
from tqdm import tqdm
import random
import functions as fn
import matplotlib.pyplot as plt


def setting_inputs():
    f = 1.4 #minimum of the potential
    n = 800 #lattice points
    a = 0.05 #lattice spacing
    N_tot = 10 #instanton
    neq = 100 #number of equilibration sweeps
    nmc = 10**5 #number of MonteCarlo sweeps
    dx = 0.5 #width of updates
    n_p = 35 #number max of points in the correlation functions
    nc = 5 #number of correlator measurements in a configuration
    kp = 50 #number of sweeps between writeout of complete configuration 
    tcore = 0.3
    acore = 3.0
    dz = 1
    seed = 123456
    return f, n, a, neq, nmc, dx, n_p, nc, kp, N_tot, tcore, acore, dz, seed

f, n, a, neq, nmc, dx, n_p, nc, kp, N_inst, tcore, score, dz, seed = setting_inputs()
#random.seed(seed)
'''----------Definitions-------------------------------------------------------'''
pi    = np.pi
tcore = tcore/f
tmax  = n*a
s0    = 4.0/3.0*f**3
score = score*s0
dens  = 8*np.sqrt(2.0/pi)*f**2.5*np.exp(-s0) #classical tunneling rate
dens2 = 8*np.sqrt(2.0/pi)*f**2.5*np.exp(-s0-71.0/72.0/s0) #NLO tunneling rate
xnin  = dens*tmax         # LO total tunneling events
xnin2 = dens2*tmax        #NLO total tunneling events
nexp  = int(xnin+0.5)
nexp2 = int(xnin2+0.5)
x = np.zeros(n+1)
#   plot S_IA                                                              
#------------------------------------------------------------------------------
fn.directory('streamline')
sum_ansatz = open('Data/streamline/sum_ansatz.dat', 'w')
zero_crossing = open('Data/streamline/zero_crossing.dat', 'w')

ni = n//4

action_int_ansatz = np.zeros(ni)
tau_ia_ansatz = np.zeros(ni)

#-----------------sum ansatz----------------------------------------------------
for na in range(ni, ni*2):
    z = np.array([ni*a, na*a])
    tau_array = np.linspace(0., n * a, n, False)
    x = fn.new_config(x, n, 2, z, f, a)
    s = fn.action(x, n, a, f)
    s /= s0
    s -= 2
    tau_ia_ansatz[na - ni] = z[1] - z[0] 
    sum_ansatz.write('{:.2f}\t{:.4f}\n'.format(tau_ia_ansatz[na-ni], s))
    #sia.write('{:.4f}\t{:.4f}\n'.format((na-ni)*a, S/s0-2.0))
sum_ansatz.close()

#------------zero crossing-----------------------------------------------------
tau_ia_zcr_list = []
action_int_zcr_list = []
for n_counter in range(ni, 2 * ni):
    z = np.array([ni*a, n_counter*a])
    tau_array = np.linspace(0., n * a, n, False)
    x = fn.new_config(x,n,2,z,f,a)
    n_inst, n_a_inst, pos_roots, neg_roots = fn.find_instantons(x,a)
    if n_counter % 2 == 0 \
            and n_inst == n_a_inst \
            and n_inst > 0 \
            and n_inst == len(pos_roots) \
            and n_a_inst == len(neg_roots):

        
        for i in range(n_inst):
            if i == 0:
                zero_m = neg_roots[-1] - n * a
            else:
                zero_m = neg_roots[i - 1]

            z_ia = np.minimum(np.abs(neg_roots[i] - pos_roots[i]),
                              np.abs(pos_roots[i] - zero_m))

        
            tau_ia_zcr_list.append(z_ia)
            action_int_zcr_list.append(fn.action(x,n,a,f))

tau_ia_zcr = np.array(tau_ia_zcr_list, float)
action_int_zcr = np.array(action_int_zcr_list, float)
action_int_zcr /= s0
action_int_zcr -= 2
for k in range(len(tau_ia_zcr)):
    zero_crossing.write('{}\t{}\n'.format(tau_ia_zcr[k], action_int_zcr[k]))
zero_crossing.close()
#------------stream line--------------------------------------------------------
stream = open('Data/streamline/streamline.dat', 'w')
conf = open('Data/streamline/conf.dat', 'w')
action = open('Data/streamline/action.dat', 'w')
conf1 = open('Data/streamline/conf1.dat', 'w')
conf2 = open('Data/streamline/conf2.dat', 'w')
conf3 = open('Data/streamline/conf3.dat', 'w')
conf4 = open('Data/streamline/conf4.dat', 'w')
conf5 = open('Data/streamline/conf5.dat', 'w')
conf6 = open('Data/streamline/conf6.dat', 'w')
conf7 = open('Data/streamline/conf7.dat', 'w')
conf8 = open('Data/streamline/conf8.dat', 'w')
conf9 = open('Data/streamline/conf9.dat', 'w')
conf10 = open('Data/streamline/conf10.dat', 'w')
action1 = open('Data/streamline/action1.dat', 'w')
action2 = open('Data/streamline/action2.dat', 'w')
action3 = open('Data/streamline/action3.dat', 'w')
action4 = open('Data/streamline/action4.dat', 'w')
action5 = open('Data/streamline/action5.dat', 'w')
action6 = open('Data/streamline/action6.dat', 'w')
action7 = open('Data/streamline/action7.dat', 'w')
action8 = open('Data/streamline/action8.dat', 'w')
action9 = open('Data/streamline/action9.dat', 'w')
action10 = open('Data/streamline/action10.dat', 'w')
n_lattice_half = 50
dtau = a
r_initial_sep = 1.8
n_streamline = 70001
tau_store = 1.5
stream_time_step = 0.001

tau_centers_ia = np.array([n_lattice_half * dtau - r_initial_sep / 2.0,
                           n_lattice_half * dtau + r_initial_sep / 2.0])

tau_array = np.linspace(0.0, n_lattice_half * 2 *
                        dtau, n_lattice_half * 2, False)
xconf = np.zeros(n_lattice_half*2)
xconf = fn.new_config(xconf, n_lattice_half*2, len(tau_centers_ia), tau_centers_ia, f, a)
for k in range(len(tau_array)):
    conf.write('{:.4f}\t{:.4f}\n'.format(tau_array[k], xconf[k]))

action_density = np.zeros(2 * n_lattice_half, float)
# Derivative in streamline parameter
lambda_derivative = np.zeros(2 * n_lattice_half)
xconf = np.append(xconf, xconf[1])
xconf = np.insert(xconf, 0, xconf[0])
xconf = np.insert(xconf, 0, xconf[0])
xconf[-1] = xconf[-2]
xconf = np.append(xconf, xconf[-1])
# Streamline evolution
for i_s in tqdm(range(n_streamline)):
    # Evaluate the derivative of the action
    for i_pos in range(2, 2 * n_lattice_half+2, 1):
        der_2 = (-xconf[i_pos + 2] + 16 * xconf[i_pos + 1] - 30 *
                 xconf[i_pos]
                 + 16 * xconf[i_pos - 1] - xconf[i_pos - 2]) \
                / (12 * dtau * dtau)

        lambda_derivative[i_pos - 2] = - der_2 / 2.0 + 4 * xconf[i_pos] \
                                       * (xconf[i_pos]
                                          * xconf[i_pos]
                                          - f
                                          * f)

    for i_pos in range(2, 2 * n_lattice_half +2):
        xconf[i_pos] += -lambda_derivative[i_pos - 2] * stream_time_step
        
    xconf[0] = xconf[2]
    xconf[1] = xconf[2]
    xconf[-1] = xconf[-3]
    xconf[-2] = xconf[-3]
    for i in range(2, 2 * n_lattice_half+2):
        v = (xconf[i] ** 2 - f ** 2) ** 2
        k = (xconf[i + 1] - xconf[i - 1]) / (2. * dtau)
        action_density[i - 2] = k ** 2 / 4. + v
        action.write('{:.4f}\t{:.4f}\n'.format(tau_array[i-2], action_density[i-2]))
    
    s = fn.action(xconf,n_lattice_half*2,a,f)
    print(s)
    n_i, n_a, pos_root, neg_root = fn.find_instantons(xconf[2:-2], dtau)
    
    if 59000 < i_s < 64000 and i_s % 10 == 0:
        if n_i == n_a \
                and n_i != 0 \
                and pos_root.size == n_i and neg_root.size == n_a:
            interactive_action = s - 2 * s0

            tau_i_a = np.abs(pos_root[0] - neg_root[0])
            if tau_i_a < tau_store - 0.08:
                stream.write('{:.4f}\t{:.4f}\n'.format(tau_i_a, interactive_action/s0))
    
    if s > 0.0001:
        if s /s0 > 1.8:
            print(1.8)
            for k in range(len(tau_array)):
                conf1.write('{:.4f}\t{:.4f}\n'.format(tau_array[k], xconf[k+2]))
                action1.write('{:.4f}\t{:.4f}\n'.format(tau_array[k], action_density[k]))
        elif s /s0 > 1.6 :
            print(1.6)
            for k in range(len(tau_array)):
                conf2.write('{:.4f}\t{:.4f}\n'.format(tau_array[k], xconf[k+2]))
                action2.write('{:.4f}\t{:.4f}\n'.format(tau_array[k], action_density[k]))
        elif s /s0 > 1.4 :
            print(1.4)
            for k in range(len(tau_array)):
                conf3.write('{:.4f}\t{:.4f}\n'.format(tau_array[k], xconf[k+2]))
                action3.write('{:.4f}\t{:.4f}\n'.format(tau_array[k], action_density[k]))
        elif s /s0 > 1.2 :
            print(1.2)
            for k in range(len(tau_array)):
                conf4.write('{:.4f}\t{:.4f}\n'.format(tau_array[k], xconf[k+2]))
                action4.write('{:.4f}\t{:.4f}\n'.format(tau_array[k], action_density[k]))
        elif s /s0 > 1.0 :
            print(1.0)
            for k in range(len(tau_array)):
                conf5.write('{:.4f}\t{:.4f}\n'.format(tau_array[k], xconf[k+2]))
                action5.write('{:.4f}\t{:.4f}\n'.format(tau_array[k], action_density[k]))
        elif s /s0 > 0.8 :
            print(0.8)
            for k in range(len(tau_array)):
                conf6.write('{:.4f}\t{:.4f}\n'.format(tau_array[k], xconf[k+2]))
                action6.write('{:.4f}\t{:.4f}\n'.format(tau_array[k], action_density[k]))
        elif s /s0 > 0.6 :
            print(0.6)
            for k in range(len(tau_array)):
                conf7.write('{:.4f}\t{:.4f}\n'.format(tau_array[k], xconf[k+2]))
                action7.write('{:.4f}\t{:.4f}\n'.format(tau_array[k], action_density[k]))
        elif s /s0 > 0.4 :
            print(0.4)
            for k in range(len(tau_array)):
                conf8.write('{:.4f}\t{:.4f}\n'.format(tau_array[k], xconf[k+2])) 
                action8.write('{:.4f}\t{:.4f}\n'.format(tau_array[k], action_density[k]))
        elif s /s0 > 0.2 :
            print(0.2)
            for k in range(len(tau_array)):
                conf9.write('{:.4f}\t{:.4f}\n'.format(tau_array[k], xconf[k+2]))
                action9.write('{:.4f}\t{:.4f}\n'.format(tau_array[k], action_density[k]))
        else:
            print(0.1)
            for k in range(len(tau_array)):
                conf10.write('{:.4f}\t{:.4f}\n'.format(tau_array[k], xconf[k+2]))
                action10.write('{:.4f}\t{:.4f}\n'.format(tau_array[k], action_density[k]))
plt.plot(tau_array-2.5, action_density)
plt.show()
conf.close()
conf1.close()
conf2.close()
conf3.close()
conf4.close()
conf5.close()
conf6.close()
conf7.close()
conf8.close()
conf9.close()
conf10.close()
action.close()
stream.close()
action1.close()
action2.close()
action3.close()
action4.close() 
action5.close() 
action6.close() 
action7.close() 
action8.close() 
action9.close() 
action10.close() 
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

plt.xlim(0,2)
plt.ylim(-2,0)
plt.legend()
plt.show()

