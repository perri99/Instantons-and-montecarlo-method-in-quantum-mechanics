import numpy as np
from tqdm import tqdm
import random
import functions as fn


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
'''-------------outputs file---------------------------------------------------'''
fn.directory('iilm')
        
iilm = open('Data/iilm/iilm.dat', 'w')
iilm.write('qm iilm\n n\t a\t f\n'+'{}\t{:.4f}\t{:.4f}\n'.format(n, a, f))
iilm.write('N_inst\t nmc\t neq\t n_p\t nc\n'+'{}\t{}\t{}\t{}\t{}\n'.format(N_inst, nmc, neq, n_p, nc))
iilm.write('dz\t tcore\t score\n'+'{:.4f}\t{:.4f}\t{:.4f}\n\n'.format(dz, tcore, score))
config_iilm = open('Data/iilm/config_iilm.dat', 'w')
trajectory_iilm = open('Data/iilm/trajectory_iilm.dat', 'w')
trajectory_iilm.write('i\t S\t T\t V\t S/(N_inst*s0)\n')
icor = open('Data/iilm/icor_iilm.dat', 'w')
icor2 = open('Data/iilm/icor2_iilm.dat', 'w')
icor3 = open('Data/iilm/icor3_iilm.dat', 'w')
iconf = open('Data/iilm/iconf_iilm.dat', 'w')
zdist = open('Data/iilm/zdist.dat', 'w')
sia  = open('Data/iilm/sia.dat', 'w')
'''#     parameters for histograms                                              
#------------------------------------------------------------------------------|'''
nzhist    = 40
stzhist   = 4.01/float(nzhist) 
#------------------------------------------------------------------------------
nconf = 0
ncor  = 0
S_sum  = 0.0
S2_sum = 0.0
V_sum  = 0.0
V2_sum = 0.0
T_sum  = 0.0
T2_sum = 0.0
Tvir_sum  = 0.0
T2vir_sum = 0.0
x_sum     = 0.0
x2_sum    = 0.0
x4_sum    = 0.0
x8_sum    = 0.0
'''-------Array Defintions-----------------------------------------------------'''
z          = np.zeros(N_inst)
x          = np.zeros(n+1)
iz         = np.zeros(nzhist)
xcor_av    = np.zeros(n_p)
xcor_er    = np.zeros(n_p)
x2cor_av   = np.zeros(n_p)
x2cor_er   = np.zeros(n_p)
x3cor_av   = np.zeros(n_p)
x3cor_er   = np.zeros(n_p)
x2sub_av   = np.zeros(n_p)
x2sub_er   = np.zeros(n_p)       
xcor_sum   = np.zeros(n_p)
xcor2_sum  = np.zeros(n_p)
x2cor_sum  = np.zeros(n_p)
x2cor2_sum = np.zeros(n_p)
x3cor_sum  = np.zeros(n_p)
x3cor2_sum = np.zeros(n_p)     
#   plot S_IA                                                              
#------------------------------------------------------------------------------
ni = n//4
sia.write('(na-ni)*a\t S/s0 - 2\n')
for na in range(ni, ni*2):
    z[0] = ni*a
    z[1] = na*a
    x = fn.new_config(x, n, 2, z, f, a)
    T = fn.kinetic(x, n, a)
    V = fn.potential(x, n, a, f)
    S = T+V
    S  += fn.hardcore_interaction(z, 2, tcore, score, tmax, s0)
    sia.write('{:.4f}\t{:.4f}\n'.format((na-ni)*a, S/s0-2.0))
#   setup and intial action                                                
#------------------------------------------------------------------------------
z = np.random.uniform(0, tmax, size = N_inst)
z = np.sort(z)
x = fn.new_config(x, n, N_inst, z, f, a)

#   loop over configs                                                      
#------------------------------------------------------------------------------
for i in tqdm(range(nmc)):
    nconf += 1
    if i == neq :
        ncor       = 0
        nconf      = 0
        stot_sum   = 0.0
        stot2_sum  = 0.0
        vtot_sum   = 0.0
        vtot2_sum  = 0.0
        ttot_sum   = 0.0
        ttot2_sum  = 0.0
        tvir_sum   = 0.0
        tvir2_sum  = 0.0
        x_sum      = 0.0
        x2_sum     = 0.0
        x4_sum     = 0.0
        x8_sum     = 0.0
        xcor_sum   = np.zeros(n_p)
        xcor2_sum  = np.zeros(n_p)
        x2cor_sum  = np.zeros(n_p)
        x2cor2_sum = np.zeros(n_p)
        x3cor_sum  = np.zeros(n_p)
        x3cor2_sum = np.zeros(n_p)
        iz         = np.zeros(nzhist)
    z, x = fn.update_interacting_instanton(N_inst, z, tmax, tcore, score, dz, x, n, a, f, s0)
    if i > 100 and i < 3000 :
        for ipr in range(min(10,len(z))):
            iconf.write(f'{z[ipr]:.4f}\t')
        iconf.write('\n')
    fn.instanton_distribution(z, N_inst, tmax, stzhist, nzhist, iz)
    #   action etc.                                                            
   #--------------------------------------------------------------------------
    T = fn.kinetic(x, n, a)
    V = fn.potential(x, n, a, f)
    S = T + V
    S += fn.hardcore_interaction(z, N_inst, tcore, score, tmax, s0)
       
    S_sum  += S
    S2_sum += S**2
    V_sum  += V
    V2_sum += V**2
    T_sum  += T
    T2_sum += T**2
    
    x_sum  += np.sum(x)
    x2_sum += np.sum(x**2)
    x4_sum += np.sum(x**4)
    x8_sum += np.sum(x**8)
        
    for ic in range(nc):
        ncor += 1 
        xcor = fn.correlations_functions(x, n, n_p)
        xcor_sum   = np.add(xcor, xcor_sum)
        xcor2_sum  = np.add(xcor**2, xcor2_sum)
        x2cor_sum  = np.add(xcor**2, x2cor_sum)
        x2cor2_sum = np.add(xcor**4, x2cor2_sum)
        x3cor_sum  = np.add(xcor**3, x3cor_sum)
        x3cor2_sum = np.add(xcor**6, x3cor2_sum)
    
    trajectory_iilm.write('{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(i, S, T, V, S/(N_inst*s0)))
    if i % kp == 0:
        config_iilm.write('configuration: '+str(i)+'\n k*a\t x[k]\n')
        for k in range(n):
            config_iilm.write('{:.4f}\t{:.4f}\n'.format(k*a, x[k]))
        config_iilm.write('--------------------------------------\n')
#----------Averages and error computations----------------------|
stot_av,stot_err = fn.dispersion(nconf,S_sum,S2_sum)
vtot_av,vtot_err = fn.dispersion(nconf,V_sum,V2_sum)
ttot_av,ttot_err = fn.dispersion(nconf,T_sum,T2_sum)
x_av,x_err       = fn.dispersion(nconf*n,x_sum,x2_sum)
x2_av,x2_err     = fn.dispersion(nconf*n,x2_sum,x4_sum)
x4_av,x4_err     = fn.dispersion(nconf*n,x4_sum,x8_sum)
for ip in range(n_p):
    xcor_av[ip],xcor_er[ip]   = fn.dispersion(ncor,xcor_sum[ip],xcor2_sum[ip])
    x2cor_av[ip],x2cor_er[ip] = fn.dispersion(ncor,x2cor_sum[ip],x2cor2_sum[ip])
    x3cor_av[ip],x3cor_er[ip] = fn.dispersion(ncor,x3cor_sum[ip],x3cor2_sum[ip])  
v_av   = vtot_av/tmax
v_err  = vtot_err/tmax
t_av   = ttot_av/tmax
t_err  = ttot_err/tmax
e_av   = v_av + t_av
e_err  = np.sqrt(v_err**2 + t_err**2)
iilm.write("stot_av\t stot_err\t v_av\t v_err\t t_av\t t_err\t stot_av/N_inst\t stot_err/N_inst\n")
iilm.write("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(stot_av,stot_err, v_av, v_err, t_av, t_err,stot_av/float(N_inst),stot_err/float(N_inst)))
iilm.write("s0\t stot_av/float(N_inst*s0)\t stot_err/float(N_inst*s0)\n")
iilm.write("{:.4f}\t{:.4f}\t{:.4f}\n".format(s0,stot_av/float(N_inst*s0),stot_err/float(N_inst*s0))) 
iilm.write('e_av, e_err, x_av, x_err, x2_av,x2_err, x4_av, x4_err\n')  
iilm.write("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(e_av, e_err, x_av, x_err, x2_av, x2_err, x4_av, x4_err))
#----------log derivatives------------------------------------------------------------|
dx, dxe = fn.log_derivatives(xcor_av, xcor_er, a, n_p)
x2sub_av, x2sub_er = fn.substract(x2cor_av, x2cor_er, n_p) #substracting
dx2, dxe2 = fn.log_derivatives(x2sub_av, x2sub_er, a, n_p)
dx3, dxe3 = fn.log_derivatives(x3cor_av, x3cor_er, a, n_p)
icor.write('tau\t   x(tau)\t    dx(tau)\t dlogx(tau)\n')
icor2.write('tau\t   x^2(tau)\t    dx^2(tau)\t dlogx3(tau)\n')
icor3.write('tau\t   x^3(tau)\t    dx^3(tau)\t dlogx3(tau)\n')
#output log derivatives
for ip in range(n_p-1):
    icor.write("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(ip*a, xcor_av[ip], xcor_er[ip], dx[ip], dxe[ip]))
    icor2.write("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(ip*a, x2cor_av[ip], x2cor_er[ip], dx2[ip], dxe2[ip]))
    icor3.write("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(ip*a, x3cor_av[ip], x3cor_er[ip], dx3[ip], dxe3[ip]))
#   histograms                                                             
#------------------------------------------------------------------------------
zdist.write('xx\t iz\n')
for i in range(nzhist):
    xx = (i+0.5)*stzhist
    zdist.write("{:.4f}\t{:.4f}\n".format(xx, iz[i]))    



iilm.close()
config_iilm.close()
trajectory_iilm.close()
icor.close()
icor2.close()
icor3.close()
iconf.close()
zdist.close()
sia.close()