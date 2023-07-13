import numpy as np
import functions as fn
import random
from tqdm import tqdm

def setting_inputs():
    npri = 100
    nst = 20
    kp2 = 50
    ncool = 200
    kp = 50
    #f = 1.4 #minimum of the potential
    n = 800 #lattice points
    a = 0.05 #lattice spacing
    neq = 100 #number of equilibration sweeps
    nmc = 10**5 #number of MonteCarlo sweeps
    dx = 0.5 #width of updates
    n_p = 20 #number max of points in the correlation functions
    nc = 5 #number of correlator measurements in a configuration
    kp = 50 #number of sweeps between writeout of complete configuration 
    mode = 0 # ih=0: cold start, x_i=-f; ih=1: hot start, x_i=random
    seed = 597
    return n, a, neq, nmc, dx, n_p, nc, kp, mode, seed, npri, nst, kp2, ncool

n, a, neq, nmc, dx, n_p, nc, kp, mode, seed, npri, nst, kp2, ncool = setting_inputs()
random.seed(seed)

fn.directory('qmcool')
instdensity = open('Data/qmcool/instdensity.dat', 'w')
inst_action = open('Data/qmcool/inst_action.dat', 'w')

f = np.linspace(1.4, 1.6, num = 3)
for loop in range(3):
    #----------- setting constants--------------------------------------------------|
    pi  = np.pi
    s0  = 4.0/3.0*f[loop]**3     #instanton action
    de  = 8*np.sqrt(2.0/pi)*f[loop]**2.5*np.exp(-s0)
    de2 = de*(1.0-71.0/72.0/s0)
    tmax = n*a
    #-------counter variables-------------------------------------------|
    nconf = 0 #it counts the number of randomly generated configurations 
    ncor = 0 #it counts the number of correlations taken
    ncoolconf = 0
    ncoolcor  = 0
    #-------histogram parameters----------------------------------------|
    nzhist = 40
    stzhist = 4.01 / float(nzhist)
    z = np.zeros(n)
    nin_sum    = np.zeros(ncool+1)
    nin2_sum   = np.zeros(ncool+1)
    scool_sum  = np.zeros(ncool+1)
    scool2_sum = np.zeros(ncool+1)
    iz         = np.zeros(nzhist)
    xi         = np.zeros(n)
    xa         = np.zeros(n)
    nin_av      = np.zeros(ncool+1)
    nin_er      = np.zeros(ncool+1)
    scool_av    = np.zeros(ncool+1)
    scool_er    = np.zeros(ncool+1)
    #starting configuration-----------------------------------------|
    x = fn.periodic_starting_conf(n, f[loop], mode)
    #---------montecarlo generations--------------------------------|
    for i in tqdm(range(nmc)):
        nconf += 1
        if i == neq:       #when we reach equilibrium all is set to zero
            nconf = 0
            ncor = 0
            ncoolconf = 0
            ncoolcor = 0
            nin_sum    = np.zeros(ncool+1)
            nin2_sum   = np.zeros(ncool+1)
            scool_sum  = np.zeros(ncool+1)
            scool2_sum = np.zeros(ncool+1)
            iz         = np.zeros(nzhist)
            xi         = np.zeros(n)
            xa         = np.zeros(n)
            nin_av      = np.zeros(ncool+1)
            nin_er      = np.zeros(ncool+1)
            scool_av    = np.zeros(ncool+1)
            scool_er    = np.zeros(ncool+1)
        x = fn.periodic_update(x,n,a,f[loop], dx)  #metropolis algorithm implementation with periodic boundary conditions
        xs = np.copy(x)
        #cooling sweeps
        if i % kp2 == 0:
            ncoolconf += 1
            
            ni, na     = fn.instantons(f[loop], a, n, xs, xi, xa, z)
            Sc, Vc, Tc, TVc = fn.compute_energy(xs, n, a, f[loop])
            nin = ni + na
            nin_sum[0]   += nin
            nin2_sum[0]  += nin**2
            scool_sum[0] += Sc
            scool2_sum[0]+= Sc**2
            for icool in range(1,ncool+1):                     
               xs = fn.cooling_update(xs, n, a, f[loop], dx)
               
               ni, na     = fn.instantons(f[loop], a, n, xs, xi, xa, z)
               Sc, Vc, Tc, TVc = fn.compute_energy(xs, n, a, f[loop])
               nin = ni + na
               nin_sum[icool]   += nin
               nin2_sum[icool]  += nin**2
               scool_sum[icool] += Sc
               scool2_sum[icool]+= Sc**2
            #--------------cooled configuration: instanton distribution  -------------|                          
            fn.instanton_distribution(z, nin, tmax, stzhist, nzhist, iz)
    #   instanton density, cooled action                                       
    #------------------------------------------------------------------------------

    for ic in range(ncool + 1):
        nin_av[ic], nin_er[ic] = fn.dispersion(ncoolconf, nin_sum[ic]  , nin2_sum[ic]) 
        scool_av[ic], scool_er[ic] = fn.dispersion(ncoolconf, scool_sum[ic], scool2_sum[ic]) 
    instdensity.write('f = '+'{:.1f}\n'.format(f[loop])+'conf\t nin\t nin_err\n')
    for ic in range(ncool+1):
        instdensity.write("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(ic, nin_av[ic], nin_er[ic], de*tmax, de2*tmax))
    inst_action.write('f = '+'{:.1f}\n'.format(f[loop])+'conf\t sin\t sin_err\n')
    for ic in range(ncool+1):
        si_av= scool_av[ic]/nin_av[ic]                    
        del2 =(scool_er[ic]/scool_av[ic])**2+(nin_er[ic]/nin_av[ic])**2
        si_er= si_av*np.sqrt(del2)
        inst_action.write("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(ic, si_av, si_er, s0))
inst_action.close()
instdensity.close()