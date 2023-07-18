import numpy as np
import functions as fn
import random
from tqdm import tqdm
import inputs
'''
Computations of instanton density as function of eta (double well separation)\
    after 10 cooling sweeps
'''
#----------------------setting inputs-----------------------------------------------
f, n, a, neq, nmc, dx, n_p, nc, kp, mode, seed, kp2, ncool = inputs.qmcool()
random.seed(seed)
#--------------------output file----------------------------------------------------
fn.directory('qmcool')
instdensity = open('Data/qmcool/instdensity_varf.dat', 'w')
#------------------loops over etas--------------------------------------------------
f = np.linspace(0.25, 1.75, num = 10)
for loop in range(10):
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
            nin_sum    = np.zeros(ncool)
            nin2_sum   = np.zeros(ncool)
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
            
            ni, na     = fn.instantons(a, n, xs, xi, xa, z)
            Sc, Vc, Tc, TVc = fn.compute_energy(xs, n, a, f[loop])
            nin = ni + na
            nin_sum[0]   += nin
            nin2_sum[0]  += nin**2
            scool_sum[0] += Sc
            scool2_sum[0]+= Sc**2
            for icool in range(1,ncool):                     
               xs = fn.cooling_update(xs, n, a, f[loop], dx)
               
               ni, na     = fn.instantons( a, n, xs, xi, xa, z)
               Sc, Vc, Tc, TVc = fn.compute_energy(xs, n, a, f[loop])
               nin = ni + na
               nin_sum[icool]   += nin
               nin2_sum[icool]  += nin**2
               scool_sum[icool] += Sc
               scool2_sum[icool]+= Sc**2
            #--------------cooled configuration: instanton distribution  -------------|                          
            fn.instanton_distribution(z, nin, tmax, stzhist, nzhist, iz)
    #   instanton density                                      
    #------------------------------------------------------------------------------
    for ic in range(ncool):
        nin_av[ic], nin_er[ic] = fn.dispersion(ncoolconf, nin_sum[ic]  , nin2_sum[ic])
    instdensity.write('{:.4f}\t{:.4f}\t{:.4f}\n'.format(f[loop], nin_av[9],  nin_er[9]))
instdensity.close()