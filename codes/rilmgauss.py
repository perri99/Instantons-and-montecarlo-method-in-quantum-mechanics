import numpy as np
import random
from tqdm import tqdm
import functions as fn
import inputs
'''
This program generates the same random instanton ensemble as rilm.py\
    but it also includes Gaussian fluctuations\
        around the classical path. This is done by performing a few\
            heating sweeps in the Gaussian effective potential
Instanton distribution is saved in output file 'hist_z_riilm'
Input:
------------------------------------------------------------------------------
   f       minimum of harmonic oxillator: (x^2-f^2)^2
   n       number of lattice points in the euclidean time direction (n=800)
   a       lattice spacing (a=0.05)
   N_inst   number of instantons(has to be even).
   nmc     number of Monte Carlo sweeps (nmc=10^5)
   n_p      number of points on which the correlation functions are measured: 
           <x_i x_(i+1)>,...,<x_i x_(i+np)> (np=20)
   nc      number of correlator measurements in a single configuration (nc=5)                               
   kp      number of sweeps between writeout of complete configuration 
   nheat   number of heating steps (nheat = 10)
   dx      gaussian update width
-------------------------------------------------------------------------------
Output:
    Stot        average total action per configuration
    Vav, Tav    average potential and kinetic energy
    <x^n>       expectation value <x^n> (n=1,2,4)
    Pi(tau)     euclidean correlation function Pi(tau)=<O(0)O(tau)>,
                for O=x,x^2,x^3;results are given in the format: tau, Pi(tau),
                     DeltaPi(tau), dlog(Pi)/dtau, Delta[dlog(Pi)/dtau],where 
                         DeltaPi(tau) is the statistical error in Pi(tau)
    iz          Instanton distribution
'''
#----------------input parameters------------------------------------------------
f, n, a, neq, nmc, dx, n_p, nc, kp, N_inst,nheat, seed = inputs.rilmgauss()
#random.seed(seed)
#-------defining constants-------------------------------------------------------|
tmax  = n*a
#--------opening output files----------------------------------------------------|
fn.directory('rilmgauss')

rilmgauss = open('Data/rilmgauss/rilmgauss.dat','w')
rilmgauss.write('gaussian\n f, n, a, N_inst, nmc, n_p, nc, dx, nheat\n')
rilmgauss.write("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(f, n, a, N_inst,nmc, n_p,nc, dx, nheat))
rilmgauss.write('n nmc/kp n*a f\n')
rilmgauss.write("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(n, nmc/kp,  n*a, f))

config1 = open('Data/rilmgauss/config1_rikmgauss.dat', 'w')
config1.write('configuration, S_tot, T_tot, V_tot\n')

config2 = open('Data/rilmgauss/config2_rilmgauss.dat', 'w')
config2.write('i*a, x[i]\n')

config_hot = open('Data/rilmgauss/config_hot_rilm.dat', 'w')
config_hot.write('i*a, x_hot[i]\n')

averages = open('Data/rilmgauss/averages_rilmgauss.dat', 'w')
averages.write('Stot_av, S_tot_err, V_av, Verr,T_av, T_err, TV_av, TV_err\n')

correlations = open('Data/rilmgauss/correlations_rilmgauss.dat', 'w')
correlations.write("x correlation function\n")

correlations2 = open('Data/rilmgauss/correlations2_rilmgauss.dat', 'w')
correlations.write("x2 correlation function\n")

correlations3 = open('Data/rilmgauss/correlations3_rilmgauss.dat', 'w')
correlations3.write("x3 correlation function\n")


histz = open('Data/rilmgauss/hist_z_rilmgauss.dat', 'w')
histz.write('xx iz[i]\n')
#   parameters for histograms                                              
#------------------------------------------------------------------------------
nzhist    = 40
stzhist   = 4.01/float(nzhist)
#------------------------------------------------------------------------------
#   inizialize                                                 
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


iz         = np.zeros(nzhist)
x          =  np.zeros(n+1)
x_hot      = np.zeros(n+1)
w          = np.zeros(n+1)
z          =  np.zeros(N_inst)     
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


#------------------------------------------------------------------------------
#   loop over configurations                                                            
#------------------------------------------------------------------------------
for i in tqdm(range(nmc)):
    nconf += 1
    z = np.random.uniform(0, tmax, size = N_inst)
    z = np.sort(z)
    x = fn.new_config(x, n, N_inst, z, f, a)
    #   distribution of instantons                                             
   #--------------------------------------------------------------------------
    zero_crossing_histogram = \
        fn.instanton_distribution(z, N_inst, tmax, nzhist)
    iz = np.add(iz, zero_crossing_histogram)
    #Computations
    tvtot = 0.0
    ttot = fn.kinetic(x, n, a)
    vtot = fn.potential(x, n, a, f)
    S_tot = ttot + vtot
    for j in range(1,n):
        tv = 2.0*x[j]**2*(x[j]**2-f**2)
        tvtot += a*tv
    #------------------------------
    #   heat configuration: start from classical path  
    #--------------------------------------------------------------------------
    x_hot = np.copy(x)
    #--------------------------------------------------------------------------
    #   heating sweeps   
    #--------------------------------------------------------------------------
    x_hot = fn.heating_update(x_hot, x, n, nheat, a, f, dx)
    if i % kp == 0:
        config1.write("{}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(i, S_tot, ttot, vtot))
        config2.write('config: '+str(nconf)+'\n')
        config_hot.write('config: '+str(nconf)+'\n')
        for j in range(n):
            config_hot.write("{:.4f}\t{:.4f}\n".format(j*a, x_hot[j]))
            config2.write("{:.4f}\t{:.4f}\n".format(j*a, x[j]))
        config2.write("------------------\n")
        config_hot.write("------------------\n")
    
    S_sum  += S_tot
    S2_sum += S_tot**2
    V_sum  += vtot
    V2_sum += vtot**2
    T_sum  += ttot
    T2_sum += ttot**2
    Tvir_sum  += tvtot
    T2vir_sum += tvtot**2
    
    x_sum  += np.sum(x)
    x2_sum += np.sum(x**2)
    x4_sum += np.sum(x**4)
    x8_sum += np.sum(x**8)
    #correlations
    for ic in range(nc):
        ncor += 1 
        xcor = fn.correlations_functions(x_hot, n, n_p)
        xcor_sum   = np.add(xcor, xcor_sum)
        xcor2_sum  = np.add(xcor**2, xcor2_sum)
        x2cor_sum  = np.add(xcor**2, x2cor_sum)
        x2cor2_sum = np.add(xcor**4, x2cor2_sum)
        x3cor_sum  = np.add(xcor**3, x3cor_sum)
        x3cor2_sum = np.add(xcor**6, x3cor2_sum)
#----------Averages and error computations----------------------|
stot_av,stot_err = fn.dispersion(nconf,S_sum,S2_sum)
vtot_av,vtot_err = fn.dispersion(nconf,V_sum,V2_sum)
ttot_av,ttot_err = fn.dispersion(nconf,T_sum,T2_sum)
tvir_av,tvir_err = fn.dispersion(nconf,Tvir_sum,T2vir_sum)
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
tv_av  = tvir_av/tmax
tv_err = tvir_err/tmax
e_av   = v_av + tv_av
e_err  = np.sqrt(v_err**2 + tv_err**2)
#output vari
averages.write("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(stot_av,stot_err, v_av, v_err, t_av,tv_av,tv_err)) 
averages.write('e_av, e_err, x_av, x_err, x2_av,x2_err, x4_av, x4_err\n')  
averages.write("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(e_av, e_err, x_av, x_err, x2_av, x2_err, x4_av, x4_err))
#----------log derivatives------------------------------------------------------------|
dx, dxe = fn.log_derivatives(xcor_av, xcor_er, a)
x2sub_av, x2sub_er = fn.substract(x2cor_av, x2cor_er) #substracting
dx2, dxe2 = fn.log_derivatives(x2sub_av, x2sub_er, a)
dx3, dxe3 = fn.log_derivatives(x3cor_av, x3cor_er, a)
correlations.write('tau\t   x(tau)\t    dx(tau)\t dlogx(tau)\n')
correlations2.write('tau\t   x^2(tau)\t    dx^2(tau)\t dlogx3(tau)\n')
correlations3.write('tau\t   x^3(tau)\t    dx^3(tau)\t dlogx3(tau)\n')
#----------output log derivatives-------------------------------------------------------|
for ip in range(n_p-1):
    correlations.write("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(ip*a, xcor_av[ip], xcor_er[ip], dx[ip], dxe[ip]))
    correlations2.write("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(ip*a, x2cor_av[ip], x2cor_er[ip], dx2[ip], dxe2[ip]))
    correlations3.write("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(ip*a, x3cor_av[ip], x3cor_er[ip], dx3[ip], dxe3[ip]))
#----------output instanton distribution -----------------------------------------------|                                                                     
for i in range(nzhist): 
    xx = (i+0.5)*stzhist
    histz.write("{:.4f}\t{:.4f}\n".format(xx, iz[i]))
 #--------closing output files-----------------------------------------------------------|  
config1.close()
config2.close()
averages.close()
correlations.close()
correlations2.close()
correlations3.close()
histz.close()
config_hot.close()
rilmgauss.close()

