import numpy as np
import functions as fn
import random
from tqdm import tqdm
import inputs
'''
Montecarlo implementation with cooling procedures. This programme computes \
    the same things of qm.py but with cooled configuration. Analysis\
        of instanton content is carried out
Input:
------------------------------------------------------------------------------
   f       minimum of harmonic oxillator: (x^2-f^2)^2
   n       number of lattice points in the euclidean time direction (n=800)
   a       lattice spacing (a=0.05)
   neq     number of equilibration sweeps
   nmc     number of Monte Carlo sweeps (nmc=10^5)
   dx      gaussian update width
   n_p      number of points on which the correlation functions are measured: 
           <x_i x_(i+1)>,...,<x_i x_(i+np)> (np=20)
   nc      number of correlator measurements in a single configuration (nc=5)                               
   kp      number of sweeps between writeout of complete configuration 
   mode    0 = cold start, otherwise hot start
   seed    seed for random generations
   kp2     number of montecarlo sweeps between cooling procedure
   ncool   number of cooling sweeps
-------------------------------------------------------------------------------
Relevant outputs:
    1. correlation functions and their log derivatives for cooled configuration
    2. instanton density as function of cooling sweeps
    3. instanton action as function of cooling sweeps
    4. instanton distribution
'''
#-----------setting inputs--------------------------------------------------
f, n, a, neq, nmc, dx, n_p, nc, kp, mode, seed, kp2, ncool = inputs.qmcool()
random.seed(seed)
#----------- setting constants--------------------------------------------------|
pi  = np.pi
s0  = 4.0/3.0*f**3     #instanton action
de  = 8*np.sqrt(2.0/pi)*f**2.5*np.exp(-s0)
de2 = de*(1.0-71.0/72.0/s0)
tmax = n*a
#------------output files-------------------------------------------------------|
fn.directory('qmcool')
        
averages = open('Data/qmcool/cool_averages.dat', 'w')
averages.write('\tStot_av, S_tot_err, V_av, Verr,T_av, T_err, TV_av, TV_err\n')

'''instdensity = open('Data/qmcool/instdensity.dat', 'w')
instdensity.write('number of instantons\n ic, nin_av[ic], nin_er[ic], de*tmax, de2*tmax\n')'''

coolingsweeps = open('Data/qmcool/coolingsweeps.dat', 'w')
coolingsweeps.write(' action vs cooling sweeps\n ic, scool_av[ic], scool_er[ic], sin\n')

'''inst_action = open('Data/qmcool/inst_action.dat', 'w')
inst_action.write('Action per instanton, S0 = ' + str(4.0/3.0*f**3) +'\n ic, si_av, si_er, s0\n' )'''

histz = open('Data/qmcool/hist_z.dat', 'w')
histz.write('xx iz[i]\n')

config1 = open('Data/qmcool/config1.dat', 'w')
config1.write('configuration, S_tot, T_tot, V_tot\n')

config2 = open('Data/qmcool/config2.dat', 'w')
config2.write('i*a, x[i]\n')

config_cool = open('Data/qmcool/config_cool.dat', 'w')
config_cool.write('i*a\t xs[i]\n')

correlations = open('Data/qmcool/correlations_qm.dat', 'w')
correlations.write("x correlation function\n")

correlations_cool = open('Data/qmcool/correlations_cool.dat', 'w')
correlations_cool.write("x correlation function(cool)\n ip*a, xcool_av[ip], xcool_er[ip], dx, dxe\n")

correlations2 = open('Data/qmcool/correlations2_qm.dat', 'w')
correlations2.write("x2 correlation function\n")

correlations2_cool = open('Data/qmcool/correlations2_cool.dat', 'w')
correlations2_cool.write("x2 correlation function(cool)\n ip*a, xcool_av[ip], xcool_er[ip], dx, dxe\n")

correlations3 = open('Data/qmcool/correlations3_qm.dat', 'w')
correlations3.write("x3 correlation function\n")

correlations3_cool = open('Data/qmcool/correlations3_cool.dat', 'w')
correlations3_cool.write("x3 correlation function (cool)\n ip*a, x3cool_av[ip], x3cool_er[ip], dx, dxe\n")
#-------counter variables-------------------------------------------|
nconf = 0        #it counts the number of randomly generated configurations 
ncor = 0         #it counts the number of correlations taken
ncoolconf = 0    #it counts the number of cooled configurations
ncoolcor  = 0    #it counts the number of  cooled correlations taken
#-------histogram parameters----------------------------------------|
nzhist = 40
stzhist = 4.0 / float(nzhist)

#-------variable definitions----------------------------------------|
S_sum = 0.0
S2_sum = 0.0
T_sum = 0.0
T2_sum = 0.0
V_sum = 0.0
V2_sum = 0.0
TV_sum = 0.0
TV2_sum = 0.0
x_sum = 0
x2_sum = 0
x4_sum = 0
x8_sum = 0
#-----Array defnitions-----------------------------------------------|
xcor_sum   = np.zeros(n_p)         
xcor2_sum  = np.zeros(n_p) 
x2cor_sum  = np.zeros(n_p)
x2cor2_sum = np.zeros(n_p)
x3cor_sum  = np.zeros(n_p)
x3cor2_sum = np.zeros(n_p)
xcor_av    = np.zeros(n_p)
x2cor_av   = np.zeros(n_p)
x3cor_av   = np.zeros(n_p)
xcor_er    = np.zeros(n_p)
x2cor_er   = np.zeros(n_p)
x3cor_er   = np.zeros(n_p)
ipa        = np.zeros(nc)
xs         = np.zeros(n+1)
nin_sum    = np.zeros(ncool+1)
nin2_sum   = np.zeros(ncool+1)
scool_sum  = np.zeros(ncool+1)
scool2_sum = np.zeros(ncool+1)
iz         = np.zeros(nzhist)
nin_av      = np.zeros(ncool+1)
nin_er      = np.zeros(ncool+1)
scool_av    = np.zeros(ncool+1)
scool_er    = np.zeros(ncool+1)
xcool_sum  = np.zeros(n_p)
xcool2_sum = np.zeros(n_p)
xcool_av   = np.zeros(n_p)
xcool_er   = np.zeros(n_p)
x2cool_sum    = np.zeros(n_p)
x2cool2_sum   = np.zeros(n_p)
x2cool_av     = np.zeros(n_p)
x2cool_er     = np.zeros(n_p)
x2cool_sub_av = np.zeros(n_p)
x2cool_sub_er = np.zeros(n_p)
x3cool_sum  = np.zeros(n_p)
x3cool2_sum = np.zeros(n_p)
x3cool_av   = np.zeros(n_p)
x3cool_er   = np.zeros(n_p)

#starting configuration-----------------------------------------|
x = fn.periodic_starting_conf(n, f, mode)
#---------montecarlo generations--------------------------------|
for i in tqdm(range(nmc)):
    nconf += 1
    if i == neq:       #when we reach equilibrium all is set to zero
        nconf = 0
        ncor = 0
        ncoolconf = 0
        ncoolcor = 0
        xcor_sum  = np.zeros(n_p)
        x2cor_sum = np.zeros(n_p)
        x3cor_sum = np.zeros(n_p)
        xcor2_sum = np.zeros(n_p)
        x2cor2_sum = np.zeros(n_p)
        x3cor2_sum = np.zeros(n_p)
        xcool_sum  = np.zeros(n_p)
        xcool2_sum = np.zeros(n_p)
        xcool_av   = np.zeros(n_p)
        xcool_er   = np.zeros(n_p)
        x2cool_sum    = np.zeros(n_p)
        x2cool2_sum   = np.zeros(n_p)
        x2cool_av     = np.zeros(n_p)
        x2cool_er     = np.zeros(n_p)
        x2cool_sub_av = np.zeros(n_p)
        x2cool_sub_er = np.zeros(n_p)
        x3cool_sum  = np.zeros(n_p)
        x3cool2_sum = np.zeros(n_p)
        x3cool_av   = np.zeros(n_p)
        x3cool_er   = np.zeros(n_p) 
        nin_sum    = np.zeros(ncool+1)
        nin2_sum   = np.zeros(ncool+1)
        scool_sum  = np.zeros(ncool+1)
        scool2_sum = np.zeros(ncool+1)
        iz         = np.zeros(nzhist)
        nin_av      = np.zeros(ncool+1)
        nin_er      = np.zeros(ncool+1)
        scool_av    = np.zeros(ncool+1)
        scool_er    = np.zeros(ncool+1)
    x = fn.periodic_update(x,n,a,f, dx)  #metropolis algorithm implementation with periodic boundary conditions
    xs = np.copy(x)
    #computation of action, kinetic energy, potential and virial for the current configuration
    S, V, T, TV = fn.compute_energy(x, n, a, f)
    S_sum += S
    S2_sum += S**2
    T_sum += T
    T2_sum += T**2
    V_sum += V
    V2_sum += V**2 
    TV_sum += TV
    TV2_sum += TV**2
    for k in range(n):
         x_sum  += x[k]
         x2_sum += x[k]**2
         x4_sum += x[k]**4
         x8_sum += x[k]**8
    ipa = np.zeros(nc)   #here will be stored initial evaluation points of correlations
    for ic in range(nc):
        ncor += 1 
        xcor, ipa[ic] = fn.correlations_functions_ipa(x, n, n_p)
        xcor_sum = np.add(xcor_sum, xcor)
        xcor2_sum = np.add(xcor2_sum, xcor**2)
        x2cor_sum = np.add(x2cor_sum, xcor**2)
        x2cor2_sum = np.add(x2cor2_sum, xcor**4)
        x3cor_sum = np.add(x3cor_sum, xcor**3)
        x3cor2_sum = np.add(x3cor2_sum, xcor**6)
    if i % kp == 0:
         config1.write("{}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(i, S, T, V))
         config2.write('configuration: '+ str(ncor) + '\n' )
         for j in range(n):
             config2.write("{:.4f}\t{:.4f}\n".format(j*a, x[j]))
         config2.write("------------------\n")
    #cooling sweeps
    if i % kp2 == 0:
        ncoolconf += 1
        
        ni, na, z     = fn.instantons(a, n, xs)
        Sc, Vc, Tc, TVc = fn.compute_energy(xs, n, a, f)
        nin = ni + na
        nin_sum[0]   += nin
        nin2_sum[0]  += nin**2
        scool_sum[0] += Sc
        scool2_sum[0]+= Sc**2
        for icool in range(1,ncool+1):                     
           xs = fn.cooling_update(xs, n, a, f, dx)
           
           ni, na, z     = fn.instantons(a, n, xs)
           Sc, Vc, Tc, TVc = fn.compute_energy(xs, n, a, f)
           nin = ni + na
           nin_sum[icool]   += nin
           nin2_sum[icool]  += nin**2
           scool_sum[icool] += Sc
           scool2_sum[icool]+= Sc**2
        #--------------cooled configuration: instanton distribution  -------------|                          
        fn.instanton_distribution(z, nin, tmax, stzhist, nzhist, iz)
        #--------------cooled correlator------------------------------------------|
        for ic in range(nc):
            ipa[ic] = int((n-n_p)*random.random())
            ncoolcor += 1
            xcor_cool = fn.cool_correlations_functions(xs, int(ipa[ic]), n, n_p)
            xcool_sum = np.add(xcool_sum, xcor_cool)
            xcool2_sum = np.add(xcool2_sum, xcor_cool**2)
            xcool_sum = np.add(xcool_sum, xcor_cool)
            xcool2_sum = np.add(xcool2_sum, xcor_cool**2)
            x2cool_sum = np.add(x2cool_sum, xcor_cool**2)
            x2cool2_sum = np.add(x2cool2_sum, xcor_cool**4)
            x3cool_sum = np.add(x3cool_sum, xcor_cool**3)
            x3cool2_sum = np.add(x3cool2_sum, xcor_cool**6)
       #----------------output cooled configurations------------------------------
        config_cool.write('configuration: ' + str(ncor)+'\n')
        for j in range(n):
              config_cool.write("{:.4f}\t{:.4f}\n".format(j*a, xs[j]))
        config_cool.write("------------------\n")
#----------Averages and error computations----------------------|
stot_av,stot_err = fn.dispersion(nconf,S_sum,S2_sum)
vtot_av,vtot_err = fn.dispersion(nconf,V_sum,V2_sum)
ttot_av,ttot_err = fn.dispersion(nconf,T_sum,T2_sum)
tvir_av,tvir_err = fn.dispersion(nconf,TV_sum,TV2_sum)
x_av,x_err       = fn.dispersion(nconf*n,x_sum,x2_sum)
x2_av,x2_err     = fn.dispersion(nconf*n,x2_sum,x4_sum)
x4_av,x4_err     = fn.dispersion(nconf*n,x4_sum,x8_sum)
for ip in range(n_p):
    xcor_av[ip],xcor_er[ip]   = fn.dispersion(ncor,xcor_sum[ip],xcor2_sum[ip])
    x2cor_av[ip],x2cor_er[ip] = fn.dispersion(ncor,x2cor_sum[ip],x2cor2_sum[ip])
    x3cor_av[ip],x3cor_er[ip] = fn.dispersion(ncor,x3cor_sum[ip],x3cor2_sum[ip])
    xcool_av[ip] , xcool_er[ip]  = fn.dispersion(ncoolcor, xcool_sum[ip] , xcool2_sum[ip]) 
    x2cool_av[ip], x2cool_er[ip] = fn.dispersion(ncoolcor, x2cool_sum[ip], x2cool2_sum[ip]) 
    x3cool_av[ip], x3cool_er[ip] = fn.dispersion(ncoolcor, x3cool_sum[ip], x3cool2_sum[ip]) 
#   instanton density, cooled action                                       
#------------------------------------------------------------------------------

for ic in range(ncool+1):
    nin_av[ic], nin_er[ic] = fn.dispersion(ncoolconf, nin_sum[ic]  , nin2_sum[ic]) 
    scool_av[ic], scool_er[ic] = fn.dispersion(ncoolconf, scool_sum[ic], scool2_sum[ic]) 

#densities averages   
v_av   = vtot_av/tmax
v_err  = vtot_err/tmax
t_av   = ttot_av/tmax
t_err  = ttot_err/tmax
tv_av  = tvir_av/tmax
tv_err = tvir_err/tmax
e_av   = v_av + tv_av
e_err  = np.sqrt(v_err**2 + tv_err**2)

#output vari
averages.write("\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(stot_av,stot_err, v_av, v_err, t_av,tv_av,tv_err)) 
averages.write('\t e_av, e_err, x_av, x_err, x2_av,x2_err, x4_av, x4_err\n')  
averages.write("\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(e_av, e_err, x_av, x_err, x2_av, x2_err, x4_av, x4_err))
#----------log derivatives------------------------------------------------------------|"""
dx, dxe = fn.log_derivatives(xcor_av, xcor_er, a)
x2sub_av, x2sub_er = fn.substract(x2cor_av, x2cor_er) #substracting
dx2, dxe2 = fn.log_derivatives(x2sub_av, x2sub_er, a)
dx3, dxe3 = fn.log_derivatives(x3cor_av, x3cor_er, a)

dxc, dxec = fn.log_derivatives(xcool_av, xcool_er, a)
x2subc_av, x2subc_er = fn.substract(x2cool_av, x2cool_er) #substracting
dx2c, dxe2c = fn.log_derivatives(x2subc_av, x2subc_er, a)
dx3c, dxe3c = fn.log_derivatives(x3cool_av, x3cool_er, a)

#output log derivatives
for ip in range(n_p-1):
    correlations.write("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(ip*a, xcor_av[ip], xcor_er[ip], dx[ip], dxe[ip]))
    correlations2.write("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(ip*a, x2cor_av[ip], x2cor_er[ip], dx2[ip], dxe2[ip]))
    correlations3.write("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(ip*a, x3cor_av[ip], x3cor_er[ip], dx3[ip], dxe3[ip]))
    correlations_cool.write("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(ip*a, xcool_av[ip], xcool_er[ip], dxc[ip], dxec[ip]))
    correlations2_cool.write("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(ip*a, x2cool_av[ip], x2cool_er[ip], dx2c[ip], dxe2c[ip]))
    correlations3_cool.write("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(ip*a, x3cool_av[ip], x3cool_er[ip], dx3c[ip], dxe3c[ip]))

'''for ic in range(ncool+1):
    instdensity.write("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(ic, nin_av[ic], nin_er[ic], de*tmax, de2*tmax))
'''
for ic in range(ncool+1):    
    sin = nin_av[ic]*s0
    coolingsweeps.write("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(ic, scool_av[ic], scool_er[ic], sin))
'''
for ic in range(ncool+1):
    si_av= scool_av[ic]/nin_av[ic]                    
    del2 =(scool_er[ic]/scool_av[ic])**2+(nin_er[ic]/nin_av[ic])**2
    si_er= si_av*np.sqrt(del2)
    inst_action.write("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(ic, si_av, si_er, s0))
 '''   
for i in range(nzhist): 
    xx = (i+0.5)*stzhist
    histz.write("{:.4f}\t{}\n".format(xx, iz[i]))
#closing files
config1.close()
config2.close()
config_cool.close()
averages.close()
correlations.close()
correlations_cool.close()
correlations2.close()
correlations2_cool.close()
correlations3.close()
correlations3_cool.close()
#instdensity.close()
coolingsweeps.close()
#inst_action.close()
histz.close()
