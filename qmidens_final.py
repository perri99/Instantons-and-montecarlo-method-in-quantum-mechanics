import numpy as np
import functions as fn
import random
from tqdm import tqdm
import os

'''
------------------------------------------------------------------------------
   Lattice calculation in quantum mechanics                                                                         
------------------------------------------------------------------------------
   Action m/2(\dot x)^2+k(x^2-f^2)^2, units 2m=k=1 
------------------------------------------------------------------------------
   Calculate instanton density by adiabatically switching from gaussian
   approximation to full potential. Have d(inst)=d(gaussian)*exp(-S(non-gaussian)) 
   where S(non-gaussian)=\int d\alpha <S-S(gaussian)>_\alpha. Perform
   reference calculation of fluctuations around trivial vacuum.
------------------------------------------------------------------------------
   Instanton is placed at beta/2. anti-symmetric boundary conditions are
   used. position is fixed during update by requiring x(beta/2)=0
------------------------------------------------------------------------------
'''
#   Input parameters 
#------------------------------------------------------------------------------

def setting_inputs():
    f = 1.4 #minimum of the potential
    n = 800 #lattice points
    a = 0.05 #lattice spacing
    neq = 100 #number of equilibration sweeps
    nmc = 10**5 #number of MonteCarlo sweeps
    dx = 0.5 #width of updates
    n_alpha = 20 #number max of points in the correlation functions
    nc = 5 #number of correlator measurements in a configuration
    kp = 50 #number of sweeps between writeout of complete configuration 
    mode = 1 # ih=0: cold start, x_i=-f; ih=1: hot start, x_i=random
    seed = 597
    w0 = 5.6
    return f, n, a, neq, nmc, dx, n_alpha, nc, kp, mode, seed, w0
#-------initialize inputs------------------------------------------------------|
f, n, a, neq, nmc, dx, n_alpha, nc, kp, mode, seed, w0 = setting_inputs()
random.seed(seed)
pi     = np.pi
dalpha = 1.0/float(n_alpha)
beta   = n*a
tau0   = beta/2.0
s0   = 4.0/3.0*f**3     #classical action for instanton solution
dens = 8*np.sqrt(2.0/pi)*f**2.5*np.exp(-s0)   #unperturbed tunneling rate
f0   = dens
#---------opening files--------------------------------------------------------|
data_folder = 'Data'
subfolder = 'qmidens' 
if not os.path.exists(data_folder):
    os.makedirs(data_folder)
folder_path = os.path.join(data_folder, subfolder)
if not os.path.exists(folder_path):
        os.makedirs(folder_path)

qmidens = open('Data/qmidens/qmidens_test.dat','w')
qmidens.write('lattice qmiden\n----------\n f, n, a, nmc, neq, dx, n_alpha\n')
qmidens.write("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(f, n, a, nmc, neq, dx, n_alpha))
qmidens.write('fluctuations around instanton path\n ---------------------\n')

idens_conf = open('Data/qmidens/idens_conf_test.dat', 'w')
idens_conf.write('tau, x[i]\n')

vac_conf = open('Data/qmidens/vac_conf_test.dat','w')
vac_conf.write('tau, x[i]\n')
#---------variable initialization----------------------------------------------|
Sinst_tot = 0
Sinst_sum = 0 #Action
S2inst_sum = 0
Vavinst_sum = 0 #Potential
Vav2inst_sum = 0
Valphainst_sum = 0 #Potential
Valpha2inst_sum = 0
Svac_tot = 0
Svac_sum = 0 #Action
S2vac_sum = 0
Vavvac_sum = 0 #Potential
Vav2vac_sum = 0
Valphavac_sum = 0 #Potential
Valpha2vac_sum = 0
sng = 0.0
svacng = 0.0
#---------array definitions----------------------------------------------------|
t = np.linspace(0,40,801)
w_vac = np.zeros(n)
Vainst_av = np.zeros(2*n_alpha+1)
Vainst_err = np.zeros(2*n_alpha+1)
Vavac_av = np.zeros(2*n_alpha+1)
Vavac_err = np.zeros(2*n_alpha+1)
#------initialize instanton and gaussian potential-----------------------------|
x_inst = fn.initialize_instanton(n, a, f, tau0)
x0_inst = np.copy(x_inst) 
x_vac = fn.initialize_vacuum(n,f)
x0_vac = np.copy(x_vac)

w_inst = -4.0*(f**2-3.0*x0_inst**2)
v_inst = (x0_inst**2-f**2)**2 

w_vac = np.full(n+1, 8 * f**2)
v_vac = np.zeros(n+1)
#initial actionn
Sinst_tot = fn.action(x_inst, n, a, f)
Svac_tot = fn.action(x_vac, n, a, f)
qmidens.write('f0\t Sinst_tot\t s0\n')
qmidens.write('{:.4f}\t{:.4f}\t{:.4f}\n'.format(f0, Sinst_tot, s0))
for ialpha in tqdm(range((2 * n_alpha + 1))):
    if ialpha <= n_alpha:
        alpha = ialpha * dalpha  #da 0 a 1
    else:
        alpha = 2.0 - ialpha * dalpha #da 1 a 0
    nconf = 0
    ncor  = 0
    #---------montecarlo generations-------------------------------------------|
    for i in range(nmc):
        nconf += 1
        if i == neq:
            nconf = 0
            
            Svac_sum = 0 #Action
            S2vac_sum = 0
            Vavvac_sum = 0 #Potential
            Vav2vac_sum = 0
            Valphavac_sum = 0 #Potential
            Valpha2vac_sum = 0
            
            Sinst_sum = 0 #Action
            S2inst_sum = 0
            Vavinst_sum = 0 #Potential
            Vav2inst_sum = 0
            Valphainst_sum = 0 #Potential
            Valpha2inst_sum = 0
            
        x_inst = fn.update_instanton(x_inst, x0_inst, w_inst, v_inst, n, a, f, alpha, dx)
        x_vac = fn.update_vacuum(x_vac, x0_vac, w_vac, v_vac, n, a, f, alpha, dx)
        
        S_inst = fn.action_fluctuations(x_inst, x0_inst, w_inst, v_inst, n, f, a, alpha)
        S_vac = fn.action_fluctuations(x_vac, x0_vac, w_vac, v_vac, n, f, a, alpha)
        
        V_inst = fn.fluctuations_total_potential(x_inst, x0_inst, w_inst, v_inst, f, a, alpha, n)
        V_vac = fn.fluctuations_total_potential(x_vac, x0_vac, w_vac, v_vac, f, a, alpha, n)
        
        Valpha_inst = fn.delta_V(x_inst, x0_inst, w_inst, v_inst, f, a, n)
        Valpha_vac = fn.delta_V(x_vac, x0_vac, w_vac, v_vac, f, a, n)
        
        Sinst_sum  += S_inst
        S2inst_sum += S_inst**2
        Vavinst_sum  += V_inst/beta
        Vav2inst_sum += V_inst**2/beta
        Valphainst_sum  += Valpha_inst/beta
        Valpha2inst_sum += Valpha_inst**2/beta
        
        Svac_sum  += S_vac
        S2vac_sum += S_vac**2
        Vavvac_sum  += V_vac/beta
        Vav2vac_sum += V_vac**2/beta
        Valphavac_sum  += Valpha_vac/beta
        Valpha2vac_sum += Valpha_vac**2/beta
       
           
    #---------end of montecarlo generations------------------------------------|
    stotinst_av, stotinst_err     = fn.dispersion(nconf, Sinst_sum, S2inst_sum)
    vinst_av, vinst_err           = fn.dispersion(nconf, Vavinst_sum, Vav2inst_sum)
    valphainst_av, valphainst_err = fn.dispersion(nconf,Valphainst_sum, Valpha2inst_sum)
       
    Vainst_av[ialpha]  = valphainst_av
    Vainst_err[ialpha] = valphainst_err
        
    stotvac_av, stotvac_err     = fn.dispersion(nconf, Svac_sum, S2vac_sum)
    vvac_av, vvac_err           = fn.dispersion(nconf, Vavvac_sum, Vav2vac_sum)
    valphavac_av, valphavac_err = fn.dispersion(nconf,Valphavac_sum, Valpha2vac_sum)
       
    Vavac_av[ialpha]  = valphavac_av
    Vavac_err[ialpha] = valphavac_err
        
    if ialpha % (2 * n_alpha) == 0:
        da = dalpha / 4.0
    else:
        da = dalpha / 2.0   #chiedere informazioni
    dsng = da * valphainst_av
    sng += dsng
    dsvacng = da * valphavac_av
    svacng += dsvacng
        
    #outputs
    qmidens.write('alpha\t stot_av\t stot_err\n')
    qmidens.write("{:.2f}\t{:.4f}\t{:.4f}\n".format(alpha,stotinst_av,stotinst_err))
    qmidens.write('vinst_av\t vinst_err\t valphainst_av\t valphainst_err\t sng\t dsng\n')
    qmidens.write("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(vinst_av, vinst_err,valphainst_av,valphainst_err,sng,dsng))
    qmidens.write('stotvac_av\t stotvac_err\t vvac_av\t vvac_err\n')
    qmidens.write("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format( stotvac_av,stotvac_err, vvac_av, vvac_err))
    qmidens.write('valphavac_av\t valphavac_err\t svacng\t dsvacng\n')
    qmidens.write("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format( valphavac_av,valphavac_err,svacng,dsvacng))
    qmidens.write('\n --------------------------------- \n\n')
    for j in range(n):
        idens_conf.write("{:.4f}\t{:.4f}\n".format(j*a, x_inst[j]))
        vac_conf.write("{:.4f}\t{:.4f}\n".format(j*a, x_vac[j]))
            
#----------------end of loops over coupling constant alpha---------------------|
#------have sum=1/2(up+down) and up = 1/2*f0+f1+...+1/2*fn, down=..------------|
supinst_sum, supinst_err, supinst_hal, sdwinst_sum, sdwinst_err, sdwinst_hal = fn.summing(n_alpha, dalpha, Vainst_av, Vainst_err)
svacup_sum, svacup_err, svacup_hal, svacdw_sum, svacdw_err, svacdw_hal = fn.summing(n_alpha, dalpha, Vainst_av, Vainst_err)
#   uncertainties                                                                         
#------------------------------------------------------------------------------
ds     = supinst_sum + sdwinst_sum
sng    = ds
dens_ng= dens*np.exp(-sng)
ds_err = np.sqrt(supinst_err + sdwinst_err)
ds_hal = supinst_hal + sdwinst_hal
ds_dif = abs(supinst_sum - sdwinst_sum)
ds_dis = abs(ds - ds_hal)/2.0
ds_tot = np.sqrt(ds_err**2 + ds_dif**2 + ds_dis**2)
dens_er= dens_ng*ds_tot 

dsvac     = svacup_sum + svacdw_sum
svacng    = dsvac
fvac      = np.exp(-svacng)
dsvac_err = np.sqrt(svacup_err + svacdw_err)
dsvac_hal = svacup_hal + svacdw_hal
dsvac_dif = abs(svacup_sum - svacdw_sum)
dsvac_dis = abs(dsvac - dsvac_hal)/2.0
dsvac_tot = np.sqrt(dsvac_err**2 + dsvac_dif**2 + dsvac_dis**2)
fvac_er   = fvac*dsvac_tot 
#------------------------------------------------------------------------------
# output                                                                         
qmidens.write('sng\t ds_tot\t ds_err\t ds_dif\t ds_dis\t svacng\n')
qmidens.write("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n\n".format(sng, ds_tot, ds_err, ds_dif, ds_dis, svacng))
qmidens.write('dsvac_tot\t dsvac_err\t dsvac_dif\t dsvac_dis\t fvac\t fvac_err\n')
qmidens.write("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n\n".format(dsvac_tot, dsvac_err, dsvac_dif, dsvac_dis, fvac, fvac_er))
# final answer                                                                          
#------------------------------------------------------------------------------
seff    = sng - svacng
seff_er = np.sqrt(ds_tot**2+dsvac_tot**2)
dens_ng = dens*np.exp(-seff)
dens_er = dens_ng*seff_er
#------------------------------------------------------------------------------
#   output                                                                         
#------------------------------------------------------------------------------
qmidens.write('seff\t, sng\t svacng\t seff\t seff_er\n')
qmidens.write("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(seff, sng, svacng, seff, seff_er))
qmidens.write('dens_ng\t dens\t dens_ng\t dens_er\n')
qmidens.write("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(dens_ng, dens, dens_ng, dens_er))

qmidens.close()
