import numpy as np
import os
import random
import numba as nb
#------------------------------------------------------------------------------|
'''
Functions needed in qm.py, qmswitch.py, qmcool.py, qmidens.py
    rilm.py, rilmgauss.py, iilm.py
'''
#------------------------------------------------------------------------------|
#-------------------------------------------------------------------------------------|
def directory(name):
    '''
    check if 'Data/name' directory exists. If not create it

    Parameters
    ----------
    name : string
        subfolder name.

    Returns
    -------
    None.

    '''
    data_folder = 'Data'
    subfolder = name 
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    folder_path = os.path.join(data_folder, subfolder)
    if not os.path.exists(folder_path):
            os.makedirs(folder_path)
#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------       
@nb.jit(nopython = True)
def normalization(x,dx):
    '''
    returns an array normalized to 1
    Parameters:
        x: array-like
        dx: floating type
    Return:     
        x/norm: array-like
    '''
    norm = np.sqrt(np.sum(x**2 * dx))
    return x/norm
#-------------------------------------------------------------------------------------|
#------------------------------------------------------------------------------|
@nb.jit(nopython = True)
def new_config(x,n,N_inst, z, f, a):
    '''
    

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    n : TYPE
        DESCRIPTION.
    N_inst : TYPE
        DESCRIPTION.
    z : TYPE
        DESCRIPTION.
    f : TYPE
        DESCRIPTION.
    a : TYPE
        DESCRIPTION.

    Returns
    -------
    x : TYPE
        DESCRIPTION.

    '''
    for j in range(1,n):
        t = a*j
        x[j] = sum_ansatz_path(N_inst, z, f, t)
    x[0] = x[n-1]
    x[n] = x[1]
    return x
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
@nb.jit(nopython = True)
def action(x,n,a,f): 
    '''
    total euclidean action of configuration x

    Parameters
    ----------
    x : array(float)
        configuration array.
    n : int
        number of lattice point.
    a : float
        lattice discretization.
    f : float
        double well separation.

    Returns
    -------
    S : float
        euclidean action.

    '''
    S = kinetic(x, n, a) + potential(x, n, a, f)
    return S
#-----------------------------------------------------------
#-----------------------------------------------------------
@nb.jit(nopython = True)
def action_fluctuations(x,x0,w,vi, n,f,a, alpha):
    return kinetic(x, n, a) + fluctuations_total_potential(x, x0, w, vi, f, a, alpha, n)    

@nb.jit(nopython = True)
def V0(x,x0,w,vi,j):
    return 1.0/2.0 * w[j] * (x[j]-x0[j])**2 + vi[j]

@nb.jit(nopython = True)
def delta_V(x,x0,w,vi,f,a, n):
    delta_V = 0
    for j in range(n):
        delta_V += (x[j]**2-f**2)**2 - V0(x, x0, w, vi, j)
    return a*delta_V

@nb.jit(nopython = True)
def action_switch(x,n,a,w0,f,alpha):
    S_switch = kinetic(x, n, a) + potential_switch(x, n, a, f, w0, alpha)
    return S_switch

@nb.jit(nopython = True)
def action_local(x, j, n, a, f): #contribution to the action from the j-th point of configuration
    return (1/(4*a))*(x[j]-x[j-1])**2+a*(x[j]**2-f**2)**2 + (1/(4*a))*(x[j+1]-x[j])**2

@nb.jit(nopython = True)
def kinetic(x,n,a):
    K = 0
    for j in range(n):
        K += (1/(4*a))*(x[j+1]-x[j])**2
    return K

@nb.jit(nopython = True)
def kinetic_local(x,y,a): 
    K = (1/(4*a))*(x-y)**2
    return K

@nb.jit(nopython = True)
def potential_gauss(w,x,y,a):
    return 0.5*w*a*(x-y)**2

@nb.jit(nopython = True)
def harmonic_potential(x,w0):
    return (1/4) * x**2 * w0**2    #mass = 0.5

@nb.jit(nopython = True)
def action_switch_local(x, j, n, a, f, w0, alpha):
    return (1/(4*a))*(x[j+1]-x[j])**2 + (1/(4*a))*(x[j]-x[j-1])**2 + a*(alpha*((x[j]**2-f**2)**2-harmonic_potential(x[j], w0)) + harmonic_potential(x[j], w0)) 

@nb.jit(nopython = True)
def potential(x,n,a,f):
    V = 0
    for j in range(n):
        if x is not None:
            V += a*(x[j]**2-f**2)**2
    return V

@nb.jit(nopython = True)
def potential_switch_local(x, j, n, a, f, w0, alpha):
    return a*(alpha*((x[j]**2-f**2)**2-harmonic_potential(x[j], w0)) + harmonic_potential(x[j], w0))

@nb.jit(nopython = True)
def potential_switch(x, n, a, f, w0, alpha):
    V_switch = 0
    for j in range(n):
        V_switch += potential_switch_local(x, j, n, a, f, w0, alpha)
    return V_switch

@nb.jit(nopython = True)
def potential_diff(x, n, a, f, w0):
    V_diff = 0
    for j in range(n):
        V_diff += a*((x[j]**2-f**2)**2-harmonic_potential(x[j], w0))
    return V_diff

@nb.jit(nopython = True)
def hardcore_interaction(z, nin, tcore, score, tmax, s0):
    shc = 0.0
    if tcore == 0:
        return shc
    for i in range(nin):
        if i == 0:
            zm = z[-1] - tmax
        else:
            zm = z[i-1]
        dz = z[i] - zm
        shc = shc + score * np.exp(-dz/tcore)
    return shc
#----------------------------------------------------------------------------------
@nb.jit(nopython = True)
def periodic_starting_conf(n, f,mode):
    '''
    initialize the starting configuration array. if mode == 0 cold start, all elements\
        are initialize to -f otherwise uniform distribution between -f+f(hot start).\
            periodic boundary conditions are implemented

    Parameters
    ----------
    n : int
        lattice points.
    f : float
        position of minimum potential.
    mode : int
        0 == cold start, otherwise hot start.

    Returns
    -------
    x : array(float)
        initial configuration of dimension n+1.

    '''
    x = np.zeros(n)
    if mode == 0: #cold start
        x = np.full(n, -f)
    else:
        x = np.random.uniform(-f,f,size = n) #hotstart
    #periodic boundary condition
    x[n-1] = x[0]
    x = np.append(x,x[1])
    return x
#---------------------------------------------------------------------------------
@nb.jit(nopython = True)
def periodic_update(x,n,a,f,dx):
    '''
    

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    n : TYPE
        DESCRIPTION.
    a : TYPE
        DESCRIPTION.
    f : TYPE
        DESCRIPTION.
    dx : TYPE
        DESCRIPTION.

    Returns
    -------
    x : TYPE
        DESCRIPTION.

    '''
    for j in range(1,n):
        S_old = action_localc(x[j], x[j-1], x[j+1], n, a, f)
        xnew  = x[j] + dx*(2.0*random.random()-1.0)
        S_new = action_localc(xnew, x[j-1], x[j+1], n, a, f)
        delta_S = S_new-S_old
        delta_S = min(delta_S, 70.0)
        delta_S = max(delta_S, -70.0)
        if np.exp(-delta_S) > random.random():
            x[j] = xnew
    x[n-1] = x[0]
    x[n] = x[1]
    return x

@nb.jit(nopython = True)
def initialize_instanton(n,a, f, tau0):
    x = np.zeros(n)
    for i in range(n):
        tau   = i*a
        x[i]  = f*np.tanh(2.0*f*(tau-tau0))#instanton solution
    #antiperiodic boundary condition
    x[n-1] = -x[0]
    x  = np.append(x, -x[1])
    return x

@nb.jit(nopython = True)
def initialize_vacuum(n,f):
    x = np.zeros(n)
    for i in range(n):
        x[i] = f
    x[0] = x[n-1]
    x = np.append(x,x[1])
    return x

@nb.jit(nopython = True)
def compute_energy(x,n, a ,f):
    V = potential(x,n, a, f)
    T = kinetic(x,n, a)
    S = V + T
    tvtot = 0.0
    for j in range(n):
        tv = 2.0*x[j]**2*(x[j]**2-f**2)
        tvtot += a*tv
    return S, V, T, tvtot

@nb.jit(nopython = True)
def compute_energy_switch(x,n, a ,f, w0, alpha):
    V = potential_switch(x, n, a, f, w0, alpha)
    T = kinetic(x,n, a)
    S = V + T
    P = potential_diff(x, n, a, f, w0)
    return S, V, P

@nb.jit(nopython = True)
def update_periodic(x, n, a, f, dx): #metropolis alghorhitm
    for j in range(1,n):
        S_old = action_local(x, j, n, a, f)
        x_old = x[j]
        x[j]  = x[j] + dx*(2.0*random.random()-1.0)
        S_new = action_local(x, j, n, a, f)
        delta_S = S_new-S_old
        delta_S = min(delta_S, 69.0)
        delta_S = max(delta_S, -69.0)
        if delta_S >= 0 and np.exp(-delta_S) <= random.random():
            x[j] = x_old
    x[n-1] = x[0]
    x[n] = x[1]
    return x

@nb.jit(nopython = True)
def update_periodic_switch(x,n,a,f, w0, alpha, dx): #metropolis alghorhitm for adiabatic switching
    for j in range(n):
        S_old = action_switch_local(x, j, n, a, f, w0, alpha)
        x_old = x[j]
        x[j]  = x[j] + dx*(2.0*random.random()-1.0)
        S_new = action_switch_local(x, j, n, a, f, w0, alpha)
        delta_S = S_new-S_old
        delta_S = min(delta_S, 69.0)
        delta_S = max(delta_S, -69.0)
        if delta_S > 0 and np.exp(-delta_S) <= random.random():
            x[j] = x_old
    x[n-1] = x[0]
    x[n] = x[1]
    return x

@nb.jit(nopython = True)
def cooling_update(xs,n,a,f,dx):
    '''
    metropolis algorithm for cooling. we accept only update which lower the action 

    Parameters
    ----------
    xs : array(float)
        initial configuration.
    n : int
        number of lattice poits.
    a : float
        lattice discretization.
    f : float
        double well separation.
    dx : float
        update width.

    Returns
    -------
    xs : array(float)
        cooled configuration.

    '''
    nhit2 = 10
    delxp= 0.1*dx
    for j in range(1,n):
        Sold2 = action_localc(xs[j], xs[j-1], xs[j+1], n, a, f)
        for w in range(nhit2):                                
            xnew2 = xs[j] + delxp*(2.0*random.random()-1.0)
            Snew2 = action_localc(xnew2, xs[j-1], xs[j+1], n, a, f)
            if Snew2 < Sold2:
                xs[j] = xnew2
    return xs

@nb.jit(nopython = True)
def heating_update(x_hot, x, n, nheat, a, f, dx):
    w = -4.0*(f**2-3.0*x**2)
    for ih in range(nheat):
        for j in range(1, n): 
            S_old = kinetic_local(x_hot[j+1],x_hot[j],a)+kinetic_local(x_hot[j],x_hot[j-1],a)+potential_gauss(w[j], x_hot[j], x[j], a)
            xmin = abs(f*np.tanh(f*a))
            if abs(x[j]) < xmin:
                continue
            #   update  
            #------------------------------------------------------------------
            xnew = x_hot[j] + dx*(2.0*random.random()-1.0)
            S_new = kinetic_local(x_hot[j+1],xnew, a)+kinetic_local(xnew,x_hot[j-1],a)+potential_gauss(w[j], xnew, x[j], a)
            delta_S = S_new-S_old
            delta_S = min(delta_S, 70.0)
            delta_S = max(delta_S, -70.0)
            if np.exp(-delta_S) > random.random():
                x_hot[j] = xnew
               
        x_hot[n-1]= x_hot[0]
        x_hot[n]  = x_hot[1]
    return x_hot

@nb.jit(nopython = True)
def action_localc(xj, xjm, xjp, n, a, f): #contribution to the action from the j-th point of configuration
    return (1/(4*a))*(xj-xjm)**2 + a*(xj**2-f**2)**2 + (1/(4*a))*(xjp-xj)**2

@nb.jit(nopython = True)
def fluctuations_potential(x,x0,w,vi,f,a, alpha, j):
    delta_V = (x[j]**2-f**2)**2 - V0(x, x0, w, vi, j)
    return a*(alpha*delta_V + V0(x, x0, w, vi, j))

@nb.jit(nopython = True)
def fluctuations_total_potential(x,x0,w,vi,f,a, alpha, n):
    V = 0
    for j in range(n):
        V += fluctuations_potential(x, x0, w, vi, f, a, alpha, j)
    return  V



@nb.jit(nopython = True)
def update_instanton(x, x0, w, vi, n, a, f, alpha, dx):
    n0     = n/2
    n0p    = int(n0+1)
    n0m    = int(n0-1)
    for j in range(1,n):
        if j==n0:
           continue
        S_old = (1/(4*a))*(x[j]-x[j-1])**2 + (1/(4*a))*(x[j+1]-x[j])**2 + fluctuations_potential(x,x0,w,vi,f,a, alpha, j)
        x_old = x[j]
        if j == n0m or j == n0p:
                vel   = (x[n0p] - x[n0m]) / (2.0 * a)
                sjak  = -np.log(abs(vel))
                S_old += sjak
        x[j]  = x[j] + dx*(2.0*random.random()-1.0)
        S_new = (1/(4*a))*(x[j]-x[j-1])**2 + (1/(4*a))*(x[j+1]-x[j])**2 + fluctuations_potential(x,x0,w,vi,f,a, alpha, j)
        if j == n0m:
                vel   = (x[n0p] - x[j]) / (2.0 * a)
                sjak  = -np.log(abs(vel))
                S_new += sjak
        elif j == n0p:
                vel   = (x[j] - x[n0m]) / (2.0 * a)
                sjak  = -np.log(abs(vel))
                S_new += sjak
        delta_S = S_new-S_old
        delta_S = min(delta_S, 69.0)
        delta_S = max(delta_S, -69.0)
        if delta_S > 0 and np.exp(-delta_S) < random.random():
            x[j] = x_old
    x[0] = -x[n-1]
    x[n] = -x[1]
    return x

@nb.jit(nopython = True)
def update_interacting_instanton(nin, z, tmax,tcore, score, dz, x, n, a, f, s0):
    for iin in range(nin):
        sold   = action(x,n,a,f)
        sold  += hardcore_interaction(z, nin, tcore, score, tmax, s0)
        zstore = np.copy(z)
        zold   = z[iin]
        znew   = zold + (random.random()-0.5)*dz
        if znew > tmax:
            znew -= tmax
        if znew < -tmax:
            znew += tmax
        z[iin] = znew
        z      = np.sort(z)
        
        #----------------------------------------------------------------------
        #   calculate new action
        #----------------------------------------------------------------------
        x = new_config(x, n, nin, z, f, a)
        snew = action(x, n, a, f)
        shc = hardcore_interaction(z, nin, tcore, score, tmax, s0)
        snew += shc
        
        #----------------------------------------------------------------------
        #   accept with probability exp(-delta S)                                  
        #----------------------------------------------------------------------
        dels = snew-sold  
        if np.exp(-dels) <= random.random() :
            z = np.copy(zstore)
    return z, x


@nb.jit(nopython = True)
def update_vacuum(x,x0,w,vi,n,a,f,alpha,dx):
    for j in range(n):
        S_old = (1/(4*a))*(x[j]-x[j-1])**2 + (1/(4*a))*(x[j+1]-x[j])**2 + fluctuations_potential(x,x0,w,vi,f,a, alpha, j)
        x_old = x[j]
        x[j]  = x[j] + dx*(2.0*random.random()-1.0)
        S_new = (1/(4*a))*(x[j]-x[j-1])**2 + (1/(4*a))*(x[j+1]-x[j])**2 + fluctuations_potential(x,x0,w,vi,f,a, alpha, j)
        delta_S = S_new-S_old
        delta_S = min(delta_S, 69.0)
        delta_S = max(delta_S, -69.0)
        if delta_S > 0 and np.exp(-delta_S) < random.random():
            x[j] = x_old
    x[0] = x[n-1]
    x[n] = x[1]
    return x

@nb.jit(nopython = True)
def dispersion(n, xtot, x2tot):
    if n < 1:
        raise ValueError("Number of measurements must be at least 1")
    x_average = xtot / float(n)
    sigma2 = x2tot / float(n*n) - x_average**2 / float(n) #mean of squared elements - mean squared
    if sigma2 < 0:
        sigma2 = 0      
    xerr = np.sqrt(sigma2)  
    return x_average, xerr

@nb.jit(nopython = True)
def correlations_functions(x,n,n_p):
    xcor = np.zeros(n_p)
    ip0  = int((n-n_p)*random.random()) #prendo 5 punti di partenza a caso tra x[o] e x[779]
    x0   = x[ip0] 
    for ip in range(n_p):
            x1 = x[ip0+ip]  #scelto il punto di partenza vedo le correlazioni con i 20 successivi
            xcor[ip]  = x0*x1
    return xcor

@nb.jit(nopython = True)
def correlations_functions_ipa(x,n,n_p):
    xcor = np.zeros(n_p)
    ip0  = int((n-n_p)*random.random()) #prendo 5 punti di partenza a caso tra x[o] e x[779]
    x0   = x[ip0] 
    for ip in range(n_p):
            x1 = x[ip0+ip]  #scelto il punto di partenza vedo le correlazioni con i 20 successivi
            xcor[ip]  = x0*x1
    return xcor, ip0

@nb.jit(nopython = True)
def cool_correlations_functions(x, ip0, n, n_p):
    xcor_cool = np.zeros(n_p)
    x0_cool   = x[ip0] 
    for ip in range(n_p):
            x1_cool = x[ip0+ip]  #scelto il punto di partenza vedo le correlazioni con i 20 successivi
            xcor_cool[ip]  = x0_cool*x1_cool
    return xcor_cool

@nb.jit(nopython = True)
def log_derivatives(x,err,a,n_p):
    dx = np.zeros(n_p-1)
    dxe = np.zeros(n_p-1)
    for ip in range(n_p-1):
        dx[ip]  = (x[ip]-x[ip+1])/x[ip]/a #log derivative
        dxe2 = (err[ip+1]/x[ip])**2
        + (err[ip]*x[ip+1]/x[ip]**2)**2 #error propagation squared
        dxe[ip]  = np.sqrt(dxe2)/a
    return dx, dxe

@nb.jit(nopython = True)
def substract(x,err,n_p):
    x_sub = np.copy(x)
    err_sub = np.copy(err)
    xs = x[n_p-1]
    errs  = err[n_p-1]
    for ip in range(n_p):
        x_sub[ip] = x[ip]-xs 
        err_sub[ip] = np.sqrt(err[ip]**2+errs**2)
    return x_sub, err_sub       

@nb.jit(nopython = True)
def histogramarray(x,n, xmin, st, m, hist):
    for k in range(n):
        j = (x[k] - xmin)/st + 1.000001
        if (j < 1):
            j = 1
        if (j > m):
            j = m
        hist[int(j)-1] += 1

@nb.jit(nopython = True)
def histogramarray2(x, xmin, st, m, hist):
    j = (x - xmin)/st + 1.000001
    if (j < 1):
        j = 1
    if (j > m):
        j = m
    hist[int(j)-1] += 1

@nb.jit(nopython = True)
def instanton_distribution(z, nin, tmax, stzhist, nzhist, iz):
    for ii in range(0, nin, 2):
        if ii == 0:
            zm = z[nin] - tmax
        else:
            zm = z[ii-1]
        z0  = z[ii]
        zp  = z[ii+1]
        zia = min(zp-z0, z0-zm)    #chiedere info
        histogramarray2( zia, 0.0, stzhist, nzhist, iz)

@nb.jit(nopython = True)
def building_wavefunction(hist, nxhist, stxhist, xhist_min):
    wave_function = np.zeros(nxhist)
    position = np.zeros(nxhist)
    xnorm = 0
    for i in range(nxhist):
        xnorm += hist[i]*stxhist #sommo il valore di ogni bin per la larghezza dei bin
    for i in range(nxhist):
        position[i] = xhist_min + (i+1)*stxhist #spazzo tutto lo spazio dei bin
        wave_function[i] = hist[i]/xnorm
    return position, wave_function

@nb.jit(nopython = True)
def summing(n_alpha, dalpha, Va_av, Va_err):
    eup_sum     = 0.0
    eup_err     = 0.0
    eup_hal     = 0.0
    edw_sum     = 0.0
    edw_err     = 0.0
    edw_hal     = 0.0
    for ia in range(n_alpha+1):
        if ia % n_alpha == 0:
            da = dalpha / 4.0
        else:
            da = dalpha / 2.0
        iap      = ia + n_alpha
        eup_sum += da * Va_av[ia]
        eup_err += da * Va_err[ia] ** 2
        edw_sum += da * Va_av[iap]
        edw_err += da * Va_err[iap] ** 2
        
    for ia in range(0, n_alpha+1, 2):
        if ia % n_alpha == 0:
            da = dalpha / 2.0
        else:
            da = dalpha
        iap      = ia + n_alpha
        eup_hal += da * Va_av[ia]
        edw_hal += da * Va_av[iap]
    return eup_sum, eup_err, eup_hal, edw_sum, edw_err, edw_hal

@nb.jit(nopython = True)
def sum_ansatz_path(N_inst, z, f,t):
    neven = N_inst - (N_inst % 2)
    xsum = -f
    for i in range(0, neven, 2):
        xsum += f * np.tanh(2.0 * f * (t - z[i])) - f * np.tanh(2.0 * f * (t - z[i+1]))
    if N_inst % 2 != 0:
        xsum += f * np.tanh(2.0 * f * (t - z[N_inst])) + f   
    return xsum

@nb.jit(nopython = True)
def instantons(f, a, n, x, xi, xa, z):
    ni = 0
    na = 0
    nin= 0
    p = int(np.sign(x[0]))
    for i in range(1,n):
        tau = a * i
        ixp = int(np.sign(x[i]))
        if ixp > p:
            ni  += 1
            nin += 1
            xi[ni] = tau
            z[nin] = tau
                     
        elif ixp < p:
            na  += 1
            nin += 1 
            xa[na] = tau
            z[nin] = tau                     
        p = ixp
    return ni, na