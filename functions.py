import numpy as np
import os
import random
import numba as nb
from tqdm import tqdm
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
    returns the array x normalized to 1 w.r.t. euclidean metric \n
    Parameters:
    ------------------------
        x: array-like
        dx: floating type \n
    Return:  
    ------------------------
        x/norm: array-like
    '''
    norm = np.sqrt(np.sum(x**2 * dx))
    return x/norm
#-------------------------------------------------------------------------------------|
#------------------------------------------------------------------------------|
@nb.jit(nopython = True)
def new_config(x,n,N_inst, z, f, a):
    '''
    Initialize a configuration via sum ansatz path \
        with a number of instanton N_inst

    Parameters
    ----------
    x : array(float)
        configuration array.
    n : int
        lattice points.
    N_inst : int(even)
        total number of tunneling events(instantons+antiinstantons).
    z : array(float)
        instanton time location array.
    f : float
        double well separation.
    a : float
        lattice discretization.

    Returns
    -------
    x : array (float)
        sum ansatz path configuration.

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
def sum_ansatz_path(N_inst, z, f,t):
    '''
    Evaluation of sum ansatz configuration at time t \
        for the instantons located as in z

    Parameters
    ----------
    N_inst : int
        instantons number.
    z : array(float)
        time location of instantons.
    f : float
        position of minimum of double well.
    t : float
        time at wich the sum ansatz is evaluated.

    Returns
    -------
    xsum : float
        sum ansatz at time t.

    '''
    neven = N_inst - (N_inst % 2)
    xsum = -f
    for i in range(0, neven, 2):
        xsum += f * np.tanh(2.0 * f * (t - z[i])) - f * np.tanh(2.0 * f * (t - z[i+1]))
    if N_inst % 2 != 0:
        xsum += f * np.tanh(2.0 * f * (t - z[N_inst])) + f   
    return xsum
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
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
@nb.jit(nopython = True)
def V0(x,x0,w,vi,j):
    return 1.0/2.0 * w[j] * (x[j]-x0[j])**2 + vi[j]
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
@nb.jit(nopython = True)
def delta_V(x,x0,w,vi,f,a, n):
    delta_V = 0
    for j in range(n):
        delta_V += (x[j]**2-f**2)**2 - V0(x, x0, w, vi, j)
    return a*delta_V
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
@nb.jit(nopython = True)
def action_switch(x,n,a,w0,f,alpha):
    S_switch = kinetic(x, n, a) + potential_switch(x, n, a, f, w0, alpha)
    return S_switch
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
@nb.jit(nopython = True)
def action_local(x, j, n, a, f): #contribution to the action from the j-th point of configuration
    return (1/(4*a))*(x[j]-x[j-1])**2+a*(x[j]**2-f**2)**2 + (1/(4*a))*(x[j+1]-x[j])**2
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
@nb.jit(nopython = True)
def kinetic(x,n,a):
    K = 0
    for j in range(n):
        K += (1/(4*a))*(x[j+1]-x[j])**2
    return K
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
@nb.jit(nopython = True)
def kinetic_local(x,y,a): 
    K = (1/(4*a))*(x-y)**2
    return K
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
@nb.jit(nopython = True)
def potential_gauss(w,x,y,a):
    return 0.5*w*a*(x-y)**2
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
@nb.jit(nopython = True)
def harmonic_potential(x,w0):
    return (1/4) * x**2 * w0**2    #mass = 0.5
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
@nb.jit(nopython = True)
def action_switch_local(x, j, n, a, f, w0, alpha):
    return (1/(4*a))*(x[j+1]-x[j])**2 + (1/(4*a))*(x[j]-x[j-1])**2 + a*(alpha*((x[j]**2-f**2)**2-harmonic_potential(x[j], w0)) + harmonic_potential(x[j], w0)) 
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
@nb.jit(nopython = True)
def potential(x,n,a,f):
    V = 0
    for j in range(n):
        if x is not None:
            V += a*(x[j]**2-f**2)**2
    return V
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
@nb.jit(nopython = True)
def potential_switch_local(x, j, n, a, f, w0, alpha):
    return a*(alpha*((x[j]**2-f**2)**2-harmonic_potential(x[j], w0)) + harmonic_potential(x[j], w0))
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
@nb.jit(nopython = True)
def potential_switch(x, n, a, f, w0, alpha):
    V_switch = 0
    for j in range(n):
        V_switch += potential_switch_local(x, j, n, a, f, w0, alpha)
    return V_switch
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
@nb.jit(nopython = True)
def potential_diff(x, n, a, f, w0):
    V_diff = 0
    for j in range(n):
        V_diff += a*((x[j]**2-f**2)**2-harmonic_potential(x[j], w0))
    return V_diff
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
@nb.jit(nopython = True)
def hardcore_interaction(z, nin, tcore, score, tmax, s0):
    hardcore_action = 0.0
    if tcore == 0:
        return hardcore_action 
    for i in range(nin):
        if i == 0:
            zm = z[-1] - tmax
        else:
            zm = z[i-1]
        dz = z[i] - zm
        hardcore_action += score * np.exp(-dz/tcore)
    return hardcore_action
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
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
#---------------------------------------------------------------------------------
@nb.jit(nopython = True)
def periodic_update(x,n,a,f,dx):
    '''
    Metropolis algorithm implementation for configuration\
        with periodic boundary conditions.

    Parameters
    ----------
    x : araay(float)
        starting configuration.
    n : int
        lattice points.
    a : float
        lattice spacing.
    f : float
        location of double well minimum.
    dx : float
        update gaussian width.

    Returns
    -------
    x : array (float)
        updated configuration.

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
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
@nb.jit(nopython = True)
def initialize_instanton(n,a, f, tau0):
    '''
    create a one instanton classical solution. Instanton is placed in tau0, \
        anti periodic boundary conditions are used.

    Parameters
    ----------
    n : int
        lattice points.
    a : float
        lattice discretization.
    f : float
        location of minimum of double well.
    tau0 : float
        instanton location.

    Returns
    -------
    x : array(float)
        classical instanton configuration.

    '''
    x = np.zeros(n)
    for i in range(n):
        tau   = i*a
        x[i]  = f*np.tanh(2.0*f*(tau-tau0))#instanton solution
    #antiperiodic boundary condition
    x[n-1] = -x[0]
    x  = np.append(x, -x[1])
    return x
#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------
@nb.jit(nopython = True)
def initialize_vacuum(n,f):
    '''
    initialize a vacuum solution. Particle is placed in x = +f,\
         periodic boundary conditions are used.

    Parameters
    ----------
    n : int
        lattice points.
    f : float
        location of minimum of double well.

    Returns
    -------
    x : array float
        vacuum configuration.

    '''
    x = np.zeros(n)
    for i in range(n):
        x[i] = f
    x[0] = x[n-1]
    x = np.append(x,x[1])
    return x
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
@nb.jit(nopython = True)
def compute_energy(x,n, a ,f):
    '''
    return action, potential energy , kinetic energy\
        virial term of configuration x

    Parameters
    ----------
    x : araay(float)
        configuration.
    n : int
        lattice points.
    a : float
        lattice spacing.
    f : float
        location of double well minimum.

    Returns
    -------
    S : float
        action.
    V : float
        potential energy.
    T : float
        kinetic term.
    tvtot : float
        virial term.

    '''
    V = potential(x,n, a, f)
    T = kinetic(x,n, a)
    S = V + T
    tvtot = 0.0
    for j in range(n):
        tv = 2.0*x[j]**2*(x[j]**2-f**2)
        tvtot += a*tv
    return S, V, T, tvtot
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
@nb.jit(nopython = True)
def compute_energy_switch(x,n, a ,f, w0, alpha):
    '''
    evaluation of action and energies in adiabatic switching\
        S = S0 + alpha(S-S0)

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
    w0 : TYPE
        DESCRIPTION.
    alpha : TYPE
        DESCRIPTION.

    Returns
    -------
    S : TYPE
        DESCRIPTION.
    V : TYPE
        DESCRIPTION.
    P : TYPE
        DESCRIPTION.

    '''
    V = potential_switch(x, n, a, f, w0, alpha)
    T = kinetic(x,n, a)
    S = V + T
    P = potential_diff(x, n, a, f, w0)
    return S, V, P
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
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
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
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
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
@nb.jit(nopython = True)
def cooling_update(xs,n,a,f,dx):
    '''
    metropolis algorithm for cooling: only updates which lower the action \
        are accepted

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
    nhit2 = 10  #number of updatig trials
    delxp= 0.1*dx
    for j in range(1,n):
        Sold2 = action_localc(xs[j], xs[j-1], xs[j+1], n, a, f)
        for w in range(nhit2):                                
            xnew2 = xs[j] + delxp*(2.0*random.random()-1.0)
            Snew2 = action_localc(xnew2, xs[j-1], xs[j+1], n, a, f)
            if Snew2 < Sold2:
                xs[j] = xnew2
    return xs
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
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
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
@nb.jit(nopython = True)
def action_localc(xj, xjm, xjp, n, a, f): #contribution to the action from the j-th point of configuration
    return (1/(4*a))*(xj-xjm)**2 + a*(xj**2-f**2)**2 + (1/(4*a))*(xjp-xj)**2
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
@nb.jit(nopython = True)
def fluctuations_potential(x,x0,w,vi,f,a, alpha, j):
    delta_V = (x[j]**2-f**2)**2 - V0(x, x0, w, vi, j)
    return a*(alpha*delta_V + V0(x, x0, w, vi, j))
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
@nb.jit(nopython = True)
def fluctuations_total_potential(x,x0,w,vi,f,a, alpha, n):
    V = 0
    for j in range(n):
        V += fluctuations_potential(x, x0, w, vi, f, a, alpha, j)
    return  V
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
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
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
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
        if znew < 0:
            znew += tmax
        z[iin] = znew
        z      = np.sort(z)
        
        #----------------------------------------------------------------------
        #   calculate new action
        #----------------------------------------------------------------------
        x = new_config(x, n, nin, z, f, a)
        snew = action(x, n, a, f)
        snew +=hardcore_interaction(z, nin, tcore, score, tmax, s0)
        
        #----------------------------------------------------------------------
        #   accept with probability exp(-delta S)                                  
        #----------------------------------------------------------------------
        dels = snew-sold  
        if np.exp(-dels) <= random.random() :
            z = np.copy(zstore)
    return z, x
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
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
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
@nb.jit(nopython = True)
def dispersion(n, xtot, x2tot):
    '''
    computes mean and error

    Parameters
    ----------
    n : int
        number of samples.
    xtot : float
        total sum of n elements.
    x2tot : float
        total sum of n elements squared .

    Returns
    -------
    x_average : float
        mean over n.
    xerr : float
        standard deviation.

    '''
    if n < 1:
        raise ValueError("Number of measurements must be at least 1")
    x_average = xtot / float(n)
    sigma2 = x2tot / float(n*n) - x_average**2 / float(n) #mean of squared elements - mean squared
    if sigma2 < 0:
        sigma2 = 0      
    xerr = np.sqrt(sigma2)  
    return x_average, xerr
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
@nb.jit(nopython = True)
def correlations_functions(x,n,n_p):
    '''
    compute x(tau0)x(tau0+dtau) for random tau0 and different\
        evaluation points

    Parameters
    ----------
    x : array(float)
        configuration.
    n : int
        lattice points in euclidean time direction.
    n_p : int
        number of evaluation point.

    Returns
    -------
    xcor : array(float)
        xcor[j] = <x[tau0]x[tau0+j]>.

    '''
    xcor = np.zeros(n_p)
    ip0  = int((n-n_p)*random.random()) #prendo 5 punti di partenza a caso tra x[o] e x[779]
    x0   = x[ip0] 
    for ip in range(n_p):
            x1 = x[ip0+ip]  #scelto il punto di partenza vedo le correlazioni con i 20 successivi
            xcor[ip]  = x0*x1
    return xcor
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
@nb.jit(nopython = True)
def correlations_functions_ipa(x,n,n_p):
    xcor = np.zeros(n_p)
    ip0  = int((n-n_p)*random.random()) #prendo 5 punti di partenza a caso tra x[o] e x[779]
    x0   = x[ip0] 
    for ip in range(n_p):
            x1 = x[ip0+ip]  #scelto il punto di partenza vedo le correlazioni con i 20 successivi
            xcor[ip]  = x0*x1
    return xcor, ip0
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
@nb.jit(nopython = True)
def cool_correlations_functions(x, ip0, n, n_p):
    xcor_cool = np.zeros(n_p)
    x0_cool   = x[ip0] 
    for ip in range(n_p):
            x1_cool = x[ip0+ip]  #scelto il punto di partenza vedo le correlazioni con i 20 successivi
            xcor_cool[ip]  = x0_cool*x1_cool
    return xcor_cool
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
@nb.jit(nopython = True)
def log_derivatives(x,err,a):
    '''
    computes log derivative of x w.r.t. euclidean time

    Parameters
    ----------
    x : array
        array of wich we want to take log derivative.
    err : array
        error associated to x.
    a : float
        lattice spacing in euclidean time direction.

    Returns
    -------
    dx : array
        log derivative of x.
    dxe : array
        error associated to the log derivative.

    '''
    dx = np.zeros(x.size-1)
    dxe = np.zeros(x.size-1)
    for ip in range(x.size-1):
        dx[ip]  = (x[ip]-x[ip+1])/x[ip]/a #log derivative
        dxe2 = (err[ip+1]/x[ip])**2
        + (err[ip]*x[ip+1]/x[ip]**2)**2 #error propagation squared
        dxe[ip]  = np.sqrt(dxe2)/a
    return dx, dxe
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
@nb.jit(nopython = True)
def substract(x,err):
    '''
    substraction of constant terms in x^2 correlation function

    Parameters
    ----------
    x : array(float)
        
    err : array(float)
        

    Returns
    -------
    x_sub : array
        array x substracted.
    err_sub : array
        array err substracted.

    '''
    x_sub = np.copy(x)
    err_sub = np.copy(err)
    xs = x[x.size-1]
    errs  = err[x.size-1]
    for ip in range(x.size):
        x_sub[ip] = x[ip]-xs 
        err_sub[ip] = np.sqrt(err[ip]**2+errs**2)
    return x_sub, err_sub       

#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
@nb.jit(nopython = True)
def histogramarray(x, xmin, st, m, hist):
    '''
    evaluate in which bin of hist x must be counted

    Parameters
    ----------
    x : float/int
        value to be counted.
    xmin : float
        hist minimum.
    st : float
        bin width.
    m : int
        bins number.
    hist : array
        histograms to be filled.

    Returns
    -------
    None.

    '''
    j = (x - xmin)/st + 1.00000
    if (j < 1):
        j = 1
    if (j > m):
        j = m
    hist[int(j)-1] += 1
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
@nb.jit(nopython = True)
def instanton_distribution(z, nin, tmax, stzhist, nzhist, iz):
    for ii in range(0, nin, 2):
        if ii == 0:
            zm =  -tmax
        else:
            zm = z[ii-1]
        z0  = z[ii]
        zp  = z[ii+1]
        zia = min(zp-z0, z0-zm)    #chiedere info
        histogramarray( zia, 0.0, stzhist, nzhist, iz)
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
@nb.jit(nopython = True)
def building_wavefunction(hist, nxhist, stxhist, xhist_min):
    '''
    construction of the wave function probability density\
        from the histogram of position occurrences

    Parameters
    ----------
    hist : array
        array in which occurences are stored.
    nxhist : int
        number of hist bin.
    stxhist : float
        bin width.
    xhist_min : float
        minimum of hist.

    Returns
    -------
    position : array(float)
        discretized x axis.
    wave_function : array(float)
        probability density of ground state wave function.

    '''
    wave_function = np.zeros(nxhist)
    position = np.zeros(nxhist)
    xnorm = 0
    for i in range(nxhist):
        xnorm += hist[i]*stxhist #sommo il valore di ogni bin per la larghezza dei bin
    for i in range(nxhist):
        position[i] = xhist_min + (i+1)*stxhist #spazzo tutto lo spazio dei bin
        wave_function[i] = hist[i]/xnorm
    return position, wave_function
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
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
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
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
#---------------------------------------------------------------------------------
@nb.jit(nopython = True)
def find_instantons(x, dt):
    """Find the number of instantons and anti-instantons and save their
    positions.

    Parameters
    ----------
    x : ndarray
        Spatial configuration.
    dt : ndarray
        Euclidean time axis.

    Returns
    -------
    pos_roots : int
        Number of instantons.
    neg_roots : int
        Number of anti-instantons.
    a : array
        Instanton positions.
    b : array
        Anti-instanton positions.
    """
    pos_roots = 0
    neg_roots = 0
    pos_roots_position = np.array([0.0])
    neg_roots_position = np.array([0.0])
    # pos_roots_position = []
    # neg_roots_position = []

    if np.abs(x[0]) < 1e-7:
        x[0] = 0.0

    x_pos = x[0]

    for i in range(1, x.size - 1):
        if np.abs(x[i]) < 1e-7:
            x[i] = 0.0

            if x_pos > 0.:
                neg_roots += 1
                neg_roots_position = np.append(
                    neg_roots_position,
                    -x_pos * dt
                    / (x[i] - x_pos) + (i - 1) * dt
                )
            elif x_pos < 0.:
                pos_roots += 1
                pos_roots_position = np.append(
                    pos_roots_position,
                    -x_pos * dt
                    / (x[i] - x_pos) + dt * (i - 1)
                )
            else:
                continue

        elif x_pos * x[i] < 0.:

            if x[i] > x_pos:
                pos_roots += 1
                pos_roots_position = np.append(
                    pos_roots_position,
                    -x_pos * dt
                    / (x[i] - x_pos) + dt * (i - 1)
                )

            elif x[i] < x_pos:
                neg_roots += 1
                neg_roots_position = np.append(
                    neg_roots_position,
                    -x_pos * dt
                    / (x[i] - x_pos) + (i - 1) * dt
                )

        x_pos = x[i]

    if neg_roots == 0 or pos_roots == 0:
        return 0, 0, np.zeros(1), np.zeros(1)

    a = np.delete(pos_roots_position, 0)
    b = np.delete(neg_roots_position, 0)

    return pos_roots, neg_roots, \
           a, b
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#@nb.jit(nopython = True)
def streamline_equation(n_lattice_half, r_initial_sep,\
                        n_streamline, dtau, \
                            tau_store = 1.5, stream_time_step = 0.001,\
                                f = 1.4, a = 0.05):
    stream = open('Data/streamline/streamline.dat', 'w')
    conf = open('Data/streamline/conf.dat', 'w')
    action0 = open('Data/streamline/action.dat', 'w')
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
    s0 = 4/3 * f**3
    tau_centers_ia = np.array([n_lattice_half * dtau - r_initial_sep / 2.0,
                               n_lattice_half * dtau + r_initial_sep / 2.0])

    tau_array = np.linspace(0.0, n_lattice_half * 2 *
                            dtau, n_lattice_half * 2, False)
    xconf = np.zeros(n_lattice_half*2)
    xconf = new_config(xconf, n_lattice_half*2, len(tau_centers_ia), tau_centers_ia, f, a)
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
            action0.write('{:.4f}\t{:.4f}\n'.format(tau_array[i-2], action_density[i-2]))
        
        s = action(xconf,n_lattice_half*2,dtau,f)
        n_i, n_a, pos_root, neg_root = find_instantons(xconf[2:-2], dtau)
        
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
    action0.close()
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
    return None
#----------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#@nb.jit(nopython = True)
def zero_crossing(x, n=800, f=1.4, a = 0.05):
    ni = n//4
    s0 = 4/3 * f** 3
    tau_ia_zcr = np.array([], float)
    action_int_zcr =np.array([], float)
    for n_counter in range(ni, 2 * ni):
        z = np.array([ni*a, n_counter*a])
        x = new_config(x,n,2,z,f,a)
        n_inst, n_a_inst, pos_roots, neg_roots = find_instantons(x,a)
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

            
                tau_ia_zcr = np.append(tau_ia_zcr, z_ia)
                action_int_zcr = np.append(action_int_zcr, action(x,n,a,f))
    action_int_zcr /= s0
    action_int_zcr -= 2
    return tau_ia_zcr, action_int_zcr
#--------------------------------------------------------
#--------------------------------------------------------
#@nb.jit(nopython = True)
def cool_zero_crossing(nmc, ncool, n = 800, dtau = 0.05, f = 1.4, mode = 0,\
                       dx = 0.05, kp2 = 50):
    s0 = 4/3 * f**3
    tau_ia_zcr = np.array([], float)
    action_int_zcr =np.array([], float)
    n_inst_array = np.array([], float)
    counter = 0
    S_sum = 0
    N_sum = 0
    
    x = periodic_starting_conf(n, f, mode)
    for i in tqdm(range(nmc)):
        x = periodic_update(x,n,dtau,f, dx)  #metropolis algorithm implementation with periodic boundary conditions
        xs = np.copy(x)
    #cooling sweeps
        if i % kp2 == 0:
            for icool in range(ncool):
                xs = cooling_update(xs, n, dtau, f, dx)
            n_instantons, n_anti_instantons,\
                pos_roots, neg_roots = find_instantons(xs,dtau)
            s = action(xs, n, dtau, f)
            n_inst = n_instantons + n_anti_instantons
        # total zero crossings
            if n_instantons == n_anti_instantons \
                and n_instantons == 1 \
                and n_instantons == len(pos_roots) \
                and n_anti_instantons == len(neg_roots):

                    if pos_roots[0] < neg_roots[0]:
                        for i in range(n_instantons):
                            if i == 0:
                                zero_m = neg_roots[-1] - n * dtau
                            else:
                                zero_m = neg_roots[i - 1]

                            z_ia = np.minimum(np.abs(neg_roots[i] - pos_roots[i]),
                                      np.abs(pos_roots[i] - zero_m))
                            tau_ia_zcr = np.append(tau_ia_zcr, z_ia)
                            action_int_zcr = np.append(action_int_zcr, s)
                            n_inst_array = np.append(n_inst_array, n_inst)
                        
                            S_sum += s
                            N_sum += n_inst
                            counter += 1
                        
                    elif pos_roots[0] > neg_roots[0]:
                        for i in range(n_instantons):
                            if i == 0:
                                zero_p = pos_roots[-1] - n * dtau
                            else:
                                zero_p = pos_roots[i - 1]

                            z_ia = np.minimum(np.abs(pos_roots[i] - neg_roots[i]),
                                      np.abs(neg_roots[i] - zero_p))
                            tau_ia_zcr = np.append(tau_ia_zcr, z_ia)
                            action_int_zcr = np.append(action_int_zcr, s)
                            n_inst_array = np.append(n_inst_array, n_inst)
                        
                            S_sum += s
                            N_sum += n_inst
                            counter += 1
                    else:
                        continue
    permute = tau_ia_zcr.argsort()
    tau_ia_zcr = tau_ia_zcr[permute]
    n_inst_array = n_inst_array[permute]
    action_int_zcr = action_int_zcr[permute]
    action_int_zcr /= (s0*n_inst_array)
    action_int_zcr -= 2
    print('Action: '+'{:.4f}\n'.format(S_sum/counter)+ 'Instantons: '+'{:.4f}'.format(N_sum/counter))
    return tau_ia_zcr, action_int_zcr, n_inst_array
    

def cool_zero_crossing1(nmc, ncool, n = 800, dtau = 0.05, f = 1.4, mode = 0,\
                       dx = 0.05, kp2 = 50):
    s0 = 4/3 * f**3
    tau_ia_zcr = np.array([], float)
    action_int_zcr =np.array([], float)
    n_inst_array = np.array([], float)
    counter = 0
    S_sum = 0
    N_sum = 0
    
    xi= np.zeros(n)
    xa= np.zeros(n)
    z = np.zeros(n)
    x = periodic_starting_conf(n, f, mode)
    for i in tqdm(range(nmc)):
        x = periodic_update(x,n,dtau,f, dx)  #metropolis algorithm implementation with periodic boundary conditions
        xs = np.copy(x)
    #cooling sweeps
        if i % kp2 == 0:
            for icool in range(ncool):
                xs = cooling_update(xs, n, dtau, f, dx)
            n_instantons, n_anti_instantons = instantons(f, dtau, n, xs, xi, xa, z)
            s = action(xs, n, dtau, f)
            n_inst = n_instantons + n_anti_instantons
            if n_inst == 2:
                for ii in range(0, n_inst, 2):
                    if ii == 0:
                        zm =  -n * dtau
                    else:
                        zm = z[ii-1]
                    z0  = z[ii]
                    zp  = z[ii+1]
                    zia = min(zp-z0, z0-zm)
                    tau_ia_zcr = np.append(tau_ia_zcr, zia)
                    action_int_zcr = np.append(action_int_zcr, s)
                    n_inst_array = np.append(n_inst_array, n_inst)
                    counter +=1
                    S_sum += s
                    N_sum += n_inst
    snc = action(x,n, dtau,f)
    permute = tau_ia_zcr.argsort()
    tau_ia_zcr = tau_ia_zcr[permute]
    n_inst_array = n_inst_array[permute]
    action_int_zcr = action_int_zcr[permute]
    action_int_zcr /= s0
    action_int_zcr -= 2
    print(snc)
    print('Action: '+'{:.4f}\n'.format(S_sum/counter)+ 'Instantons: '+'{:.4f}'.format(N_sum/counter))
    return tau_ia_zcr, action_int_zcr, n_inst_array