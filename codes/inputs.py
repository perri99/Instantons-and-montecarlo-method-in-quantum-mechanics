'''
Functions that read the inputs from the file 'inputs.txt'\
    and pass the values to the various programs
'''

def loading_inputs():
    '''
    loading inputs from external file

    '''
    file = open('inputs.txt', 'r')
    params = {}
    for line in file:
        line = line.strip() 
        if line and '=' in line:  
            key, value = line.split('=')  
            key = key.strip()  
            value = value.strip()  
            params[key] = float(value)  
    return params

def schrodinger():
    '''
    setting inputs for the schroedinger resolver

    '''
    params = loading_inputs()
    x_min = params['x_min']
    x_max = params['x_max']
    mass = params['mass']
    point_num = int(params['point_num'])
    minimum = params['f']
    return x_min, x_max, minimum, mass, point_num
    
    
    
def qm():
    '''
    setting inputs for qm.py

    '''
    params = loading_inputs()
    f = params['f']
    n = int(params['n']) #lattice points
    a = params['a'] #lattice spacing
    neq = int(params['neq']) #number of equilibration sweeps
    nmc = int(params['nmc']) #number of MonteCarlo sweeps
    dx = params['dx'] #width of updates
    n_p = int(params['n_p']) #number max of points in the correlation functions
    nc = int(params['nc']) #number of correlator measurements in a configuration
    kp = int(params['kp']) #number of sweeps between writeout of complete configuration 
    mode = params['mode'] # mode=0: cold start, x_i=-f; mode != 0: hot start, x_i=random
    seed = params['seed']
    return f, n, a, neq, nmc, dx, n_p, nc, kp, mode, seed
    
def qmswitch():
    '''
    setting inputs for qmswitch.py

    '''
    params = loading_inputs()
    f = params['f']
    n = int(params['n']) #lattice points
    a = params['a'] #lattice spacing
    neq = int(params['neq']) #number of equilibration sweeps
    nmc = int(params['nmc']) #number of MonteCarlo sweeps
    dx = params['dx'] #width of updates
    n_alpha = int(params['n_alpha']) #number max of points in the correlation functions
    nc = int(params['nc']) #number of correlator measurements in a configuration
    kp = int(params['kp']) #number of sweeps between writeout of complete configuration 
    mode = params['mode'] # mode=0: cold start, x_i=-f; mode != 0: hot start, x_i=random
    seed = params['seed']
    w0 = params['w0']
    return f, n, a, neq, nmc, dx, n_alpha, nc, kp, mode, seed, w0

def qmcool():
    '''
    setting inputs for qmcool.py

    '''
    params = loading_inputs()
    f = params['f']
    n = int(params['n']) #lattice points
    a = params['a'] #lattice spacing
    neq = int(params['neq']) #number of equilibration sweeps
    nmc = int(params['nmc']) #number of MonteCarlo sweeps
    ncool = int(params['ncool'])
    dx = params['dx'] #width of updates
    n_p = int(params['n_p']) #number max of points in the correlation functions
    nc = int(params['nc']) #number of correlator measurements in a configuration
    kp = int(params['kp']) #number of sweeps between writeout of complete configuration 
    kp2 = int(params['kp2'])
    mode = params['mode'] # mode=0: cold start, x_i=-f; mode != 0: hot start, x_i=random
    seed = params['seed']
    return f, n, a, neq, nmc, dx, n_p, nc, kp, mode, seed, kp2, ncool

def qmidens():
    '''
    setting inputs for qmidens.py

    '''
    params = loading_inputs()
    f = params['f']
    n = int(params['n']) #lattice points
    a = params['a'] #lattice spacing
    neq = int(params['neq']) #number of equilibration sweeps
    nmc = int(params['nmc']) #number of MonteCarlo sweeps
    dx = params['dx'] #width of updates
    n_alpha = int(params['n_alpha']) #number max of points in the correlation functions
    nc = int(params['nc']) #number of correlator measurements in a configuration
    kp = int(params['kp']) #number of sweeps between writeout of complete configuration 
    mode = params['mode'] # mode=0: cold start, x_i=-f; mode != 0: hot start, x_i=random
    seed = params['seed']
    w0 = params['w0']
    return f, n, a, neq, nmc, dx, n_alpha, nc, kp, mode, seed, w0

def rilm():
    '''
    setting inputs for rilm.py

    '''
    params = loading_inputs()
    f = params['f']
    n = int(params['n']) #lattice points
    a = params['a'] #lattice spacing
    neq = int(params['neq']) #number of equilibration sweeps
    nmc = int(params['nmc']) #number of MonteCarlo sweeps
    dx = params['dx'] #width of updates
    n_p = int(params['n_p']) #number max of points in the correlation functions
    nc = int(params['nc']) #number of correlator measurements in a configuration
    kp = int(params['kp']) #number of sweeps between writeout of complete configuration 
    N_inst = int(params['N_inst']) # mode=0: cold start, x_i=-f; mode != 0: hot start, x_i=random
    seed = params['seed']
    return f, n, a, N_inst, neq, nmc, dx, n_p, nc, kp, seed

def rilmgauss():
    '''
    setting inputs for rilmgauss.py

    '''
    params = loading_inputs()
    f = params['f']
    n = int(params['n']) #lattice points
    a = params['a'] #lattice spacing
    neq = int(params['neq']) #number of equilibration sweeps
    nmc = int(params['nmc']) #number of MonteCarlo sweeps
    nheat = int(params['nheat'])
    dx = params['dx'] #width of updates
    n_p = int(params['n_p']) #number max of points in the correlation functions
    nc = int(params['nc']) #number of correlator measurements in a configuration
    kp = int(params['kp']) #number of sweeps between writeout of complete configuration 
    N_inst = int(params['N_inst']) # mode=0: cold start, x_i=-f; mode != 0: hot start, x_i=random
    seed = params['seed']
    return f, n, a, neq, nmc, dx, n_p, nc, kp, N_inst,nheat, seed

def iilm():
    '''
    setting inputs for iilm.py

    '''
    params = loading_inputs()
    f = params['f']
    n = int(params['n']) #lattice points
    a = params['a'] #lattice spacing
    neq = int(params['neq']) #number of equilibration sweeps
    nmc = int(params['nmc']) #number of MonteCarlo sweeps
    dx = params['dx'] #width of updates
    n_p = int(params['n_p']) #number max of points in the correlation functions
    nc = int(params['nc']) #number of correlator measurements in a configuration
    kp = int(params['kp']) #number of sweeps between writeout of complete configuration 
    N_inst = int(params['N_inst']) 
    seed = params['seed']
    tcore = params['tcore']
    acore = params['acore']
    dz = params['dz']
    return f, n, a, neq, nmc, dx, n_p, nc, kp, N_inst, tcore, acore, dz, seed
