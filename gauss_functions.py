import numpy as np
import random
import numba as nb

@nb.jit(nopython = True)
def kinetic_local(x,y,a): 
    K = (1/(4*a))*(x-y)**2
    return K

@nb.jit(nopython = True)
def potential_gauss(w,x,y,a):
    return 0.5*w*a*(x-y)**2

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