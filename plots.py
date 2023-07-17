import numpy as np
import matplotlib.pyplot as plt
import re

n = 800
a = 0.05

#------------------------------------------------------------------------------
#   FIG. 2: Typical euclidean path obtained in a Monte Carlo simulation of the 
#   discretized euclidean action of the double well potential.
#------------------------------------------------------------------------------
''' Typical path vs cooled path '''
with open('Data/qmcool/config2.dat', 'r') as file:
    lines = file.readlines()

start_line = None
end_line   = None
for i, line in enumerate(lines):
    if line.startswith('configuration: 4450'):
        start_line = i
        break
data_lines = lines[start_line+1: start_line+801]

column1 = [float(line.split()[0]) for line in data_lines]
column2 = [float(line.split()[1]) for line in data_lines]
 
x     = np.array(column1)
y     = np.array(column2)
plt.plot(x, y, color = 'black',linewidth = 0.8, label = 'Monte Carlo')

with open('Data/qmcool/config_cool.dat', 'r') as file:
    lines = file.readlines()

start_line = None
end_line   = None
for i, line in enumerate(lines):
    if line.startswith('configuration: 4450'):
        start_line = i
        break
data_lines = lines[start_line+1: start_line+801]

column1 = [float(line.split()[0]) for line in data_lines]
column2 = [float(line.split()[1]) for line in data_lines]
 
x     = np.array(column1)
y     = np.array(column2)
plt.plot(x, y, color = 'red',linewidth = 1.7, label = 'cooling')
plt.legend()
plt.xlabel('time t')
plt.ylabel('x(t)')
plt.xlim(0,20)
#plt.title('Typical configuration')
plt.savefig('Data/qmcool/config.pdf')
plt.savefig('Data/qmcool/config.png')
plt.show()

#---------Fig3: Probability distribution-----------------------------------|
'''Exact groundstate vs Montecarlo groundstate'''

with open('Data/qmdiag/ground_state1_doublewell.dat', 'r') as file:
    lines = file.readlines()[1:1000]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]

x = np.array(column1)
y = np.array(column2)
plt.plot(x, y**2, color = 'red', label = 'exact')

with open('Data/qm/wavefunction_qm.dat', 'r') as file:
    lines = file.readlines()[1:50]  # read lines 1 to 51
    
column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]

x = np.array(column1)
y = np.array(column2)
plt.plot(x, y, color = 'blue', drawstyle = 'steps', label = 'Monte Carlo' )


plt.xlabel('x')
plt.ylabel('P(x)')
plt.legend()
#plt.title('Probability distribution |\u03C8|\u00B2')
plt.savefig('Data/qm/probability distribution.pdf')
plt.savefig('Data/qm/probability distribution.png')
plt.show()
#------------------------------------------------------------------------------
#   FIG. 4: Fig. a. Shows the correlation functions
#------------------------------------------------------------------------------
''' Exact correlation functions vs montecarlo '''

with open('Data/qmdiag/cor.dat', 'r') as file:
    lines = file.readlines()

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]


x = np.array(column1)
y = np.array(column2)
plt.plot(x, y, color = 'black')

with open('Data/qmdiag/cor2.dat', 'r') as file:
    lines = file.readlines()

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]


x = np.array(column1)
y = np.array(column2)
plt.plot(x, y, color = 'black')

with open('Data/qmdiag/cor3.dat', 'r') as file:
    lines = file.readlines()

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]


x = np.array(column1)
y = np.array(column2)
plt.plot(x, y, color = 'black')


with open('Data/qm/correlations_qm.dat', 'r') as file:
    lines = file.readlines()[2:]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]
column3 = [float(line.split()[2]) for line in lines]

x = np.array(column1)
y = np.array(column2)
y_err = np.array(column3)
plt.errorbar(x, y, yerr=y_err, fmt='s', markerfacecolor='none',
             markeredgecolor = 'blue', markersize=8, capsize=5, label = '<x(0)x(τ)>')

with open('Data/qm/correlations2_qm.dat', 'r') as file:
    lines = file.readlines()[2:]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]
column3 = [float(line.split()[2]) for line in lines]

x = np.array(column1)
y = np.array(column2)
y_err = np.array(column3)
plt.errorbar(x, y, yerr=y_err, fmt='s', markerfacecolor='none',
             markeredgecolor = 'red', markersize=8, capsize=5, label = '<x^2(0)x^2(τ)>')

with open('Data/qm/correlations3_qm.dat', 'r') as file:
    lines = file.readlines()[2:]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]
column3 = [float(line.split()[2]) for line in lines]

x = np.array(column1)
y = np.array(column2)
y_err = np.array(column3)
plt.errorbar(x, y, yerr=y_err, fmt='s', markerfacecolor='none',
             markeredgecolor = 'green', markersize=8, capsize=5, label = '<x^3(0)x^3(τ)>')
plt.legend()
#plt.title('Correlation functions')
plt.xlim(0, 1.5)
plt.xlabel('time')
plt.ylabel('<x^n(0)x^n(τ)>')
plt.savefig('Data/qm/correlations_qm.pdf')
plt.savefig('Data/qm/correlations_qm.png')
plt.show()
#------------------------------------------------------------------------------
#   FIG. 4: Fig. b. Shows the logarithmic derivative of the correlators
#------------------------------------------------------------------------------
''' Exact log derivatives vs Montecarlo simulation'''
with open('Data/qmdiag/dlog.dat', 'r') as file:
    lines = file.readlines()

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]
column3 = [float(line.split()[2]) for line in lines]
column4 = [float(line.split()[3]) for line in lines]

x = np.array(column1)
y = np.array(column2)
plt.plot(x, y, color = 'black')

y = np.array(column3)
plt.plot(x, y, color = 'black')

y = np.array(column4)
plt.plot(x, y, color = 'black')

with open('Data/qm/correlations_qm.dat', 'r') as file:
    lines = file.readlines()[2:]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[3]) for line in lines]
column3 = [float(line.split()[4]) for line in lines]

x = np.array(column1)
y = np.array(column2)
y_err = np.array(column3)
plt.errorbar(x, y, yerr=y_err, fmt='s', markerfacecolor='none',
             markeredgecolor = 'blue', markersize=8, capsize=5, label = 'dlog<x(0)x(τ)>')

with open('Data/qm/correlations2_qm.dat', 'r') as file:
    lines = file.readlines()[2:]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[3]) for line in lines]
column3 = [float(line.split()[4]) for line in lines]

x = np.array(column1)
y = np.array(column2)
y_err = np.array(column3)
plt.errorbar(x, y, yerr=y_err, fmt='s', markerfacecolor='none',
             markeredgecolor = 'red', markersize=8, capsize=5, label = 'dlog<x^2(0)x^2(τ)>')

with open('Data/qm/correlations3_qm.dat', 'r') as file:
    lines = file.readlines()[2:]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[3]) for line in lines]
column3 = [float(line.split()[4]) for line in lines]

x = np.array(column1)
y = np.array(column2)
y_err = np.array(column3)
plt.errorbar(x, y, yerr=y_err, fmt='s', markerfacecolor='none',
             markeredgecolor = 'green', markersize=8, capsize=5, label = 'dlog<x^2(0)x^2(τ)>')
plt.legend()
plt.ylim(0, 6)
plt.xlim(0, 1.5)
#plt.title('Logarithmic derivatives of correlation functions')
plt.xlabel('time')
plt.ylabel('dlog<x^n(0)x^n(τ)>')
plt.savefig('Data/qm/dlog_qm.pdf')
plt.savefig('Data/qm/dlog_qm.png')
plt.show()
#------------------------------------------------------------------------------
#   FIG. 5: Free energy F = −T log(Z) of the anharmonic oscillator as a 
#   function of the temperature t = i/b
#------------------------------------------------------------------------------
with open('Data/qmdiag/partition_function.dat', 'r') as file:
    lines = file.readlines()[1:]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[2]) for line in lines]

x     = np.array(column1)
y     = np.array(column2)

plt.plot(x, y, color = 'black', label = 'exact')

with open('Data/qmswitch/energies.dat', 'r') as file:
    lines = file.readlines()[1:]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]
column3 = [float(line.split()[2]) for line in lines]

x     = np.array(column1)
y     = np.array(column2)
y_err  = np.array(column3)

plt.errorbar(x, y, yerr=y_err, fmt='s', markerfacecolor='none',
             markeredgecolor = 'blue', markersize=8, capsize=5, label = 'adiabatic switching')

plt.xlabel('T')
plt.ylabel('F')
#plt.title('Free energy of anharmonic oscillator')
plt.xscale('log')
plt.xlim(0.01,2.5)
plt.ylim(-4, 0)
plt.legend()
plt.savefig('Data/qmswitch/free_energy.pdf')
plt.savefig('Data/qmswitch/free_energy.png')
plt.show()

#------------------------------------------------------------------------------
#   FIG. 6: Fig. a. Shows the cooled correlation functions
#------------------------------------------------------------------------------
''' Exact correlation functions vs cooled correlations'''

with open('Data/qmdiag/cor.dat', 'r') as file:
    lines = file.readlines()

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]


x = np.array(column1)
y = np.array(column2)
plt.plot(x, y, color = 'black')

with open('Data/qmdiag/cor2.dat', 'r') as file:
    lines = file.readlines()

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]


x = np.array(column1)
y = np.array(column2)
plt.plot(x, y, color = 'black')

with open('Data/qmdiag/cor3.dat', 'r') as file:
    lines = file.readlines()

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]


x = np.array(column1)
y = np.array(column2)
plt.plot(x, y, color = 'black')


with open('Data/qmcool/correlations_cool.dat', 'r') as file:
    lines = file.readlines()[2:]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]
column3 = [float(line.split()[2]) for line in lines]

x = np.array(column1)
y = np.array(column2)
y_err = np.array(column3)
plt.errorbar(x, y, yerr=y_err, fmt='s', markerfacecolor='none',
             markeredgecolor = 'blue', markersize=8, capsize=5, label = '<x(0)x(τ)>')

with open('Data/qmcool/correlations2_cool.dat', 'r') as file:
    lines = file.readlines()[2:]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]
column3 = [float(line.split()[2]) for line in lines]

x = np.array(column1)
y = np.array(column2)
y_err = np.array(column3)
plt.errorbar(x, y, yerr=y_err, fmt='s', markerfacecolor='none',
             markeredgecolor = 'red', markersize=8, capsize=5, label = '<x^2(0)x^2(τ)>')

with open('Data/qmcool/correlations3_cool.dat', 'r') as file:
    lines = file.readlines()[2:]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]
column3 = [float(line.split()[2]) for line in lines]

x = np.array(column1)
y = np.array(column2)
y_err = np.array(column3)
plt.errorbar(x, y, yerr=y_err, fmt='s', markerfacecolor='none',
             markeredgecolor = 'green', markersize=8, capsize=5, label = '<x^3(0)x^3(τ)>')
plt.legend()
#plt.title('Cooled correlation functions')
plt.xlim(0, 1.5)
plt.ylim(0,8)
plt.xlabel('time')
plt.ylabel('<x^n(0)x^n(τ)>')
plt.savefig('Data/qmcool/correlations_cool.pdf')
plt.savefig('Data/qmcool/correlations_cool.png')
plt.show()

#------------------------------------------------------------------------------
#   FIG. 6: Fig. b. Shows the logarithmic derivative of the cooled correlators
#------------------------------------------------------------------------------
''' Exact log derivatives vs cooled ones'''
with open('Data/qmdiag/dlog.dat', 'r') as file:
    lines = file.readlines()

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]
column3 = [float(line.split()[2]) for line in lines]
column4 = [float(line.split()[3]) for line in lines]

x = np.array(column1)
y = np.array(column2)
plt.plot(x, y, color = 'black')

y = np.array(column3)
plt.plot(x, y, color = 'black')

y = np.array(column4)
plt.plot(x, y, color = 'black')

with open('Data/qmcool/correlations_cool.dat', 'r') as file:
    lines = file.readlines()[2:]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[3]) for line in lines]
column3 = [float(line.split()[4]) for line in lines]

x = np.array(column1)
y = np.array(column2)
y_err = np.array(column3)
plt.errorbar(x, y, yerr=y_err, fmt='s', markerfacecolor='none',
             markeredgecolor = 'blue', markersize=8, capsize=5, label = 'dlog<x(0)x(τ)>')

with open('Data/qmcool/correlations2_cool.dat', 'r') as file:
    lines = file.readlines()[2:]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[3]) for line in lines]
column3 = [float(line.split()[4]) for line in lines]

x = np.array(column1)
y = np.array(column2)
y_err = np.array(column3)
plt.errorbar(x, y, yerr=y_err, fmt='s', markerfacecolor='none',
             markeredgecolor = 'red', markersize=8, capsize=5, label = 'dlog<x^2(0)x^2(τ)>')

with open('Data/qmcool/correlations3_cool.dat', 'r') as file:
    lines = file.readlines()[2:]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[3]) for line in lines]
column3 = [float(line.split()[4]) for line in lines]

x = np.array(column1)
y = np.array(column2)
y_err = np.array(column3)
plt.errorbar(x, y, yerr=y_err, fmt='s', markerfacecolor='none',
             markeredgecolor = 'green', markersize=8, capsize=5, label = 'dlog<x^2(0)x^2(τ)>')
plt.legend()
plt.ylim(-0.2, 6)
plt.xlim(0, 1.5)
plt.title('Logarithmic derivatives of cooled correlation functions')
plt.xlabel('time')
plt.ylabel('dlog<x^n(0)x^n(τ)>')
plt.savefig('Data/qmcool/dlog_cool.pdf')
plt.savefig('Data/qmcool/dlog_cool.png')
plt.show()

#------------------------------------------------------------------------------
#   FIG. 7: Fig. a. instanton density as a function of the number of cooling 
#   sweeps for different values of the parameter η
#------------------------------------------------------------------------------
with open('Data/qmcool/instdensity.dat', 'r') as file:
    lines = file.readlines()

start_line = None
end_line   = None
for i, line in enumerate(lines):
    if line.startswith('f = 1.4'):
        start_line = i
    elif line.startswith('f = 1.5'):
        end_line = i
        break
data_lines = lines[start_line+2: end_line-1]

column1 = [float(line.split()[0]) for line in data_lines]
column2 = [float(line.split()[1]) for line in data_lines]
column3 = [float(line.split()[3]) for line in data_lines]
column4 = [float(line.split()[4]) for line in data_lines]
x     = np.array(column1)
y     = np.array(column2)/(n*a)


plt.errorbar(x, y, fmt ='s', markerfacecolor = 'none',
             markeredgecolor = 'blue', markersize = 8, capsize = 5, label = '\u03b7 = 1.4')

y     = np.array(column3)/(n*a)
plt.plot(x, y, color = 'green', linewidth = 0.8, linestyle = '--', label = '2-loop')

y     = np.array(column4)/(n*a)
plt.plot(x, y, color = 'green', linewidth = 0.8, label = '1-loop')

#-------------------------------------------------------------------------------
for i, line in enumerate(lines):
    if line.startswith('f = 1.5'):
        start_line = i
    elif line.startswith('f = 1.6'):
        end_line = i
        break
data_lines = lines[start_line+2: end_line-1]

column1 = [float(line.split()[0]) for line in data_lines]
column2 = [float(line.split()[1]) for line in data_lines]
column3 = [float(line.split()[3]) for line in data_lines]
column4 = [float(line.split()[4]) for line in data_lines]
x     = np.array(column1)
y     = np.array(column2)/(n*a)


plt.errorbar(x, y, fmt ='o', markerfacecolor = 'none',
             markeredgecolor = 'blue', markersize = 8, capsize = 5, label = '\u03b7 = 1.5')

y     = np.array(column3)/(n*a)
plt.plot(x, y, color = 'green', linewidth = 0.8, linestyle = '--')

y     = np.array(column4)/(n*a)
plt.plot(x, y, color = 'green', linewidth = 0.8)
#--------------------------------------------------------------------------------
for i, line in enumerate(lines):
    if line.startswith('f = 1.6'):
        start_line = i
        break
data_lines = lines[start_line+2:]

column1 = [float(line.split()[0]) for line in data_lines]
column2 = [float(line.split()[1]) for line in data_lines]
column3 = [float(line.split()[3]) for line in data_lines]
column4 = [float(line.split()[4]) for line in data_lines]
x     = np.array(column1)
y     = np.array(column2)/(n*a)


plt.errorbar(x, y, fmt ='v', markerfacecolor = 'none',
             markeredgecolor = 'blue', markersize = 8, capsize = 5, label = '\u03b7 = 1.6')

y     = np.array(column3)/(n*a)
plt.plot(x, y, color = 'green', linewidth = 0.8, linestyle = '--')

y     = np.array(column4)/(n*a)
plt.plot(x, y, color = 'green', linewidth = 0.8)
plt.xlabel('n_cool')
plt.ylabel('N_top/\u03B2')
plt.title('Instanton density')
plt.xscale('log')
plt.yscale('log')
plt.xlim(1, )
plt.legend()
plt.savefig('Data/qmcool/instdensity.pdf')
plt.savefig('Data/qmcool/instdensity.png')
plt.show()

#------------------------------------------------------------------------------
#   FIG. 7: Fig. b. instanton action as a function of the number of cooling 
#   sweeps for different values of the parameter eta
#------------------------------------------------------------------------------
with open('Data/qmcool/inst_action.dat', 'r') as file:
    lines = file.readlines()

start_line = None
end_line   = None
for i, line in enumerate(lines):
    if line.startswith('f = 1.4'):
        start_line = i
    elif line.startswith('f = 1.5'):
        end_line = i
        break
data_lines = lines[start_line+2: end_line-1]

column1 = [float(line.split()[0]) for line in data_lines]
column2 = [float(line.split()[1]) for line in data_lines]
column3 = [float(line.split()[2]) for line in data_lines]
column4 = [float(line.split()[3]) for line in data_lines]
x     = np.array(column1)
y     = np.array(column2)
y_err = np.array(column3)

plt.errorbar(x, y, yerr=y_err, fmt='s',markerfacecolor='none',
             markeredgecolor = 'blue', markersize=8, capsize=5, label = '\u03b7 = 1.4')

y     = np.array(column4)

plt.plot(x, y, color='green', linewidth = 0.8, label= 'classical')
#---------------------------------------------------------------------------------
for i, line in enumerate(lines):
    if line.startswith('f = 1.5'):
        start_line = i
    elif line.startswith('f = 1.6'):
        end_line = i
        break
data_lines = lines[start_line+2: end_line-1]

column1 = [float(line.split()[0]) for line in data_lines]
column2 = [float(line.split()[1]) for line in data_lines]
column3 = [float(line.split()[2]) for line in data_lines]
column4 = [float(line.split()[3]) for line in data_lines]
x     = np.array(column1)
y     = np.array(column2)
y_err = np.array(column3)

plt.errorbar(x, y, yerr=y_err, fmt='o',markerfacecolor='none',
             markeredgecolor = 'blue', markersize=8, capsize=5, label = '\u03b7 = 1.5')

y     = np.array(column4)

plt.plot(x, y, color='green', linewidth = 0.8)
#----------------------------------------------------------------------------------
for i, line in enumerate(lines):
    if line.startswith('f = 1.6'):
        start_line = i
        break
data_lines = lines[start_line+2:]
column1 = [float(line.split()[0]) for line in data_lines]
column2 = [float(line.split()[1]) for line in data_lines]
column3 = [float(line.split()[2]) for line in data_lines]
column4 = [float(line.split()[3]) for line in data_lines]
x     = np.array(column1)
y     = np.array(column2)
y_err = np.array(column3)

plt.errorbar(x, y, yerr=y_err, fmt='v',markerfacecolor='none',
             markeredgecolor = 'blue', markersize=8, capsize=5, label = '\u03b7 = 1.6')

y     = np.array(column4)

plt.plot(x, y, color='green', linewidth = 0.8)
plt.xlabel('n_cool')
plt.ylabel('S/N_inst')
plt.title('Action per instanton')

plt.xscale('log')
plt.yscale('log')
plt.xlim(1, )
plt.legend()
plt.savefig('Data/qmcool/inst_action.pdf')
plt.savefig('Data/qmcool/inst_action.png')
plt.show()
#------------------------------------------------------------------------------
#   FIG. 8: Instanton density as a function of the parameter f.
#------------------------------------------------------------------------------
def l1(x):
    return 8*x**(5/2)*np.sqrt(2/np.pi)*np.exp(-4/3*x**3)
def l2(x):
    return 8*x**(5/2)*np.sqrt(2/np.pi)*np.exp(-4/3*x**3-71/72*1/(4/3*x**3))
def dE(x):
    return np.sqrt((6*(4/3)*x**3)/np.pi)*4*x*np.exp(-4/3*x**3)

x = np.linspace(0.02, 2, 100)
y = l1(x)

plt.plot(x, y, color = 'green', linewidth = 0.8, linestyle = '--', label = '1-loop')

y = l2(x)
plt.plot(x, y, color = 'green', linewidth = 0.8, label = '2-loop')

with open('Data/qmidens/density.dat') as file:
    lines = file.readlines()[1:]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]
column3 = [float(line.split()[2]) for line in lines]

x     = np.array(column1)
y     = np.array(column2)
y_err  = np.array(column3)

plt.errorbar(x, y, yerr = y_err, fmt='v',markerfacecolor='none',
             markeredgecolor = 'red', markersize=8, capsize=5, label = 'qmidens')
#--------------------------------    
with open('Data/qmdiag/splitting.dat') as file:
    lines = file.readlines()[1:]
column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]

x     = np.array(column1)
y     = np.array(column2)

plt.plot(x, y/2, color = 'black', linewidth = 0.8, label = '\u0394E/2')
#----------------------------------------
with open('Data/qmcool/instdensity_varf.dat') as file:
    lines = file.readlines()
column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]
column3 = [float(line.split()[2]) for line in lines]

x     = np.array(column1)
y     = np.array(column2)
y_err  = np.array(column3)

plt.errorbar(x, y/40, yerr = y_err/40, fmt='s',markerfacecolor='none',
             markeredgecolor = 'blue', markersize=8, capsize=5, label = 'qmcool')
plt.yscale('log')
plt.ylim(0.01, 3.1)
plt.xlim(0, 1.85)
plt.legend()
plt.xlabel('\u03b7')
plt.ylabel('N_top/\u03B2')
plt.savefig('Data/fig8.pdf')
plt.savefig('Data/fig8.png')

plt.show()
#------------------------------------------------------------------------------
#   FIG. 9: Quantum mechanical paths which appear in a Monte-Carlo calculation
#   of the one-instanton partition function in the double well potential.
#------------------------------------------------------------------------------
for i in range(3):
    with open('Data/qmidens/idens_conf.dat', 'r') as file:
        lines = file.readlines()[i*n+1:(i+1)*n]

    column1 = [float(line.split()[0]) for line in lines]
    column2 = [float(line.split()[1]) for line in lines]

    x     = np.array(column1)
    y     = np.array(column2)
    if i==0:
        plt.plot(x, y, color = 'black', linewidth = 2.5,label = 'instanton')
    else:
        plt.plot(x, y, color = 'green', linewidth = 0.8)
for i in range(3):
    with open('Data/qmidens/vac_conf.dat', 'r') as file:
        lines = file.readlines()[i*n+1:(i+1)*n]

    column1 = [float(line.split()[0]) for line in lines]
    column2 = [float(line.split()[1]) for line in lines]

    x     = np.array(column1)
    y     = np.array(column2)
    if i==0:
        plt.plot(x, y, color = 'red', linewidth = 2.5, label = 'vacuum')
    else:
        plt.plot(x, y, color = 'blue', linewidth = 0.8)
plt.legend()
plt.xlabel('τ')
plt.ylabel('X')
plt.title('Quantum mechanical path of one-instanton partition function')
plt.xlim(0, n*a-a)
plt.savefig('Data/qmidens/path.pdf')
plt.savefig('Data/qmidens/path.png')
plt.show()

#   FIG. 10: Fig. a. Shows the rilm correlation functions
#------------------------------------------------------------------------------
''' Exact correlation functions vs montecarlo '''

with open('Data/qmdiag/cor.dat', 'r') as file:
    lines = file.readlines()

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]


x = np.array(column1)
y = np.array(column2)
plt.plot(x, y, color = 'black')

with open('Data/qmdiag/cor2.dat', 'r') as file:
    lines = file.readlines()

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]


x = np.array(column1)
y = np.array(column2)
plt.plot(x, y, color = 'black')

with open('Data/qmdiag/cor3.dat', 'r') as file:
    lines = file.readlines()

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]


x = np.array(column1)
y = np.array(column2)
plt.plot(x, y, color = 'black')


with open('Data/rilm/correlations_rilm.dat', 'r') as file:
    lines = file.readlines()[3:]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]
column3 = [float(line.split()[2]) for line in lines]

x = np.array(column1)
y = np.array(column2)
y_err = np.array(column3)
plt.errorbar(x, y, yerr=y_err, fmt='s', markerfacecolor='none',
             markeredgecolor = 'blue', markersize=8, capsize=5, label = '<x(0)x(τ)>')

with open('Data/rilm/correlations2_rilm.dat', 'r') as file:
    lines = file.readlines()[1:]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]
column3 = [float(line.split()[2]) for line in lines]

x = np.array(column1)
y = np.array(column2)
y_err = np.array(column3)
plt.errorbar(x, y, yerr=y_err, fmt='s', markerfacecolor='none',
             markeredgecolor = 'red', markersize=8, capsize=5, label = '<x^2(0)x^2(τ)>')

with open('Data/rilm/correlations3_rilm.dat', 'r') as file:
    lines = file.readlines()[2:]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]
column3 = [float(line.split()[2]) for line in lines]

x = np.array(column1)
y = np.array(column2)
y_err = np.array(column3)
plt.errorbar(x, y, yerr=y_err, fmt='s', markerfacecolor='none',
             markeredgecolor = 'green', markersize=8, capsize=5, label = '<x^3(0)x^3(τ)>')
plt.legend()
#plt.title('Correlation functions rilm')
plt.xlim(0, 1.5)
plt.xlabel('time')
plt.ylabel('<x^n(0)x^n(τ)>')
plt.savefig('Data/rilm/correlations_rilm.pdf')
plt.savefig('Data/rilm/correlations_rilm.png')
plt.show()
#------------------------------------------------------------------------------
#   FIG. 10.b Shows the logarithmic derivative of the correlators
#------------------------------------------------------------------------------
''' Exact log derivatives vs Montecarlo simulation'''
with open('Data/qmdiag/dlog.dat', 'r') as file:
    lines = file.readlines()

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]
column3 = [float(line.split()[2]) for line in lines]
column4 = [float(line.split()[3]) for line in lines]

x = np.array(column1)
y = np.array(column2)
plt.plot(x, y, color = 'black')

y = np.array(column3)
plt.plot(x, y, color = 'black')

y = np.array(column4)
plt.plot(x, y, color = 'black')

with open('Data/rilm/correlations_rilm.dat', 'r') as file:
    lines = file.readlines()[3:]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[3]) for line in lines]
column3 = [float(line.split()[4]) for line in lines]

x = np.array(column1)
y = np.array(column2)
y_err = np.array(column3)
plt.errorbar(x, y, yerr=y_err, fmt='s', markerfacecolor='none',
             markeredgecolor = 'blue', markersize=8, capsize=5, label = 'dlog<x(0)x(τ)>')

with open('Data/rilm/correlations2_rilm.dat', 'r') as file:
    lines = file.readlines()[2:]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[3]) for line in lines]
column3 = [float(line.split()[4]) for line in lines]

x = np.array(column1)
y = np.array(column2)
y_err = np.array(column3)
plt.errorbar(x, y, yerr=y_err, fmt='s', markerfacecolor='none',
             markeredgecolor = 'red', markersize=8, capsize=5, label = 'dlog<x^2(0)x^2(τ)>')

with open('Data/rilm/correlations3_rilm.dat', 'r') as file:
    lines = file.readlines()[2:]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[3]) for line in lines]
column3 = [float(line.split()[4]) for line in lines]

x = np.array(column1)
y = np.array(column2)
y_err = np.array(column3)
plt.errorbar(x, y, yerr=y_err, fmt='s', markerfacecolor='none',
             markeredgecolor = 'green', markersize=8, capsize=5, label = 'dlog<x^2(0)x^2(τ)>')
plt.legend()
plt.ylim(0, 6)
plt.xlim(0, 1.5)
#plt.title('Logarithmic derivatives of correlation functions')
plt.xlabel('time')
plt.ylabel('dlog<x^n(0)x^n(τ)>')
plt.savefig('Data/rilm/dlog_rilm.pdf')
plt.savefig('Data/rilm/dlog_rilm.png')
plt.show()
#------------------------------------------------------------------------------
#   FIG. 12: Typical euclidean path obtained in a Monte Carlo simulation of the 
#   discretized euclidean action of the double well potential for  = 1.4.
#------------------------------------------------------------------------------

with open('Data/rilmgauss/config2_rilmgauss.dat', 'r') as file:
    lines = file.readlines()

start_line = None
end_line   = None
for i, line in enumerate(lines):
    if line.startswith('config: 701'):
        start_line = i
    elif line.startswith('config: 751'):
        end_line = i
        break
data_lines = lines[start_line+1: end_line-1]

column1 = [float(line.split()[0]) for line in data_lines]
column2 = [float(line.split()[1]) for line in data_lines]
 
x     = np.array(column1)
y     = np.array(column2)


plt.plot(x, y, color = 'black',linewidth = 0.8, label = 'RILM')

with open('Data/rilmgauss/config_hot_rilm.dat', 'r') as file:
    lines = file.readlines()

start_line = None
end_line   = None
for i, line in enumerate(lines):
    if line.startswith('config: 701'):
        start_line = i
    elif line.startswith('config: 751'):
        end_line = i
        break
data_lines = lines[start_line+1: end_line-1]

column2 = [float(line.split()[1]) for line in data_lines]
 
y     = np.array(column2)


plt.plot(x, y, color = 'green',linewidth = 0.8, label = 'Gaussian fl')


plt.xlabel('τ')
plt.ylabel('x')
plt.legend()
plt.title('Random instanton configuration vs Gaussian fluctuations')

plt.xlim(0, 20)
plt.savefig('Data/rilmgauss/config.pdf')
plt.savefig('Data/rilmgauss/config.jpeg')
plt.show()
#------------------------------------------------------------------------------
#   FIG. 13.a Shows the correlation functions in a random instanton ensamble
#   with gaussian fluctuations
#------------------------------------------------------------------------------
''' Exact correlation functions vs montecarlo '''

with open('Data/qmdiag/cor.dat', 'r') as file:
    lines = file.readlines()

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]


x = np.array(column1)
y = np.array(column2)
plt.plot(x, y, color = 'black')

with open('Data/qmdiag/cor2.dat', 'r') as file:
    lines = file.readlines()

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]


x = np.array(column1)
y = np.array(column2)
plt.plot(x, y, color = 'black')

with open('Data/qmdiag/cor3.dat', 'r') as file:
    lines = file.readlines()

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]


x = np.array(column1)
y = np.array(column2)
plt.plot(x, y, color = 'black')


with open('Data/rilmgauss/correlations_rilmgauss.dat', 'r') as file:
    lines = file.readlines()[3:]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]
column3 = [float(line.split()[2]) for line in lines]

x = np.array(column1)
y = np.array(column2)
y_err = np.array(column3)
plt.errorbar(x, y, yerr=y_err, fmt='s', markerfacecolor='none',
             markeredgecolor = 'blue', markersize=8, capsize=5, label = '<x(0)x(τ)>')

with open('Data/rilmgauss/correlations2_rilmgauss.dat', 'r') as file:
    lines = file.readlines()[1:]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]
column3 = [float(line.split()[2]) for line in lines]

x = np.array(column1)
y = np.array(column2)
y_err = np.array(column3)
plt.errorbar(x, y, yerr=y_err, fmt='s', markerfacecolor='none',
             markeredgecolor = 'red', markersize=8, capsize=5, label = '<x^2(0)x^2(τ)>')

with open('Data/rilmgauss/correlations3_rilmgauss.dat', 'r') as file:
    lines = file.readlines()[2:]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]
column3 = [float(line.split()[2]) for line in lines]

x = np.array(column1)
y = np.array(column2)
y_err = np.array(column3)
plt.errorbar(x, y, yerr=y_err, fmt='s', markerfacecolor='none',
             markeredgecolor = 'green', markersize=8, capsize=5, label = '<x^3(0)x^3(τ)>')
plt.legend()
#plt.title('Correlation functions rilm')
plt.xlim(0, 1.5)
plt.xlabel('time')
plt.ylabel('<x^n(0)x^n(τ)>')
plt.savefig('Data/rilmgauss/correlations_rilmgauss.pdf')
plt.savefig('Data/rilmgauss/correlations_rilmgauss.png')
plt.show()
#------------------------------------------------------------------------------
#   FIG. 13.b Shows the logarithmic derivative of the correlators
#------------------------------------------------------------------------------
''' Exact log derivatives vs Montecarlo simulation'''
with open('Data/qmdiag/dlog.dat', 'r') as file:
    lines = file.readlines()

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]
column3 = [float(line.split()[2]) for line in lines]
column4 = [float(line.split()[3]) for line in lines]

x = np.array(column1)
y = np.array(column2)
plt.plot(x, y, color = 'black')

y = np.array(column3)
plt.plot(x, y, color = 'black')

y = np.array(column4)
plt.plot(x, y, color = 'black')

with open('Data/rilmgauss/correlations_rilmgauss.dat', 'r') as file:
    lines = file.readlines()[3:]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[3]) for line in lines]
column3 = [float(line.split()[4]) for line in lines]

x = np.array(column1)
y = np.array(column2)
y_err = np.array(column3)
plt.errorbar(x, y, yerr=y_err, fmt='s', markerfacecolor='none',
             markeredgecolor = 'blue', markersize=8, capsize=5, label = 'dlog<x(0)x(τ)>')

with open('Data/rilmgauss/correlations2_rilmgauss.dat', 'r') as file:
    lines = file.readlines()[2:]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[3]) for line in lines]
column3 = [float(line.split()[4]) for line in lines]

x = np.array(column1)
y = np.array(column2)
y_err = np.array(column3)
plt.errorbar(x, y, yerr=y_err, fmt='s', markerfacecolor='none',
             markeredgecolor = 'red', markersize=8, capsize=5, label = 'dlog<x^2(0)x^2(τ)>')

with open('Data/rilmgauss/correlations3_rilmgauss.dat', 'r') as file:
    lines = file.readlines()[2:]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[3]) for line in lines]
column3 = [float(line.split()[4]) for line in lines]

x = np.array(column1)
y = np.array(column2)
y_err = np.array(column3)
plt.errorbar(x, y, yerr=y_err, fmt='s', markerfacecolor='none',
             markeredgecolor = 'green', markersize=8, capsize=5, label = 'dlog<x^2(0)x^2(τ)>')
plt.legend()
plt.ylim(0, 6)
plt.xlim(0, 1.5)
#plt.title('Logarithmic derivatives of correlation functions')
plt.xlabel('time')
plt.ylabel('dlog<x^n(0)x^n(τ)>')
plt.savefig('Data/rilmgauss/dlog_gauss.pdf')
plt.savefig('Data/rilmgauss/dlog_gauss.png')
plt.show()
#------------------------------------------------------------------------------
#   Fig.16 Distribution of instanton-anti-instanton separations
#------------------------------------------------------------------------------
with open('Data/rilm/hist_z_rilm.dat', 'r') as file:
    lines = file.readlines()[1:]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]

x     = np.array(column1)
y     = np.array(column2)

plt.plot(x, y, color = 'red', label= 'random')

with open('Data/iilm/zdist.dat', 'r') as file:
    lines = file.readlines()[1:]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]

x     = np.array(column1)
y     = np.array(column2)

plt.plot(x, y, color = 'black', drawstyle = 'steps', label = 'interactive')

plt.xlabel('τ_z')
plt.ylabel('n_IA(τ_z)')
plt.title('Istanton-anti-istanton separation distribution')
plt.xlim(0,3.85)
plt.ylim(0,45000)
plt.legend()
plt.savefig('Data/iilm/Istanton-anti-istanton separation distribution.pdf')
plt.savefig('Data/iilm/Istanton-anti-istanton separation distribution.png')
plt.show()
#-----------------------------------------------------------------------------
#   Fig.17 Typical instanton configuration in an instanton calculation
#------------------------------------------------------------------------------
with open('Data/iilm/iconf_iilm.dat', 'r') as file:
    lines = file.readlines()[:3001]

for i in range(10):
    column  = [float(line.split()[i]) for line in lines]
    
    y = np.array(column)
    x = range(len(y))
    if i % 2 == 0:
        plt.plot(x, y, color = 'blue', linewidth = 0.3)
    else:
        plt.plot(x, y, color = 'red', linewidth = 0.3)

plt.xlabel('configurations')
plt.ylabel('x')
plt.title('Instanton configuration in an interacting instanton calculation')

plt.xlim(0, 3000)
plt.savefig('Data/iilm/inst_position.pdf')
plt.savefig('Data/iilm/inst_position.png')
plt.show()
#------------------------------------------------------------------------------
#   FIG. new.a Shows the correlation functions in a random instanton ensamble
#   with interactions
#------------------------------------------------------------------------------
''' Exact correlation functions vs montecarlo '''

with open('Data/qmdiag/cor.dat', 'r') as file:
    lines = file.readlines()

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]


x = np.array(column1)
y = np.array(column2)
plt.plot(x, y, color = 'black')

with open('Data/qmdiag/cor2.dat', 'r') as file:
    lines = file.readlines()

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]


x = np.array(column1)
y = np.array(column2)
plt.plot(x, y, color = 'black')

with open('Data/qmdiag/cor3.dat', 'r') as file:
    lines = file.readlines()

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]


x = np.array(column1)
y = np.array(column2)
plt.plot(x, y, color = 'black')


with open('Data/iilm/icor_iilm.dat', 'r') as file:
    lines = file.readlines()[3:]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]
column3 = [float(line.split()[2]) for line in lines]

x = np.array(column1)
y = np.array(column2)
y_err = np.array(column3)
plt.errorbar(x, y, yerr=y_err, fmt='s', markerfacecolor='none',
             markeredgecolor = 'blue', markersize=8, capsize=5, label = '<x(0)x(τ)>')

with open('Data/iilm/icor2_iilm.dat', 'r') as file:
    lines = file.readlines()[1:]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]
column3 = [float(line.split()[2]) for line in lines]

x = np.array(column1)
y = np.array(column2)
y_err = np.array(column3)
plt.errorbar(x, y, yerr=y_err, fmt='s', markerfacecolor='none',
             markeredgecolor = 'red', markersize=8, capsize=5, label = '<x^2(0)x^2(τ)>')

with open('Data/iilm/icor3_iilm.dat', 'r') as file:
    lines = file.readlines()[2:]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]
column3 = [float(line.split()[2]) for line in lines]

x = np.array(column1)
y = np.array(column2)
y_err = np.array(column3)
plt.errorbar(x, y, yerr=y_err, fmt='s', markerfacecolor='none',
             markeredgecolor = 'green', markersize=8, capsize=5, label = '<x^3(0)x^3(τ)>')
plt.legend()
#plt.title('Correlation functions rilm')
plt.xlim(0, 1.5)
plt.xlabel('time')
plt.ylabel('<x^n(0)x^n(τ)>')
plt.savefig('Data/iilm/correlations_iilm.pdf')
plt.savefig('Data/iilm/correlations_iilm.png')
plt.show()
#------------------------------------------------------------------------------
#   FIG. 13.b Shows the logarithmic derivative of the correlators
#------------------------------------------------------------------------------
''' Exact log derivatives vs Montecarlo simulation'''
with open('Data/qmdiag/dlog.dat', 'r') as file:
    lines = file.readlines()

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]
column3 = [float(line.split()[2]) for line in lines]
column4 = [float(line.split()[3]) for line in lines]

x = np.array(column1)
y = np.array(column2)
plt.plot(x, y, color = 'black')

y = np.array(column3)
plt.plot(x, y, color = 'black')

y = np.array(column4)
plt.plot(x, y, color = 'black')

with open('Data/iilm/icor_iilm.dat', 'r') as file:
    lines = file.readlines()[3:]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[3]) for line in lines]
column3 = [float(line.split()[4]) for line in lines]

x = np.array(column1)
y = np.array(column2)
y_err = np.array(column3)
plt.errorbar(x, y, yerr=y_err, fmt='s', markerfacecolor='none',
             markeredgecolor = 'blue', markersize=8, capsize=5, label = 'dlog<x(0)x(τ)>')

with open('Data/iilm/icor2_iilm.dat', 'r') as file:
    lines = file.readlines()[2:]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[3]) for line in lines]
column3 = [float(line.split()[4]) for line in lines]

x = np.array(column1)
y = np.array(column2)
y_err = np.array(column3)
plt.errorbar(x, y, yerr=y_err, fmt='s', markerfacecolor='none',
             markeredgecolor = 'red', markersize=8, capsize=5, label = 'dlog<x^2(0)x^2(τ)>')

with open('Data/iilm/icor3_iilm.dat', 'r') as file:
    lines = file.readlines()[2:]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[3]) for line in lines]
column3 = [float(line.split()[4]) for line in lines]

x = np.array(column1)
y = np.array(column2)
y_err = np.array(column3)
plt.errorbar(x, y, yerr=y_err, fmt='s', markerfacecolor='none',
             markeredgecolor = 'green', markersize=8, capsize=5, label = 'dlog<x^2(0)x^2(τ)>')
plt.legend()
plt.ylim(-0.2, 6)
plt.xlim(0, 1.5)
#plt.title('Logarithmic derivatives of correlation functions')
plt.xlabel('time')
plt.ylabel('dlog<x^n(0)x^n(τ)>')
plt.savefig('Data/iilm/dlog_iilm.pdf')
plt.savefig('Data/iilm/dlog_iilm.png')
plt.show()