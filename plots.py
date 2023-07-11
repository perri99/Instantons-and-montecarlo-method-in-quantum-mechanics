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
    if line.startswith('configuration: 355'):
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
    if line.startswith('configuration: 355'):
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
plt.title('Typical configuration')
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
plt.title('Probability distribution |\u03C8|\u00B2')
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
    lines = file.readlines()[2:20]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]
column3 = [float(line.split()[2]) for line in lines]

x = np.array(column1)
y = np.array(column2)
y_err = np.array(column3)
plt.errorbar(x, y, yerr=y_err, fmt='s', markerfacecolor='none',
             markeredgecolor = 'blue', markersize=8, capsize=5, label = '<x(0)x(τ)>')

with open('Data/qm/correlations2_qm.dat', 'r') as file:
    lines = file.readlines()[2:20]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]
column3 = [float(line.split()[2]) for line in lines]

x = np.array(column1)
y = np.array(column2)
y_err = np.array(column3)
plt.errorbar(x, y, yerr=y_err, fmt='s', markerfacecolor='none',
             markeredgecolor = 'red', markersize=8, capsize=5, label = '<x^2(0)x^2(τ)>')

with open('Data/qm/correlations3_qm.dat', 'r') as file:
    lines = file.readlines()[2:20]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]
column3 = [float(line.split()[2]) for line in lines]

x = np.array(column1)
y = np.array(column2)
y_err = np.array(column3)
plt.errorbar(x, y, yerr=y_err, fmt='s', markerfacecolor='none',
             markeredgecolor = 'green', markersize=8, capsize=5, label = '<x^3(0)x^3(τ)>')
plt.legend()
plt.title('Correlation functions')
plt.xlim(0, 1.5)
plt.xlabel('time')
plt.ylabel('<x^n(0)x^n(τ)>')
plt.savefig('Data/qm/correlations.pdf')
plt.savefig('Data/qm/correlations.png')
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
    lines = file.readlines()[2:20]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[3]) for line in lines]
column3 = [float(line.split()[4]) for line in lines]

x = np.array(column1)
y = np.array(column2)
y_err = np.array(column3)
plt.errorbar(x, y, yerr=y_err, fmt='s', markerfacecolor='none',
             markeredgecolor = 'blue', markersize=8, capsize=5, label = 'dlog<x(0)x(τ)>')

with open('Data/qm/correlations2_qm.dat', 'r') as file:
    lines = file.readlines()[2:20]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[3]) for line in lines]
column3 = [float(line.split()[4]) for line in lines]

x = np.array(column1)
y = np.array(column2)
y_err = np.array(column3)
plt.errorbar(x, y, yerr=y_err, fmt='s', markerfacecolor='none',
             markeredgecolor = 'red', markersize=8, capsize=5, label = 'dlog<x^2(0)x^2(τ)>')

with open('Data/qm/correlations3_qm.dat', 'r') as file:
    lines = file.readlines()[2:20]

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
plt.xlim(0, 1)
plt.title('Logarithmic derivatives of correlation functions')
plt.xlabel('time')
plt.ylabel('dlog<x^n(0)x^n(τ)>')
plt.savefig('Data/qm/dlog.pdf')
plt.savefig('Data/qm/dlog.png')
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
             markeredgecolor = 'blue', markersize=8, capsize=5, label = 'simulated')

plt.xlabel('T')
plt.ylabel('F')
plt.title('Free energy of anharmonic oscillator')
plt.xscale('log')
plt.xlim(0.01,2.5)
plt.ylim(-2.5, -1)
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
    lines = file.readlines()[2:20]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]
column3 = [float(line.split()[2]) for line in lines]

x = np.array(column1)
y = np.array(column2)
y_err = np.array(column3)
plt.errorbar(x, y, yerr=y_err, fmt='s', markerfacecolor='none',
             markeredgecolor = 'blue', markersize=8, capsize=5, label = '<x(0)x(τ)>')

with open('Data/qmcool/correlations2_cool.dat', 'r') as file:
    lines = file.readlines()[2:20]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]
column3 = [float(line.split()[2]) for line in lines]

x = np.array(column1)
y = np.array(column2)
y_err = np.array(column3)
plt.errorbar(x, y, yerr=y_err, fmt='s', markerfacecolor='none',
             markeredgecolor = 'red', markersize=8, capsize=5, label = '<x^2(0)x^2(τ)>')

with open('Data/qmcool/correlations3_cool.dat', 'r') as file:
    lines = file.readlines()[2:20]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]
column3 = [float(line.split()[2]) for line in lines]

x = np.array(column1)
y = np.array(column2)
y_err = np.array(column3)
plt.errorbar(x, y, yerr=y_err, fmt='s', markerfacecolor='none',
             markeredgecolor = 'green', markersize=8, capsize=5, label = '<x^3(0)x^3(τ)>')
plt.legend()
plt.title('Cooled correlation functions')
plt.xlim(0, 1)
plt.ylim(0,8)
plt.xlabel('time')
plt.ylabel('<x^n(0)x^n(τ)>')
plt.savefig('Data/qmcool/correlations.pdf')
plt.savefig('Data/qmcool/correlations.png')
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
    lines = file.readlines()[2:20]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[3]) for line in lines]
column3 = [float(line.split()[4]) for line in lines]

x = np.array(column1)
y = np.array(column2)
y_err = np.array(column3)
plt.errorbar(x, y, yerr=y_err, fmt='s', markerfacecolor='none',
             markeredgecolor = 'blue', markersize=8, capsize=5, label = 'dlog<x(0)x(τ)>')

with open('Data/qmcool/correlations2_cool.dat', 'r') as file:
    lines = file.readlines()[2:20]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[3]) for line in lines]
column3 = [float(line.split()[4]) for line in lines]

x = np.array(column1)
y = np.array(column2)
y_err = np.array(column3)
plt.errorbar(x, y, yerr=y_err, fmt='s', markerfacecolor='none',
             markeredgecolor = 'red', markersize=8, capsize=5, label = 'dlog<x^2(0)x^2(τ)>')

with open('Data/qmcool/correlations3_cool.dat', 'r') as file:
    lines = file.readlines()[2:20]

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
plt.xlim(0, 1)
plt.title('Logarithmic derivatives of cooled correlation functions')
plt.xlabel('time')
plt.ylabel('dlog<x^n(0)x^n(τ)>')
plt.savefig('Data/qmcool/dlog.pdf')
plt.savefig('Data/qmcool/dlog.png')
plt.show()

#------------------------------------------------------------------------------
#   FIG. 7: Fig. a. instanton density as a function of the number of cooling 
#   sweeps for different values of the parameter η
#------------------------------------------------------------------------------
with open('Data/qmcool/instdensity.dat', 'r') as file:
    lines = file.readlines()[2:]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]
column3 = [float(line.split()[3]) for line in lines]
column4 = [float(line.split()[4]) for line in lines]
x     = np.array(column1)
y     = np.array(column2)/(n*a)


plt.errorbar(x, y, fmt ='s', markerfacecolor = 'none',
             markeredgecolor = 'blue', markersize = 8, capsize = 5, label = 'f = 1.4')

y     = np.array(column3)/(n*a)
plt.plot(x, y, color = 'green', linewidth = 0.8, linestyle = '--', label = '2-loop')

y     = np.array(column4)/(n*a)
plt.plot(x, y, color = 'green', linewidth = 0.8, label = '1-loop')

plt.xlabel('n_cool')
plt.ylabel('N_top/\u03B2')
plt.title('Instanton density')

plt.xscale('log')
plt.yscale('log')
plt.xlim(1, )
plt.legend()
plt.show()

#------------------------------------------------------------------------------
#   FIG. 7: Fig. b. instanton action as a function of the number of cooling 
#   sweeps for different values of the parameter eta
#------------------------------------------------------------------------------
with open('Data/qmcool/inst_action.dat', 'r') as file:
    lines = file.readlines()[2:]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]
column3 = [float(line.split()[2]) for line in lines]
column4 = [float(line.split()[3]) for line in lines]
x     = np.array(column1)
y     = np.array(column2)
y_err = np.array(column3)

plt.errorbar(x, y, yerr=y_err, fmt='s',markerfacecolor='none',
             markeredgecolor = 'blue', markersize=8, capsize=5, label = 'η = 1.4')

y     = np.array(column4)

plt.plot(x, y, color='green', linewidth = 0.8, label= 'classical')

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
#   FIG. 9: Quantum mechanical paths which appear in a Monte-Carlo calculation
#   of the one-instanton partition function in the double well potential.
#------------------------------------------------------------------------------
for i in range(4):
    with open('Data/qmidens/idens_conf.dat', 'r') as file:
        lines = file.readlines()[i*n+1:(i+1)*n]

    column1 = [float(line.split()[0]) for line in lines]
    column2 = [float(line.split()[1]) for line in lines]

    x     = np.array(column1)
    y     = np.array(column2)
    if i==0:
        plt.plot(x, y, color = 'black', linewidth = 0.8)
    else:
        plt.plot(x, y, color = 'green', linewidth = 0.8, label = 'instanton fluctuations')
for i in range(4):
    with open('Data/qmidens/vac_conf.dat', 'r') as file:
        lines = file.readlines()[i*n+1:(i+1)*n]

    column1 = [float(line.split()[0]) for line in lines]
    column2 = [float(line.split()[1]) for line in lines]

    x     = np.array(column1)
    y     = np.array(column2)
    if i==0:
        plt.plot(x, y, color = 'black', linewidth = 0.8)
    else:
        plt.plot(x, y, color = 'blue', linewidth = 0.8, label = 'vacuum fluctuations')
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
plt.title('Correlation functions rilm')
plt.xlim(0, 1)
plt.xlabel('time')
plt.ylabel('<x^n(0)x^n(τ)>')
plt.savefig('Data/rilm/correlations.pdf')
plt.savefig('Data/rilm/correlations.png')
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
    lines = file.readlines()[2:20]

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
plt.xlim(0, 1)
plt.title('Logarithmic derivatives of correlation functions')
plt.xlabel('time')
plt.ylabel('dlog<x^n(0)x^n(τ)>')
plt.savefig('Data/rilm/dlog.pdf')
plt.savefig('Data/rilm/dlog.png')
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
plt.title('Correlation functions rilm')
plt.xlim(0, 1)
plt.xlabel('time')
plt.ylabel('<x^n(0)x^n(τ)>')
plt.savefig('Data/rilmgauss/correlations.pdf')
plt.savefig('Data/rilmgauss/correlations.png')
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
    lines = file.readlines()[2:20]

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
plt.xlim(0, 1)
plt.title('Logarithmic derivatives of correlation functions')
plt.xlabel('time')
plt.ylabel('dlog<x^n(0)x^n(τ)>')
plt.savefig('Data/rilmgauss/dlog.pdf')
plt.savefig('Data/rilmgauss/dlog.png')
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

plt.plot(x, y, color = 'red')

with open('Data/iilm/zdist.dat', 'r') as file:
    lines = file.readlines()[1:]

column1 = [float(line.split()[0]) for line in lines]
column2 = [float(line.split()[1]) for line in lines]

x     = np.array(column1)
y     = np.array(column2)

plt.plot(x, y, color = 'black', drawstyle = 'steps')

plt.xlabel('τ_z')
plt.ylabel('n_IA(τ_z)')
plt.title('Istanton-anti-istanton separation distribution')

plt.xlim(0,3.85)
plt.ylim(0,40000)
plt.show()