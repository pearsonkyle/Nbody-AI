import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import copy
import pickle
import os
import sys
import pdb
import glob
import gzip
import pymultinest
import math
import os, threading, subprocess
import matplotlib.pyplot as plt
if not os.path.exists("chains"): os.mkdir("chains")
import time


'''---------------------------------------------------
Script to try out PyMultiNest 
---------------------------------------------------'''

def model_quad(params,x):
    '''Function to model data. Gotta have one of these!
    This one is for a quadratic model.'''
    fakedata = params[0] + params[1]*x + params[2]*x**2
    return fakedata

def model_lin(params,x):
	'''Linear model'''
	return params[0]+params[1]*x

def myprior_quad(cube, ndim, n_params,paramlimits=[-10.,10.,-10.,10.,-10,10]):
	'''This transforms a unit cube into the dimensions of your prior
	space to search. Make sure you do this right!'''
	cube[0] = (paramlimits[1] - paramlimits[0])*cube[0]+paramlimits[0]
	cube[1] = (paramlimits[3] - paramlimits[2])*cube[1]+paramlimits[2]
	cube[2] = (paramlimits[5] - paramlimits[4])*cube[2]+paramlimits[4]

def myprior_lin(cube, ndim, n_params,paramlimits=[-10.,10.,-10.,10.]):
	'''This transforms a unit cube into the dimensions of your prior
	space to search. Make sure you do this right!'''
	cube[0] = (paramlimits[1] - paramlimits[0])*cube[0]+paramlimits[0]
	cube[1] = (paramlimits[3] - paramlimits[2])*cube[1]+paramlimits[2]

'''----------------------------------------------------
First, fake some data - comment out the line you
don't want
----------------------------------------------------'''

error = 0.1

xx = np.linspace(0,1,100) 

data = model_quad([5,2,4],xx) + error*np.random.randn(100) # quad test
# data = model_lin([7,3],xx) + error*np.random.randn(100) # linear test

def myloglike_quad(cube, ndim, n_params):
	'''The most important function. What is your likelihood function?
	I have chosen a simple chi2 gaussian errors likelihood here.'''
	loglike = -np.sum((data-model_quad(cube,xx))**2)/error**2
	return loglike/2.

def myloglike_lin(cube, ndim, n_params):
	'''The most important function. What is your likelihood function?
	I have chosen a simple chi2 gaussian errors likelihood here.'''
	loglike = -np.sum((data-model_lin(cube,xx))**2)/error**2
	return loglike/2.

'''----------------------------------------------------
Now we set up the multinest routine 
----------------------------------------------------'''
# number of dimensions our problem has
ndim = 3
n_params = ndim #oddly, this needs to be specified

pymultinest.run(myloglike_quad, myprior_quad, n_params, 
	resume = False, verbose = True, sampling_efficiency = 0.3)

# lets analyse the results
a = pymultinest.Analyzer(n_params = n_params) #retrieves the data that has been written to hard drive
s = a.get_stats()
values = s['marginals'] # gets the marginalized posterior probability distributions 
evidence_quad = s['global evidence']

print()
print( "-" * 30, 'ANALYSIS', "-" * 30 )
print( "Global Evidence:\n\t%.15e +- %.15e" % ( s['global evidence'], s['global evidence error'] ))

print( 'Parameter values:')
print( '1:',values[0]['median'],'pm',values[0]['sigma'])
print( '2:',values[1]['median'],'pm',values[1]['sigma'])
print( '3:',values[2]['median'],'pm',values[2]['sigma'])
fit_quad = [values[0]['median'],values[1]['median'],values[2]['median']]

'''----------------------------------------------------
Now do a linear fit 
----------------------------------------------------'''

print( 'Now doing a linear fit')

# number of dimensions our problem has
ndim = 2
n_params = ndim #oddly, this needs to be specified

tic = time.time()

pymultinest.run(myloglike_lin, myprior_lin,n_params, resume = False, verbose = True, sampling_efficiency = 0.3)

toc = time.time()

# lets analyse the results
a = pymultinest.Analyzer(n_params = n_params) #retrieves the data that has been written to hard drive
s = a.get_stats()
values_lin = s['marginals'] # gets the marginalized posterior probability distributions 
evidence_lin = s['global evidence']

print()
print( "-" * 30, 'ANALYSIS', "-" * 30 )
print( 'Time elapsed =',toc-tic,'s')
print( "Global Evidence:\n\t%.15e +- %.15e" % ( s['global evidence'], s['global evidence error'] ) )

print( 'Parameter values:' )
print( '1:',values_lin[0]['median'],'pm',values_lin[0]['sigma'])
print( '2:',values_lin[1]['median'],'pm',values_lin[1]['sigma'])
fit_lin = [values_lin[0]['median'],values_lin[1]['median']]

odds = evidence_quad-evidence_lin

print( "-" * 30, 'RESULTS', "-" * 30)

if odds >0:
	print( 'Log-odds in favour of quadratic fit by ',abs(odds))
	print( 'Params:',fit_quad)
	bestmodel = model_quad(fit_quad,xx)
else: 
	print( 'Log-odds in favour of linear fit by ',abs(odds))
	print( 'Params:',fit_lin)
	bestmodel = model_lin(fit_lin,xx)

plt.clf()
plt.errorbar(xx,data,yerr=error)
plt.plot(xx,bestmodel)
plt.legend(['Data','Best Fit'])
plt.title('MultiNest Fit to a Simple Function')
plt.xlabel('Ordinate')
plt.ylabel('Abscissa')
plt.show()