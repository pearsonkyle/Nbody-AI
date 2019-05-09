import matplotlib.pyplot as plt
import numpy as np
import pickle 
import copy 

import os
import pymultinest
if not os.path.exists("chains"): os.mkdir("chains")

from nbody.simulation import randomize, generate, integrate, analyze, report
from nbody.tools import mjup,msun,mearth,G,au,rearth,sa

def lfit(xx,yy,error, bounds=[-10.,10.,-10.,10.]):
    # linear fit with nested sampling

    # prevents seg fault in MultiNest
    error[error==0] = 1
    
    def model_lin(params):
        return xx*params[0]+params[1]

    def myprior_lin(cube, ndim, n_params):
        '''This transforms a unit cube into the dimensions of your prior space.'''
        cube[0] = (bounds[1] - bounds[0])*cube[0]+bounds[0]
        cube[1] = (bounds[3] - bounds[2])*cube[1]+bounds[2]

    def myloglike_lin(cube, ndim, n_params):
        loglike = -np.sum( ((yy-model_lin(cube))/error)**2 )
        return loglike/2.

    pymultinest.run(myloglike_lin, myprior_lin, 2, resume = False, sampling_efficiency = 0.5)

    # retrieves the data that has been written to hard drive
    a = pymultinest.Analyzer(n_params=2)
    return a.get_stats(), a.get_data()

def get_ttv(objects, ndays=60, ttvfast=True):
    sim = generate(objects)
    sim_data = integrate(sim, objects, ndays, ndays*24*2) # dt=30 minutes
    return analyze(sim_data, ttvfast=ttvfast)

def nlpfit( xx,yy,yerr, objects, bounds, myloss='linear'):
    '''
        have bounds that follow same format as objects
    '''

    # format bounds list
    blist = [bounds[0][0], bounds[0][1]] # tmid
    for i in range(1,len(bounds)):
        for k in bounds[i].keys():
            blist.append( bounds[i][k][0] )
            blist.append( bounds[i][k][1] )

    # compute integrations for as long as the max epoch 
    ndays = np.round( 1.5*(max(xx)+1)*objects[1]['P'] ).astype(int)

    # prevents seg fault in MultiNest
    yerr[yerr==0] = 1
    
    def model_sim(params):
        c=1
        for i in range(1,len(bounds)):
            for k in bounds[i].keys():                
                objects[i][k] = params[c]
                c+=1
        # create REBOUND simulation
        return get_ttv(objects,ndays)

    def myprior(cube, ndim, n_params):
        '''This transforms a unit cube into the dimensions of your prior space.'''
        for i in range(int(len(blist)/2)): # for only the free params
            cube[i] = (blist[2*i+1] - blist[2*i])*cube[i]+blist[2*i]

    loss = {
        'linear': lambda z: z,
        'soft_l1' : lambda z : 2 * ((1 + z)**0.5 - 1),
    }

    omegas = [] 

    def myloglike(cube, ndim, n_params):
        epochs,ttv,tt = model_sim(cube)
        ttv_data = yy - (cube[1]*xx+cube[2])
        ttvm = tt[xx.astype(int)] - (cube[1]*xx+cube[2])
        return -np.sum( ((ttv_data-ttvm)/yerr)**2 )*0.5
        #return -np.sum( ((ttv_data-ttv)/yerr)**2 )*0.5


    pymultinest.run(myloglike, myprior, int(len(blist)/2), resume=False,
                    evidence_tolerance=0.5, sampling_efficiency=0.5,# n_clustering_params=2,
                    n_live_points=200, verbose=True)

    a = pymultinest.Analyzer(n_params=int(len(blist)/2)) 

    # gets the marginalized posterior probability distributions 
    s = a.get_stats()
    posteriors = a.get_data()

    # map posteriors back to object dict
    obj = copy.deepcopy(objects)
    mask = posteriors[:,1] < np.percentile(posteriors[:,1],25)
    import pdb; pdb.set_trace() 
    c = 1
    for i in range(1,len(bounds)):
        for k in bounds[i].keys():                
            obj[i][k] = np.mean(posteriors[mask,2+c])
            c+=1

    return obj, posteriors, s



'''
# deprecated functions

def nlfit( xx,yy,yerr, objects, bounds=[-1,1,-1,1,-1,1], myloss='soft_l1'):

    # compute integrations for as long as the max epoch 
    ndays = np.round( 2*(max(xx)+1)*objects[1]['P'] ).astype(int)

    # prevents seg fault in MultiNest
    yerr[yerr==0] = 1
    
    def model_sim(params):
        # follow same order as bounds 
        objects[1]['P'] = params[0]
        objects[2]['m'] = params[2]
        objects[2]['P'] = params[3]
        # create REBOUND simulation
        return get_ttv(objects,ndays)

    def myprior(cube, ndim, n_params):
        '''This transforms a unit cube into the dimensions of your prior space.'''
        for i in range(int(len(bounds)/2)): # for only the free params
            cube[i] = (bounds[2*i+1] - bounds[2*i])*cube[i]+bounds[2*i]

    loss = {
        'linear': lambda z: z,
        'soft_l1' : lambda z : 2 * ((1 + z)**0.5 - 1),
    }

    def myloglike(cube, ndim, n_params):

        # compute ttv from nbody 
        epochs,ttv = model_sim(cube)

        # subtract period from data 
        ttv_data = yy - (cube[0]*xx+cube[1])

        # fudge argument of periastron with shift of transit epochs 
        try:
            chis = []
            # only works for 1 planet 
            for i in range(len(epochs)-len(xx)):
                chi2 = ((ttv_data-np.roll(ttv,-i)[xx])/yerr)**2 
                chis.append( -np.sum( loss[myloss](chi2) ) )
            
            # check return 
            #plt.plot(xx,yy,'ko');plt.plot(epochs[xx],ttv[xx],'r-'); plt.plot(epochs[xx],np.roll(ttv,-2)[xx],'g-'); plt.show()
            #import pdb; pdb.set_trace() 

            return np.max(chis)
        except:
            #print('error with model, -999')
            #import pdb; pdb.set_trace()
            print("error with model:",cube[0],cube[1],cube[2],cube[3])
            return -999 

    pymultinest.run(myloglike, myprior, int(len(bounds)/2), resume=False,
                    evidence_tolerance=0.5, sampling_efficiency=0.5, n_clustering_params=2,
                    n_live_points=200, verbose=True)

    a = pymultinest.Analyzer(n_params=4) 

    # gets the marginalized posterior probability distributions 
    s = a.get_stats()
    posteriors = a.get_data()

    mask = posteriors[:,1] < np.percentile(posteriors[:,1],25)
    objects[1]['P'] = np.mean(posteriors[mask,2])
    objects[2]['m'] = np.mean(posteriors[mask,4])
    objects[2]['P'] = np.mean(posteriors[mask,5])
    
    return objects, posteriors, s

def nlpfit2( xx,yy,yerr, objects, bounds, myloss='linear'):
     #   have bounds that follow same format as objects

    # format bounds list
    blist = []
    for i in range(len(bounds)):
        for k in bounds[i].keys():
            blist.append( bounds[i][k][0] )
            blist.append( bounds[i][k][1] )

    # compute integrations for as long as the max epoch 
    ndays = np.round( 3*(max(xx)+1)*objects[1]['P'] ).astype(int)

    # prevents seg fault in MultiNest
    yerr[yerr==0] = 1
    
    def model_sim(params):
        c=0
        for i in range(len(bounds)):
            for k in bounds[i].keys():                
                objects[i][k] = params[c]
                c+=1

        # create REBOUND simulation
        return get_ttv(objects,ndays)

    def myprior(cube, ndim, n_params):
        '''This transforms a unit cube into the dimensions of your prior space.'''
        for i in range(int(len(blist)/2)): # for only the free params
            cube[i] = (blist[2*i+1] - blist[2*i])*cube[i]+blist[2*i]

    loss = {
        'linear': lambda z: z,
        'soft_l1' : lambda z : 2 * ((1 + z)**0.5 - 1),
    }

    omegas = [] 

    def myloglike(cube, ndim, n_params):
        try:
            # compute ttv from nbody 
            epochs,ttv = model_sim(cube)

            # computer piece wise averages 
            tmids = np.linspace(min(yy)-0.125/24,min(yy)+0.125/24,30*60)
            
            chis = []
            for i in range(tmids.shape[0]):
                ttv_data = yy - (cube[0]*xx+tmids[i])

                # marginalize omega 
                bestchi, besti = shift_align(ttv_data, ttv, xx, yerr, return_chi=True)
                chis.append( bestchi )
                omegas.append( besti )

            return np.max(chis)

        except:
            print('error with model, -999')
            import pdb; pdb.set_trace()
            return -999 

    pymultinest.run(myloglike, myprior, int(len(blist)/2), resume=False,
                    evidence_tolerance=0.5, sampling_efficiency=0.5,# n_clustering_params=2,
                    n_live_points=200, verbose=True)

    a = pymultinest.Analyzer(n_params=int(len(blist)/2)) 

    # gets the marginalized posterior probability distributions 
    s = a.get_stats()
    posteriors = a.get_data()
    
    # map posteriors back to object dict
    obj = copy.deepcopy(objects)
    mask = posteriors[:,1] < np.percentile(posteriors[:,1],25)
    c = 0
    for i in range(len(bounds)):
        for k in bounds[i].keys():                
            obj[i][k] = np.mean(posteriors[mask,2+c])
            c+=1

    print('check that omegas == posterior.shape[0]')
    import pdb; pdb.set_trace()
    return obj, posteriors, s
'''