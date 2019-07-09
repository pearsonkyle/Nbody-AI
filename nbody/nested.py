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
    error[error==0] = np.mean( error[error!=0] )
    
    def model_lin(params):
        return params[0] + xx*params[1]

    def myprior_lin(cube, ndim, n_params):
        '''This transforms a unit cube into the dimensions of your prior space.'''
        cube[0] = (bounds[1] - bounds[0])*cube[0]+bounds[0]
        cube[1] = (bounds[3] - bounds[2])*cube[1]+bounds[2]

    def myloglike_lin(cube, ndim, n_params):
        loglike = -np.sum( ((yy-model_lin(cube))/error)**2 )
        return loglike/2.

    pymultinest.run(myloglike_lin, myprior_lin, 2, 
        resume = False, sampling_efficiency = 0.5, 
        evidence_tolerance=0.1,
    )

    # retrieves the data that has been written to hard drive
    a = pymultinest.Analyzer(n_params=2)
    posteriors = a.get_data()
    stats = a.get_stats()
    stats['marginals'] = get_stats(posteriors)

    return stats, posteriors

def get_ttv(objects, ndays=60, ttvfast=True):
    sim = generate(objects)
    sim_data = integrate(sim, objects, ndays, ndays*24*2) # dt=30 minutes
    return analyze(sim_data, ttvfast=ttvfast)

def nlfit( xx,yy,yerr, objects, bounds, myloss='linear'):
    # have bounds that follow same format as objects
    
    # format bounds list
    blist = [bounds[0][0], bounds[0][1]] # tmid
    for i in range(1,len(bounds)):
        for k in bounds[i].keys():
            blist.append( bounds[i][k][0] )
            blist.append( bounds[i][k][1] )

    # compute integrations for as long as the max epoch 
    ndays = np.round( 1.5*(max(xx)+1)*objects[1]['P'] ).astype(int)

    # prevents seg fault in MultiNest
    yerr[yerr==0] = np.mean( yerr[yerr!=0] )
    
    def model_sim(params):
        c=1
        for i in range(1,len(bounds)):
            for k in bounds[i].keys():                
                objects[i][k] = params[c]
                c+=1
        # create REBOUND simulation
        return get_ttv(objects,ndays)

    def myprior(cube, ndim, n_params):
        for i in range(int(len(blist)/2)): # for only the free params
            cube[i] = (blist[2*i+1] - blist[2*i])*cube[i]+blist[2*i]

    loss = {
        'linear': lambda z: z,
        'soft_l1' : lambda z : 2 * ((1 + z)**0.5 - 1),
    }

    omegas = [] 

    def myloglike(cube, ndim, n_params):
        epochs,ttv,tt = model_sim(cube)
        ttv_data = yy - (cube[1]*xx+cube[0])
        try:
            loglike = -np.sum( loss[myloss]( ((ttv_data-ttv[xx.astype(int)])/yerr)**2 ) )*0.5
            return loglike
        except:
            loglike = 0
            for i in range(len(ttv)):
                loglike += loss[myloss]( ((ttv_data[i]-ttv[i])/yerr[i])**2 )
            return -10*np.sum( loglike) # penalize for unstable orbits

        # period usually changes ~30 seconds to a minute after N-body integrations
        # therefore the method below subtracts the wrong period from the data
        #ttvm = tt[xx.astype(int)] - (cube[1]*xx+cube[0])
        #return -np.sum( ((ttv_data-ttvm)/yerr)**2 )*0.5


    pymultinest.run(myloglike, myprior, int(len(blist)/2), resume=False,
        evidence_tolerance=0.5, sampling_efficiency=0.5,
        n_live_points=200, verbose=True
    )

    a = pymultinest.Analyzer(n_params=int(len(blist)/2)) 

    # gets the marginalized posterior probability distributions 
    posteriors = a.get_data()
    stats = a.get_stats()
    stats['marginals'] = get_stats(posteriors)

    # map posteriors back to object dict
    obj = copy.deepcopy(objects)
    mask = posteriors[:,1] < np.percentile(posteriors[:,1],50)

    c = 1
    for i in range(1,len(bounds)):
        for k in bounds[i].keys():                
            obj[i][k] = np.median(posteriors[mask,2+c])
            c+=1

    return obj, posteriors, stats

def get_stats(posterior,percentile=75):
    # perform burn-in by taking top 50th percentile of fits 
    
    stats = []
    mask = (posterior[:,1] < np.percentile(posterior[:,1],percentile))

    for i in range(2, posterior.shape[1]):
        b = list(zip(posterior[mask,0], posterior[mask,i]))
        b.sort(key=lambda x: x[1])
        b = np.array(b)
        b[:,0] = b[:,0].cumsum()
        sig5 = 0.5 + 0.9999994 / 2.
        sig3 = 0.5 + 0.9973 / 2.
        sig2 = 0.5 + 0.95 / 2.
        sig1 = 0.5 + 0.6826 / 2.
        bi = lambda x: np.interp(x, b[:,0], b[:,1], left=b[0,1], right=b[-1,1])
        
        low1 = bi(1 - sig1)
        high1 = bi(sig1)
        low2 = bi(1 - sig2)
        high2 = bi(sig2)
        low3 = bi(1 - sig3)
        high3 = bi(sig3)
        low5 = bi(1 - sig5)
        high5 = bi(sig5)
        median = bi(0.5)
        q1 = bi(0.75)
        q3 = bi(0.25)
        q99 = bi(0.99)
        q01 = bi(0.01)
        q90 = bi(0.9)
        q10 = bi(0.1)
        
        stats.append({
            'median': median,
            'sigma': (high1 - low1) / 2.,
            '1sigma': [low1, high1],
            '2sigma': [low2, high2],
            '3sigma': [low3, high3],
            '5sigma': [low5, high5],
            'q75%': q1,
            'q25%': q3,
            'q99%': q99,
            'q01%': q01,
            'q90%': q90,
            'q10%': q10,
        })
    return stats

def nbody_limits( newobj, nlstats, n=1):
    # TODO create wrapper?
    upper = np.copy( ttv )
    lower = np.copy( ttv )
    
    obj = copy.deepcopy(newobj)
    obj[2]['m'] = nlstats['marginals'][3]['median']-n*nlstats['marginals'][3]['sigma']
    obj[2]['omega'] = nlstats['marginals'][5]['median']-n*nlstats['marginals'][5]['sigma']
    sim_data = generate(obj, newobj[1]['P']* (len(ttv)+1), int( (len(ttv)+1)*newobj[1]['P']*24) )
    ttv_data = analyze(sim_data)
    for i in range( len(ttv_data['planets'][0]['ttv']) ):
        try:
            upper[i] = max( upper[i],ttv_data['planets'][0]['ttv'][i] )
            lower[i] = min( lower[i],ttv_data['planets'][0]['ttv'][i] )
        except:
            pass

    obj = copy.deepcopy(newobj)
    obj[2]['m'] = nlstats['marginals'][3]['median']-n*nlstats['marginals'][3]['sigma']
    sim_data = generate(obj, newobj[1]['P']* (len(ttv)+1), int( (len(ttv)+1)*newobj[1]['P']*24) )
    ttv_data = analyze(sim_data)
    for i in range( len(ttv_data['planets'][0]['ttv']) ):
        try:
            upper[i] = max( upper[i],ttv_data['planets'][0]['ttv'][i] )
            lower[i] = min( lower[i],ttv_data['planets'][0]['ttv'][i] )
        except:
            pass

    obj = copy.deepcopy(newobj)
    obj[2]['m'] = nlstats['marginals'][3]['median']+n*nlstats['marginals'][3]['sigma']
    sim_data = generate(obj, newobj[1]['P']* (len(ttv)+1), int( (len(ttv)+1)*newobj[1]['P']*24) )
    ttv_data = analyze(sim_data)
    for i in range( len(ttv_data['planets'][0]['ttv']) ):
        try:
            upper[i] = max( upper[i],ttv_data['planets'][0]['ttv'][i] )
            lower[i] = min( lower[i],ttv_data['planets'][0]['ttv'][i] )
        except:
            pass

    obj = copy.deepcopy(newobj)
    obj[2]['m'] = nlstats['marginals'][3]['median']-n*nlstats['marginals'][3]['sigma']
    obj[2]['omega'] = nlstats['marginals'][5]['median']+n*nlstats['marginals'][5]['sigma']
    sim_data = generate(obj, newobj[1]['P']* (len(ttv)+1), int( (len(ttv)+1)*newobj[1]['P']*24) )
    ttv_data = analyze(sim_data)
    for i in range( len(ttv_data['planets'][0]['ttv']) ):
        try:
            upper[i] = max( upper[i],ttv_data['planets'][0]['ttv'][i] )
            lower[i] = min( lower[i],ttv_data['planets'][0]['ttv'][i] )
        except:
            pass

    obj = copy.deepcopy(newobj)
    obj[2]['m'] = nlstats['marginals'][3]['median']+n*nlstats['marginals'][3]['sigma']
    obj[2]['omega'] = nlstats['marginals'][5]['median']-n*nlstats['marginals'][5]['sigma']
    sim_data = generate(obj, newobj[1]['P']* (len(ttv)+1), int( (len(ttv)+1)*newobj[1]['P']*24) )
    ttv_data = analyze(sim_data)
    for i in range( len(ttv_data['planets'][0]['ttv']) ):
        try:
            upper[i] = max( upper[i],ttv_data['planets'][0]['ttv'][i] )
            lower[i] = min( lower[i],ttv_data['planets'][0]['ttv'][i] )
        except:
            pass

    obj = copy.deepcopy(newobj)
    obj[2]['m'] = nlstats['marginals'][3]['median']+n*nlstats['marginals'][3]['sigma']
    obj[2]['omega'] = nlstats['marginals'][5]['median']+n*nlstats['marginals'][5]['sigma']
    sim_data = generate(obj, newobj[1]['P']* (len(ttv)+1), int( (len(ttv)+1)*newobj[1]['P']*24) )
    ttv_data = analyze(sim_data)
    for i in range( len(ttv_data['planets'][0]['ttv']) ):
        try:
            upper[i] = max( upper[i],ttv_data['planets'][0]['ttv'][i] )
            lower[i] = min( lower[i],ttv_data['planets'][0]['ttv'][i] )
        except:
            pass
    return upper, lower
