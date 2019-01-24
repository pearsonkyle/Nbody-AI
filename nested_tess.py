import matplotlib.pyplot as plt
import numpy as np
import pickle 
import corner 
import copy

import os
import pymultinest
if not os.path.exists("chains"): os.mkdir("chains")

from nbody.simulation import randomize, generate, integrate, analyze, report
from nbody.tools import mjup,msun,mearth,G,au,rearth,sa

from nested_nbody import lfit, get_ttv, nlfit, shift_align

if __name__ == "__main__":
    
    # units: Msun, Days, au
    # wasp-126 b
    objects = [
        {'m':1.12},
        {'m':0.28*mjup/msun, 'inc':3.14159/2,'e':0 }, 
        {}, # used for fitting 
    ]

    # load light curve fits 
    lcdata = pickle.load( open('lcdata114.pkl','rb') )
    objects[1]['P'] = lcdata['p']

    # get values from each light curve 
    ttdata = np.array( [lcdata['lcfits'][i]['NS']['parameters']['tm'] for i in range(len(lcdata['lcfits']))]).astype(np.float)[2:]
    err = np.array( [lcdata['lcfits'][i]['NS']['errors']['tm'] for i in range(len(lcdata['lcfits']))] ).astype(np.float)[2:]
    epochs =  np.array( [lcdata['lcfits'][i]['epoch'] for i in range(len(lcdata['lcfits']))] ).astype(np.int)[2:]
    ocdata = lcdata['ttv'][2:]

    # perform nested sampling linear fit to transit data    
    lstats, lposteriors = lfit(epochs,ttdata,err, 
                bounds=[lcdata['p']-1/24,lcdata['p']+1/24,lcdata['t0']-1./24,lcdata['t0']+1./24 ])
    print("linear fit complete ")
    import pdb; pdb.set_trace()

    # estimate priors
    bounds = []
    objects[1]['P'] = lstats['marginals'][0]['median']
    objects[2] = {'e': 0, 'inc': 1.570795} # delete mass and period for fitting routine  
    bounds = [
        lstats['marginals'][0]['5sigma'][0], lstats['marginals'][0]['5sigma'][1], # Period #1 (day)
        lstats['marginals'][1]['5sigma'][0], lstats['marginals'][1]['5sigma'][1], # t0 #1 
        mearth/msun, 4*mjup/msun,
        #objects[1]['m'] * 0.25, objects[1]['m'] * 5, # Mass #2 (msun)
        objects[1]['P'] * 1.5, objects[1]['P'] * 4, # Period #2 (day)
    ]

    print(bounds)

    # non linear fit with priors constrained from linear ephem fit 
    newobj, posteriors, stats = nlfit( epochs,ttdata,err, objects, bounds )

    # generate the best fit model 
    ndays = np.round( 2*(max(epochs)+1)*newobj[1]['P'] ).astype(int)
    epoch,ttvfit = get_ttv(newobj, ndays)
    ttvfit = shift_align(ocdata,ttvfit,epochs,err)

    # compute limits
    def limits(key='1sigma'):
        obj = copy.deepcopy(newobj)

        models = [] 
        # Planet 1 period
        obj[1]['P'] = stats['marginals'][0][key][1]
        epoch,ttv = get_ttv(obj, ndays)
        models.append( shift_align(ocdata,ttv,epochs,err) )
        obj = copy.deepcopy(newobj)
        obj[1]['P'] = stats['marginals'][0][key][0]
        epoch,ttv = get_ttv(obj, ndays)
        models.append( shift_align(ocdata,ttv,epochs,err) )
        obj = copy.deepcopy(newobj)

        # Planet 2 mass
        obj[2]['m'] = stats['marginals'][2][key][1]
        epoch,ttv = get_ttv(obj, ndays)
        models.append( shift_align(ocdata,ttv,epochs,err) )
        obj = copy.deepcopy(newobj)
        obj[2]['m'] = stats['marginals'][2][key][0]
        epoch,ttv = get_ttv(obj, ndays)
        models.append( shift_align(ocdata,ttv,epochs,err) )
        obj = copy.deepcopy(newobj)

        # Planet 2 period 
        obj[2]['P'] = stats['marginals'][3][key][1]
        epoch,ttv = get_ttv(obj, ndays)
        models.append( shift_align(ocdata,ttv,epochs,err) )
        obj = copy.deepcopy(newobj)
        obj[2]['P'] = stats['marginals'][3][key][0]
        epoch,ttv = get_ttv(obj, ndays)
        models.append( shift_align(ocdata,ttv,epochs,err) )
        obj = copy.deepcopy(newobj)

        # all parameters 
        obj[1]['P'] = stats['marginals'][0][key][1]
        obj[2]['m'] = stats['marginals'][2][key][1]
        obj[2]['P'] = stats['marginals'][3][key][1]
        epoch,ttv = get_ttv(obj, ndays)
        models.append( shift_align(ocdata,ttv,epochs,err) )
        obj = copy.deepcopy(newobj)
        obj[1]['P'] = stats['marginals'][0][key][0]
        obj[2]['m'] = stats['marginals'][2][key][0]
        obj[2]['P'] = stats['marginals'][3][key][0]
        epoch,ttv = get_ttv(obj, ndays)
        models.append( shift_align(ocdata,ttv,epochs,err) )
        obj = copy.deepcopy(newobj)

        # some combinations
        obj[2]['m'] = stats['marginals'][2][key][1]
        obj[2]['P'] = stats['marginals'][3][key][1]
        epoch,ttv = get_ttv(obj, ndays)
        models.append( shift_align(ocdata,ttv,epochs,err) )
        obj = copy.deepcopy(newobj)
        obj[2]['m'] = stats['marginals'][2][key][0]
        obj[2]['P'] = stats['marginals'][3][key][0]
        epoch,ttv = get_ttv(obj, ndays)
        models.append( shift_align(ocdata,ttv,epochs,err) )
        obj = copy.deepcopy(newobj)
        obj[2]['m'] = stats['marginals'][2][key][0]
        obj[2]['P'] = stats['marginals'][3][key][1]
        epoch,ttv = get_ttv(obj, ndays)
        models.append( shift_align(ocdata,ttv,epochs,err) )
        obj = copy.deepcopy(newobj)

        return np.max(models,0),np.min(models,0)
    
    upper3, lower3 = limits('3sigma')

    mask = posteriors[:,1] < np.median(posteriors[:,1])
    #posteriors[:,2] *= msun/mjup
    f = corner.corner(posteriors[mask,2:], 
                    labels=['Period (day)','t0','Mass 2 (msun)', 'Period 2 (day)'],
                    bins=int(np.sqrt(posteriors.shape[0])), 
                    range=[
                        ( stats['marginals'][0]['5sigma'][0], stats['marginals'][0]['5sigma'][1]),
                        ( stats['marginals'][1]['5sigma'][0], stats['marginals'][1]['5sigma'][1]),
                        ( max(0,stats['marginals'][2]['median']-stats['marginals'][2]['sigma']*3),stats['marginals'][2]['median']+stats['marginals'][2]['sigma']*3),
                        ( stats['marginals'][3]['median']-stats['marginals'][3]['sigma']*3,stats['marginals'][3]['median']+stats['marginals'][3]['sigma']*3),
                    ],
                    plot_contours=False, 
                    plot_density=False)
    plt.show()

    # TODO recompute o-c data with nl fit 

    # generate models between the uncertainties 
    f,ax = plt.subplots(1, figsize=(7,4))
    ax.errorbar(epochs,ocdata*24*60,yerr=err*24*60,ls='none',marker='o',label='Data',color='black')
    ax.plot(epoch, ttvfit*24*60, label='Linear+Nbody ({:.1f})'.format(stats['global evidence']),color='red')
    #ax.fill_between(epoch,24*60*upper3,24*60*lower3,alpha=0.1,label='Nbody 3 sigma',color='red')
    ax.axhline(ls='--',label='Linear ({:.1f})'.format(lstats['global evidence']))
    ax.legend(loc='best')
    ax.set_xlabel('Epochs')
    ax.set_xlim([min(epochs),max(epochs)])
    ax.set_ylabel('O-C [min]')
    ax.grid(True)
    plt.show()


    '''
    

    # load lc data 
    #lcdata = pickle.load( open('TESS/reports/114.01/lcdata.pkl','rb') )
    lcdata = pickle.load( open('TESS/reports/25375553.0/lcdata.pkl','rb') )

    epochs = np.array( [j['epoch'] for j in lcdata['lcfits'] ] )
    err = np.array( [j['NS']['errors']['tm'] for j in lcdata['lcfits'] ] )

    # generate data and try to retrieve 
    objects = [
        {'m':1.12},
        {'m':0.25*mjup/msun, 'P':lcdata['p'], 'inc':np.deg2rad(lcdata['phase']['NS']['parameters']['inc']),'e':0, 'omega':0  }, 
        {'m':0.5*mjup/msun, 'P':4, 'inc':np.pi/2,'e':0,  'omega':0  }, 
    ]

    
    objects = [
        {'m':1.12},
        {'m':0.25*mjup/msun, 'P':lcdata['p'], 'inc':np.pi/2,'e':0, 'omega':0  }, 
        {'m':posteriors[:,0].mean(), 'P':posteriors[:,1].mean(), 'inc':np.pi/2,'omega':0, 'e':posteriors[:,2].mean() }, 
    ]


    '''
