import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pickle 
import corner 
import copy

import os
import pymultinest
if not os.path.exists("chains"): os.mkdir("chains")

from nbody.simulation import randomize, generate, integrate, analyze, report
from nbody.tools import mjup,msun,mearth,G,au,rearth,sa

from nested_nbody import lfit, get_ttv, nlfit, shift_align, get_stats

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
    stats['marginals'] = get_stats(posteriors,20,posteriors[:,5]<8)

    # generate the best fit model 
    ndays = np.round( (max(epochs)+1)*newobj[1]['P'] ).astype(int)
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
    

    colors = [ (1, 0, 0), (0,1,0), (0, 0, 0)] 
    cm = LinearSegmentedColormap.from_list("mylist", colors, N=3)
    posteriors[:,4]*=msun/mearth
    f = corner(posteriors[:,2:], 
                    labels=['Period (day)','t0','Mass 2 (Earth)', 'Period 2 (day)'],
                    bins=int(np.sqrt(posteriors.shape[0])), 
                    #range=[
                    #    ( stats['marginals'][0]['5sigma'][0], stats['marginals'][0]['5sigma'][1]),
                    #    ( stats['marginals'][1]['5sigma'][0], stats['marginals'][1]['5sigma'][1]),
                    #    ( stats['marginals'][2]['5sigma'][0], stats['marginals'][2]['5sigma'][1]),
                    #    ( stats['marginals'][3]['5sigma'][0], stats['marginals'][3]['5sigma'][1]),
                    #],
                    #truths=[lstats['marginals'][0]['median'], lstats['marginals'][1]['median'], stats['marginals'][2]['median'],stats['marginals'][2]['5sigma'][0] ],
                    #no_fill_contours=True,
                    truth_color='green',
                    plot_contours=False,
                    plot_density=False,
                    data_kwargs={'c':posteriors[:,1],'vmin':np.percentile(posteriors[:,1],1),'vmax':np.percentile(posteriors[:,1],50),'cmap':cm},
                    )
    posteriors[:,4]/=msun/mearth
    plt.show()

    # generate models between the uncertainties 
    f,ax = plt.subplots(1, figsize=(7,4))
    ax.errorbar(epochs,ocdata*24*60,yerr=err*24*60,ls='none',marker='o',label='Data',color='black')
    #ax.plot(epochs, ttv_data['planets'][0]['ttv']*24*60,ls='--', label='Truth',color='green')
    ax.plot(epoch, ttvfit*24*60, label='Linear+Nbody ({:.1f})'.format(stats['global evidence']),color='red')
    ax.fill_between(epoch,24*60*upper3,24*60*lower3,alpha=0.1,label='Nbody 3 sigma',color='red')
    ax.axhline(ls='--',label='Linear ({:.1f})'.format(lstats['global evidence']))
    ax.legend(loc='best')
    ax.set_xlabel('Epochs')
    ax.set_xlim([min(epochs),max(epochs)])
    ax.set_ylabel('O-C [min]')
    ax.grid(True)
    plt.show()

    
    f,ax = plt.subplots( 2,2, figsize=(7,4))
    im = ax[1,0].scatter(posteriors[:,4]*msun/mearth,posteriors[:,5],s=3.0,alpha=0.25,c=posteriors[:,1],vmin=np.percentile(posteriors[:,1],1),vmax=np.percentile(posteriors[:,1],50),cmap=cm)
    cbar = f.colorbar(im, ax=ax[1,0], orientation='vertical')
    cbar.ax.set_ylabel('-2 * log likelihood', rotation=270)

    ax[1,0].set_xlabel("Planet 2 Mass (Earth)")
    ax[1,0].set_ylabel("Planet 2 Period (day)")
    bins_mass = np.linspace( min(posteriors[:,4])*msun/mearth,max(posteriors[:,4])*msun/mearth,int(np.sqrt(posteriors.shape[0])) )
    bins_per = np.linspace( min(posteriors[:,5]),max(posteriors[:,5]),2*int(np.sqrt(posteriors.shape[0])) )
    ax[0,1].set_axis_off()
    ax[1,1].set_xlabel("Planet 2 Period (day)")
    ax[0,0].set_xlabel("Planet 2 Mass (Earth)")

    ax[0,0].hist( posteriors[:,4]*msun/mearth, bins=bins_mass,alpha=0.5, histtype=u'step',color='black'  )
    ax[1,1].hist( posteriors[:,5], bins=bins_per,alpha=0.5, histtype=u'step',color='black' )

    mask = (posteriors[:,1] < np.percentile(posteriors[:,1],35)) & (posteriors[:,5] < 8)
    ax[0,0].hist( posteriors[mask,4]*msun/mearth, bins=bins_mass,alpha=0.5, label="{:.1f}+-{:.1f}".format(np.mean(posteriors[mask,4]*msun/mearth),np.std(posteriors[mask,4]*msun/mearth)), histtype=u'step',color='red' )
    ax[1,1].hist( posteriors[mask,5], bins=bins_per,alpha=0.5, label="{:.2f}+-{:.2e}".format(np.mean(posteriors[mask,5]),np.std(posteriors[mask,5])), histtype=u'step',color='red' )

    mask = (posteriors[:,1] < np.percentile(posteriors[:,1],35)) & (posteriors[:,5] > 9)
    ax[0,0].hist( posteriors[mask,4]*msun/mearth, bins=bins_mass,alpha=0.5, label="{:.1f}+-{:.1f}".format(np.mean(posteriors[mask,4]*msun/mearth),np.std(posteriors[mask,4]*msun/mearth)), histtype=u'step',color='green' )
    ax[1,1].hist( posteriors[mask,5], bins=bins_per,alpha=0.5, label="{:.2f}+-{:.2e}".format(np.mean(posteriors[mask,5]),np.std(posteriors[mask,5])), histtype=u'step',color='green' )

    ax[0,0].yaxis.set_visible(False)
    ax[1,1].yaxis.set_visible(False)

    ax[0,0].legend(loc='best')
    ax[1,1].legend(loc='best')
    plt.show()

    # explore RV space and variability of signal 
    #plt.plot((ttv_data['times'][1:]/objects[1]['P'])%1,ttv_data['RV']['signal'],'ko')
    plt.xlabel("Phase")
    plt.ylabel("RV Semi-amplitude (m/s)")
    phase = ttv_data['times'][1:]/objects[1]['P']
    for i in range(1,int(max((ttv_data['times'][1:]/objects[1]['P'])))):
        mask = (phase < i) & (phase > i-1)
        plt.plot(phase[mask]%1, ttv_data['RV']['signal'][mask])
    plt.show()

    # explore RV space of the second mode seen in the colored posterior plot 

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
