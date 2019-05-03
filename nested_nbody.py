from collections import OrderedDict
import numpy as np
import pickle 
import corner 
import copy

import matplotlib
import matplotlib.pyplot as plt
#from matplotlib.colors import LinearSegmentedColormap

from nbody.ai import build_encoder
from nbody.nested import lfit, nlfit
from nbody.simulation import generate, analyze, report
from nbody.tools import mjup,msun,mearth,G,au,rearth,sa

def shift_align(ttv_data,ttv,xx,yerr, return_chi=False):
    chis = []
    # only works for 1 planet 
    for i in range(len(ttv)-len(xx)):
        chi2 = ((ttv_data-np.roll(ttv,-i)[xx])/yerr)**2 
        chis.append( -np.sum(chi2) )
    if return_chi:
        return np.max(chis), np.argmax(chis)
    else:
        # return shifted model
        i = np.argmax(chis)
        return np.roll(ttv, -i)

def get_stats(posterior,percentile,custom_mask=-1):
    
    stats = []
    mask = (posterior[:,1] < np.percentile(posterior[:,1],percentile))
    if np.sum(custom_mask) != -1:
        mask = mask & custom_mask

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
# check return 
# plt.plot(xx,yy,'ko');plt.plot(epochs[xx],ttv[xx],'r-'); plt.plot(epochs[xx],np.roll(ttv,-2)[xx],'g-'); plt.show()
# import pdb; pdb.set_trace() 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    help_ = "Load h5 model trained weights"
    parser.add_argument("-w", "--weights", help=help_, default="encoder.h5")
    help_ = "Number of training epochs"
    parser.add_argument("-e", "--epochs", help=help_, default=10 ,type=int)
    help_ = "Pickle file of training samples"
    parser.add_argument("-tr", "--train", help=help_)
    help_ = "Pickle file of test samples"
    parser.add_argument("-te", "--test", help=help_)
    args = parser.parse_args()

    # units: Msun, Days, au
    objects = [
        {'m': 1.12},
        {'m': 0.25*mjup/msun,
        'inc': 1.570795,
        'e': 0,
        'P': 3.2887967652699728},
        {'e': 0.06, 'inc': 1.570795, 'm': 100*mearth/msun, 'P': 7.29, 'omega':np.pi/3, }
    ]

    # create REBOUND simulation
    sim_data = generate(objects, 31*objects[1]['P'], int(31*objects[1]['P']*24) ) 
    ttv_data = analyze(sim_data)
    report(ttv_data)

    # simulate some observational data with noise 
    ttv = ttv_data['planets'][0]['ttv']
    epochs = np.arange(len(ttv))
    ttdata = ttv_data['planets'][0]['tt'] + np.random.normal(0,0.5,len(ttv))/(24*60)
    err = np.random.normal(90,30,len(ttv))/(24*60*60)

    # perform nested sampling linear fit to transit data    
    lstats, lposteriors = lfit( 
        epochs,ttdata,err, 
        bounds=[ objects[1]['P']-1, objects[1]['P']+1, 
                 min(ttdata)-1, objects[1]['P']+1
        ] 
    )

    ocdata = ttdata - (epochs*lstats['marginals'][0]['median'] + lstats['marginals'][1]['median'] )

    f = corner.corner(lposteriors[:,2:], 
                    labels=['Period (day)', 'Mid-transit (day)'],
                    bins=int(np.sqrt(lposteriors.shape[0])), 
                    range=[
                        ( lstats['marginals'][0]['5sigma'][0], lstats['marginals'][0]['5sigma'][1]),
                        ( lstats['marginals'][1]['5sigma'][0], lstats['marginals'][1]['5sigma'][1]),
                    ],
                    #no_fill_contours=True,
                    plot_contours=False,
                    plot_density=False,
                    data_kwargs={'c':lposteriors[:,1],'vmin':np.percentile(lposteriors[:,1],10),'vmax':np.percentile(lposteriors[:,1],50) },
                    )
    plt.show()

    dude()

    # estimate priors
    bounds = []
    objects[1]['P'] = lstats['marginals'][0]['median']
    objects[2] = {'e': 0, 'inc': 1.570795}
    
    bounds = [
        OrderedDict({}), # bounds for star, leave in as placeholder if none
        OrderedDict({
            'P':[lstats['marginals'][0]['5sigma'][0], lstats['marginals'][0]['5sigma'][1] ], # Period #1 (day)
        }),
        
        OrderedDict({
            'P':[objects[1]['P'] * 1.5, objects[1]['P'] * 2.5], # Period #2 (day)
            'm':[objects[1]['m'] * 0.5, objects[1]['m'] * 2], # Mass #2 (msun)
            'e':[0,0.1],
            #'inc':[0.8*np.pi/2,np.pi/2],
            #'omega':[0,np.pi],
        }),
    ]

    # non linear fit with priors constrained from linear ephem fit 
    newobj, posteriors, stats = nlpfit( epochs,ttdata,err, objects, bounds, lstats['marginals'][1]['median'] )

    print('finished')
    dude()

    stats['marginals'] = get_stats(posteriors,30)

    # generate the best fit model 
    ndays = np.round( 2*(max(epochs)+1)*newobj[1]['P'] ).astype(int)
    epoch,ttvfit = get_ttv(newobj, ndays)

    # figure out the optimal omega and t0 value 
    tmids = np.linspace(min(ttdata)-0.125/24,min(ttdata)+0.125/24,30*60)
    chis = []
    for i in range(tmids.shape[0]):
        ttvdata = ttdata - (newobj[1]['P']*epochs + tmids[i])

        # marginalize omegas
        chis.append( shift_align(ttvdata,ttvfit, epochs, err, return_chi=True) )   
    ti = np.argmax(chis)
    ttvbest = shift_align(ttvdata,ttvfit, epochs, err, return_chi=False)
    
    # TODO figure out the omega value
    # create an alignment algorithm that returns everything, the t0, omega, best fit, chi2 


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

        try:
            return np.max(models,0),np.min(models,0)
        except:
            # process the results point by points
            lengths = [len(models[i]) for i in range(len(models)) ]
            maxa = np.zeros(max(lengths))
            mina = np.zeros(max(lengths))

            # loop through data points        
            for i in range( max(lengths) ):
                mini = []
                for j in range(len(models)):
                    try:
                        mini.append( models[j][i] )
                    except:
                        pass
                maxa[i] = np.max(mini)
                mina[i] = np.min(mini)

            return maxa,mina

    upper3, lower3 = limits('3sigma')
    
    colors = [ (1, 0, 0), (0,1,0), (0, 0, 0)] 
    cm = LinearSegmentedColormap.from_list("mylist", colors, N=3)
    f = corner.corner(posteriors[:,2:], 
                    labels=['Period (day)', 'Period 2 (day)','Mass 2 (sun)', 'Eccentricity 2'],
                    bins=int(np.sqrt(posteriors.shape[0])), 
                    range=[
                        ( stats['marginals'][0]['5sigma'][0], stats['marginals'][0]['5sigma'][1]),
                        ( stats['marginals'][1]['5sigma'][0], stats['marginals'][1]['5sigma'][1]),
                        ( stats['marginals'][2]['5sigma'][0], stats['marginals'][2]['5sigma'][1]),
                        ( stats['marginals'][3]['5sigma'][0], stats['marginals'][3]['5sigma'][1]),
                    ],
                    truths=[ttv_data['planets'][0]['P'], ttv_data['planets'][1]['P'], objects[2]['m'], ttv_data['planets'][1]['e'] ],
                    #no_fill_contours=True,
                    truth_color='green',
                    plot_contours=False,
                    plot_density=False,
                    data_kwargs={'c':posteriors[:,1],'vmin':np.percentile(posteriors[:,1],1),'vmax':np.percentile(posteriors[:,1],50),'cmap':LinearSegmentedColormap.from_list("mylist", colors, N=3)},
                    )
    plt.show()

    # generate models between the uncertainties 
    f,ax = plt.subplots(1, figsize=(7,4))
    ax.errorbar(epochs,ocdata*24*60,yerr=err*24*60,ls='none',marker='o',label='Data',color='black')
    ax.plot(epochs, ttv_data['planets'][0]['ttv']*24*60, ls='--', label='Truth',color='green')
    ax.plot(epoch, ttvbest*24*60, label='Linear+Nbody ({:.1f})'.format(stats['global evidence']),color='red')
    ax.fill_between(epoch,24*60*upper3,24*60*lower3,alpha=0.1,label='Nbody 3 sigma',color='red')
    ax.axhline(ls='--',label='Linear ({:.1f})'.format(lstats['global evidence']))
    ax.legend(loc='best')
    ax.set_xlabel('Epochs')
    ax.set_xlim([min(epochs),max(epochs)])
    ax.set_ylabel('O-C [min]')
    ax.grid(True)
    plt.show()

    
    f,ax = plt.subplots( 2,2, figsize=(7,4))
    im = ax[1,0].scatter(posteriors[:,4]*msun/mearth,posteriors[:,5],c=posteriors[:,1],vmin=min(posteriors[:,1]),vmax=np.percentile(posteriors[:,1],50)); 
    f.colorbar(im, ax=ax[1,0], orientation='vertical')
    ax[1,0].set_xlabel("Planet 2 Mass (Earth)")
    ax[1,0].set_ylabel("Planet 2 Period (day)")
    bins_mass = np.linspace( min(posteriors[:,4])*msun/mearth,max(posteriors[:,4])*msun/mearth,int(np.sqrt(posteriors.shape[0])) )
    bins_per = np.linspace( min(posteriors[:,5]),max(posteriors[:,5]),2*int(np.sqrt(posteriors.shape[0])) )
    ax[0,1].set_axis_off()
    ax[1,1].set_xlabel("Planet 2 Period (day)")

    for i in [100,25]:
        mask = posteriors[:,1] < np.percentile(posteriors[:,1],i)
        ax[0,0].hist( posteriors[mask,4]*msun/mearth, bins=bins_mass,alpha=0.5, label="{:.1f}+-{:.1f}".format(np.mean(posteriors[mask,4]*msun/mearth),np.std(posteriors[mask,4]*msun/mearth)) )
        ax[1,1].hist( posteriors[mask,5], bins=bins_per,alpha=0.5, label="{:.2f}+-{:.2e}".format(np.mean(posteriors[mask,5]),np.std(posteriors[mask,5])) )
    ax[0,0].legend(loc='best')
    ax[1,1].legend(loc='best')
    plt.show()


    # explore uncertainty as a function of percentile 
    #for i in np.linspace(10,90,20):
    #    stat = get_stats(posteriors, i)
    #    plt.errorbar(i, stat[-2]['median'],yerr=stat[-2]['sigma'] )
    #plt.show()

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
