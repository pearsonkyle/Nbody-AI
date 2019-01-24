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
    sim_data = integrate(sim, objects, ndays, ndays*24) # year long integration, dt=1 hour
    return analyze(sim_data, ttvfast=ttvfast)

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
            print('error with model, -999')
            import pdb; pdb.set_trace()
            return -999 

    pymultinest.run(myloglike, myprior, int(len(bounds)/2), resume=False,
                    evidence_tolerance=0.5, sampling_efficiency=0.5, n_clustering_params=2,
                    n_live_points=200, verbose=True)

    a = pymultinest.Analyzer(n_params=4) 

    # gets the marginalized posterior probability distributions 
    s = a.get_stats()
    objects[1]['P'] = s['marginals'][0]['median']
    objects[2]['m'] = s['marginals'][2]['median']
    objects[2]['P'] = s['marginals'][3]['median']
    
    return objects, a.get_data(), s

def shift_align(ttv_data,ttv,xx,yerr):
    chis = []
    # only works for 1 planet 
    for i in range(len(ttv)-len(xx)):
        chi2 = ((ttv_data-np.roll(ttv,-i)[xx])/yerr)**2 
        chis.append( -np.sum(chi2) )
    i = np.argmax(chis)
    return np.roll(ttv, -i)

# check return 
# plt.plot(xx,yy,'ko');plt.plot(epochs[xx],ttv[xx],'r-'); plt.plot(epochs[xx],np.roll(ttv,-2)[xx],'g-'); plt.show()
# import pdb; pdb.set_trace() 

if __name__ == "__main__":
    
    # units: Msun, Days, au
    objects = [
        {'m':1.12},
        {'m':0.28*mjup/msun, 'P':3.2888, 'inc':3.14159/2,'e':0 }, 
        {'m':1*mjup/msun, 'P':7.5, 'inc':3.14159/2,'e':0 }, 
    ]

    # create REBOUND simulation
    sim = generate(objects)

    # year long integrations, timestep = 1 hour
    sim_data = integrate(sim, objects, 60, 60*24) 
    
    # collect the analytics of interest from the simulation
    ttv_data = analyze(sim_data)

    # TODO 
    # report(sim_data)

    # simulate some observational data with noise 
    ttv = ttv_data['planets'][0]['ttv']
    epochs = np.arange(len(ttv))
    ocdata = ttv + np.random.normal(1,0.5,len(ttv))/(24*60)
    ttdata = ocdata + epochs*ttv_data['planets'][0]['P'] 
    err = np.random.normal(120,30,len(ttv))/(24*60*60)
    
    # perform nested sampling linear fit to transit data    
    lstats, lposteriors = lfit(epochs,ttdata,err, 
                bounds=[ttv_data['planets'][0]['P']-6/24,ttv_data['planets'][0]['P']+6/24, min(ttdata)-1/24,ttv_data['planets'][0]['P']+1/24])
    print("linear fit complete ")
    import pdb; pdb.set_trace()

    # estimate priors
    bounds = []
    objects[1]['P'] = lstats['marginals'][0]['median']
    objects[2] = {'e': 0, 'inc': 1.570795} # delete mass and period for fitting routine  
    bounds = [
        lstats['marginals'][0]['5sigma'][0], lstats['marginals'][0]['5sigma'][1], # Period #1 (day)
        lstats['marginals'][1]['5sigma'][0], lstats['marginals'][0]['5sigma'][1], # t0 #1 
        objects[1]['m'] * 0.25, objects[1]['m'] * 5, # Mass #2 (msun)
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
    f = corner.corner(posteriors[mask,2:], 
                    labels=['Period (day)','t0','Mass 2 (Jup)', 'Period 2 (day)'],
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


    # generate models between the uncertainties 
    f,ax = plt.subplots(1, figsize=(7,4))
    ax.errorbar(epochs,ocdata*24*60,yerr=err*24*60,ls='none',marker='o',label='Data',color='black')
    ax.plot(epochs, ttv_data['planets'][0]['ttv']*24*60,ls='--', label='Truth',color='green')
    ax.plot(epoch, ttvfit*24*60, label='Linear+Nbody ({:.1f})'.format(stats['global evidence']),color='red')
    ax.fill_between(epoch,24*60*upper3,24*60*lower3,alpha=0.1,label='Nbody 3 sigma',color='red')
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
