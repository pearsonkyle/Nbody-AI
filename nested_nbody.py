import matplotlib.pyplot as plt
import numpy as np
import pickle 
import corner 

import os
import pymultinest
if not os.path.exists("chains"): os.mkdir("chains")

from nbody.simulation import randomize, generate, integrate, analyze, report
from nbody.tools import mjup,msun,mearth,G,au,rearth,sa

def nested_nbody( xx,yy,yerr, objects, bounds=[-10.,10.,-10.,10.], myloss='soft_l1'):

    # compute integrations for as long as the max epoch 
    ndays = np.round( 2*(max(xx)+1)*objects[1]['P'] ).astype(int)

    # prevents seg fault in MultiNest
    yerr[yerr==0] = 1
    
    def model_sim(params):
        # follow same order as bounds 
        objects[2]['m'] = params[0]
        objects[2]['P'] = params[1]
        objects[2]['e'] = params[2]
        #objects[2]['omega'] = params[3]
        
        # create REBOUND simulation
        sim = generate(objects)
        sim_data = integrate(sim, objects, ndays, ndays*24) # year long integration, dt=30 min
        return analyze(sim_data, ttvfast=True)

    def myprior(cube, ndim, n_params):
        '''This transforms a unit cube into the dimensions of your prior space.'''
        cube[0] = (bounds[1] - bounds[0])*cube[0]+bounds[0]
        cube[1] = (bounds[3] - bounds[2])*cube[1]+bounds[2]
        cube[2] = (bounds[5] - bounds[4])*cube[2]+bounds[4]
        #cube[3] = (bounds[7] - bounds[6])*cube[3]+bounds[6]

    loss = {
        'linear': lambda z: z,
        'soft_l1' : lambda z : 2 * ((1 + z)**0.5 - 1),
    }

    def myloglike(cube, ndim, n_params):
        epochs,ttv0 = model_sim(cube)
        try:
            chis = []

            # only works for 1 planet 
            # fudge argument of periastron with shift of transit epochs 
            for i in range(len(epochs)-len(xx)):
                chi2 = ((yy-np.roll(ttv0,-i)[xx])/yerr)**2 
                chis.append( -np.sum( loss[myloss](chi2) ) )
            # is it possible to do a linear interpolation instead of roll? 


            # check return 
            # plt.plot(xx,yy,'ko');plt.plot(epochs[xx],ttv0[xx],'r-'); plt.plot(epochs[xx],np.roll(ttv0,-2)[xx],'g-'); plt.show()
            return np.max(chis)
        except:
            print('error with model, -999')
            import pdb; pdb.set_trace()
            return -999 

    pymultinest.run(myloglike, myprior, int(len(bounds)/2), resume=False, evidence_tolerance=0.1, n_live_points=200, verbose=True)

    # lets analyse the results
    a = pymultinest.Analyzer(n_params=4) #retrieves the data that has been written to hard drive
    s = a.get_stats()
    values = s['marginals'] # gets the marginalized posterior probability distributions 
    posteriors = a.get_data()[:,2:]

    print('in nested')
    import pdb; pdb.set_trace() 
    
    # TODO return a new data object 
    return a,posteriors

if __name__ == "__main__":
    
    # units: Msun, Days, au
    objects = [
        {'m':1.12},
        {'m':0.28*mjup/msun, 'P':3.2888, 'inc':3.14159/2,'e':0, 'omega':0  }, 
        {'m':1*mjup/msun, 'P':7.5, 'inc':3.14159/2,'e':0,  'omega':np.pi/4  }, 
    ]

    # create REBOUND simulation
    sim = generate(objects)

    # year long integrations, timestep = 1 hour
    sim_data = integrate(sim, objects, 60, 60*24) 
    
    # collect the analytics of interest from the simulation
    ttv_data = analyze(sim_data)

    # simulate some observational data with noise 
    ttv = ttv_data['planets'][0]['ttv']
    ttv += np.random.normal(0.25,0.25,len(ttv))/(24*60)
    epochs = np.arange(len(ttv))
    err = np.random.normal(30,5,len(ttv))/(24*60*60)

    # estimate priors
    bounds = [5*mearth/msun, 2*mjup/msun, 6,9, 0.0,0.1]

    # TODO newobj create, create plotting routine for final solution and posters
    newobj, posteriors = nested_nbody( epochs,ttv,err, objects, bounds )

    f = corner.corner(posteriors, labels=['mass','per','ecc'],bins=int(np.sqrt(posteriors.shape[0])), plot_contours=False, plot_density=False)
    plt.show() 

    dude() 


    sim = generate(objects)
    sim_data = integrate(sim, objects, 180, 180*24) # year long integration, dt=1 hour
    plt.errorbar(epochs,lcdata['ttv'],yerr=err,ls='none')
    epochs,ttv = analyze(sim_data, ttvfast=True)
    plt.plot(epochs,ttv,'r-')
    plt.show()

    data = a.get_data()
    plt.scatter( posteriors[:,0]*msun/mearth, posteriors[:,1],c=np.log2(data[:,1]),vmin=2.5,vmax=3.25 )
    cbar = plt.colorbar()
    cbar.set_label('-Log Likelihood')
    plt.xlabel('Mass (Earth)')
    plt.ylabel('Period (day)')
    plt.title('WASP-126 c?')
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
