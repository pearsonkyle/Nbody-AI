from nbody.simulation import generate, analyze, report
from nbody.tools import mjup,msun,mearth
import time

if __name__ == "__main__":
    
    # units: Msun, Days, au
    objects = [
        {'m':0.95}, # stellar mass
        {'m':1.169*mjup/msun, 'P':2.797436, 'inc':3.14159/2, 'e':0, 'omega':0 }, 
        {'m':0.1*mjup/msun, 'P':2.797436*1.9, 'inc':3.14159/2, 'e':0.0,  'omega':0  }, 
    ] # HAT-P-37
    t1 = time.time()

    # create REBOUND simulation
    n_orbits = 1000
    sim_data = generate(objects, objects[1]['P']*n_orbits, int(n_orbits*objects[1]['P']*24) )
    print("Simulation time: {}s".format(time.time()-t1))

    # collect the analytics of interest from the simulation
    ttv_data = analyze(sim_data)

    # plot the results 
    report(ttv_data)

    # # create a fake dataset
    # tmids = 2459150 + ttv_data['planets'][0]['tt']
    # # add random noise to observations
    # tmids += np.random.normal(0,0.5,len(tmids))/(24*60)
    # # randomly select 40 observations without repeat
    # tmids = np.random.choice(tmids,40,replace=False)
    # # add random error to observations between
    # err = 1/24/60 + np.random.random()*0.25/24/60 + np.random.normal(0,0.1,len(tmids))/(24*60)
    

    dude()
    #, savefile='report.png')
    
    import numpy as np
    ttv = ttv_data['planets'][0]['ttv']
    epochs = np.arange(len(ttv))
    ttdata = ttv_data['planets'][0]['tt'] + np.random.normal(0,0.5,len(ttv))/(24*60)
    err = (30+np.random.normal(30,30,len(ttv))) /(24*60*60)
    np.vstack([epochs, ttdata, err]).T
    np.savetxt('sim_data.txt',np.vstack([epochs, ttdata, err]).T, header="{}".format(objects))