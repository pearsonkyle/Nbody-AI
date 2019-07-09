# to run in background: 
# nohup python -u generate_simulations.py > log.txt &

import copy
import pickle 
import argparse
import numpy as np
from nbody.simulation import randomize, generate, analyze, report
from nbody.tools import msun, mearth, mjup

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    help_ = "Name of pickle file to save simulations to"
    parser.add_argument("-f", "--file", help=help_, default="ttv.pkl")
    help_ = "Number of simulations"
    parser.add_argument("-s", "--samples", help=help_, default=1000, type=int)
    help_ = "Number of periastron arguments per simulation"
    parser.add_argument("-o", "--omegas", help=help_, default=10, type=int)
    help_ = "Number of planet 1 orbital periods to integrate the simulation for"
    parser.add_argument("-p", "--periods", help=help_, default=30 ,type=int)
    
    args = parser.parse_args()

    try:
        X,y = pickle.load(open(args.file,'rb'))
    except:
        X,y = [],[]

    # seed simulation 
    og_objects = randomize()
    sim_data = generate(og_objects, 
        Ndays=og_objects[1]['P']*args.periods+1, 
        Noutputs=round(og_objects[1]['P']*args.periods*24+24) # dt = 1 hr
    ) 
    ttv_data = analyze(sim_data)
    # report(ttv_data)

    for ii in range(len(X), args.samples):

        print('simulation:',ii)

        # randomize objec ts
        og_objects = randomize()
        sim_data = generate(og_objects, 
            Ndays=og_objects[1]['P']*args.periods+1, 
            Noutputs=round(og_objects[1]['P']*args.periods*24+24) # dt = 1 hr
        ) 
        ttv_data = analyze(sim_data)
        
        # re-sample if bad
        while ttv_data['planets'][0]['ttv'].shape[0] < 30  or \
              ttv_data['planets'][0]['max']*24*60 > 10 or \
              ttv_data['planets'][0]['max']*24*60 < 0.5:
            
            # randomize objec ts
            og_objects = randomize()
            sim_data = generate(og_objects, 
                Ndays=og_objects[1]['P']*args.periods+1, 
                Noutputs=round(og_objects[1]['P']*args.periods*24+24) # dt = 1 hr
            ) 
            ttv_data = analyze(sim_data)

        # loop through omega values for each simulation 
        for j in np.linspace(-np.pi,np.pi,args.omegas) + np.random.normal(0,np.pi/args.omegas,args.omegas): 
            
            objects = copy.deepcopy(og_objects)
            objects[2]['omega'] = j 
            sim_data = generate(objects, 
                Ndays=objects[1]['P']*args.periods+1, 
                Noutputs=round(objects[1]['P']*args.periods*24+24) # dt = 1 hr
            ) 

            ttv_data = analyze(sim_data)

            xn = [] 
            xn.append( ttv_data['mstar'] )
            xn.append( ttv_data['planets'][0]['P'] )
            xn.append( ttv_data['planets'][0]['mass']*msun/mearth )
            xn.append( ttv_data['planets'][1]['P'] )
            xn.append( ttv_data['planets'][1]['mass']*msun/mearth )
            xn.append( ttv_data['objects'][2]['omega'] )
            xn.append( ttv_data['planets'][1]['e'] )

            X.append( xn )
            y.append( (ttv_data['planets'][0]['ttv']*24*60)[:args.periods] )

            if ii % 100 == 0:
                pickle.dump([X,y],open(args.file,'wb') )