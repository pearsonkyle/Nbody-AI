# to run in background: 
# nohup python -u generate_simulations.py > log.txt &

import pickle 
import argparse
import numpy as np
from nbody.simulation import randomize, generate, analyze, report

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    help_ = "Name of pickle file to save simulations to"
    parser.add_argument("-f", "--file", help=help_, default="ttv.pkl")
    help_ = "Number of simulations"
    parser.add_argument("-s", "--samples", help=help_, default=1000, type=int)
    help_ = "Number of periastron arguments per simulation"
    parser.add_argument("-o", "--omegas", help=help_, default=6, type=int)
    help_ = "Number of planet 1 orbital periods to integrate the simulation for"
    parser.add_argument("-p", "--periods", help=help_, default=30 ,type=int)
    
    args = parser.parse_args()

    data = []
    for ii in range(args.samples):
        print('simulation:',ii)

        # loop through omega values for each simulation 
        for j in np.linspace(0,1.5*np.pi,args.omegas) + np.random.normal(0,0.25,args.omegas): 
        
            objects = randomize()
            objects[2]['omega'] = j 
            sim_data = generate(objects, 
                Ndays=objects[1]['P']*args.periods, 
                Noutputs=round(objects[1]['P']*args.periods*24) # dt = 1 hr
            ) 

            ttv_data = analyze(sim_data)

            data.append(ttv_data)

            if ii % 100 == 0:
                pickle.dump(data,open(args.file,'wb') )
                #report(ttv_data, savefile='report_{}.png'.format(ii))
