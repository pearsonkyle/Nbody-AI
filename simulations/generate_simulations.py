# to run in background: 
# nohup python -u generate_simulations.py > log.txt &
import copy
import pickle 
import argparse
import numpy as np
from tqdm import tqdm
import sys
sys.path.append('../')
from nbody.simulation import randomize, generate, analyze, report, transit_times
from nbody.tools import msun, mearth, mjup

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    help_ = "Name of pickle file to save simulations to"
    parser.add_argument("-f", "--file", help=help_, default="ttv.pkl")
    help_ = "Number of simulations"
    parser.add_argument("-s", "--samples", help=help_, default=100000, type=int)
    help_ = "Number of periastron arguments per simulation"
    parser.add_argument("-o", "--omegas", help=help_, default=6, type=int)
    help_ = "Number of planet 1 orbital periods to integrate the simulation for"
    parser.add_argument("-p", "--periods", help=help_, default=1000 ,type=int)

    args = parser.parse_args()

    try:
        X,y = pickle.load(open(args.file,'rb'))
    except:
        X,y = [],[]

    # seed simulation 
    og_objects = randomize()
    sim_data = generate(og_objects, 
        Ndays=og_objects[1]['P']*args.periods, 
        Noutputs=round(og_objects[1]['P']*args.periods*24) # should be atleast 20 steps per period
    ) 
    ttv_data = analyze(sim_data)
    # report(ttv_data)

    for ii in tqdm(range(len(X), args.samples)):

        print('simulation:',ii)

        # randomize objects
        og_objects = randomize()
        sim_data = generate(og_objects, 
            Ndays=og_objects[1]['P']*args.periods, 
            Noutputs=round(og_objects[1]['P']*args.periods*24) # dt ~= hour
        )
        ttv_data = analyze(sim_data)

        # re-sample if bad
        while len(ttv_data['planets'][0]['ttv']) < args.periods  or \
              ttv_data['planets'][0]['max']*24*60 > 180:

            # randomize objec ts
            og_objects = randomize()
            sim_data = generate(og_objects, 
                Ndays=og_objects[1]['P']*args.periods, 
                Noutputs=round(og_objects[1]['P']*args.periods*24)
            ) 
            ttv_data = analyze(sim_data)

        #ttv_data = analyze(sim_data)
        Tc = transit_times( sim_data['pdata'][0]['x'], sim_data['star']['x'], sim_data['times'] ) # in days

        xn = [
            og_objects[0]['m'],
            og_objects[1]['P'],
            og_objects[1]['m']*msun/mearth,
            og_objects[2]['P'],
            og_objects[2]['m']*msun/mearth,
            og_objects[2]['omega'],
            og_objects[2]['e']
        ]

        X.append( xn )
        # save transit times for each simulation
        y.append( Tc[:args.periods] )

        # loop through omega values for each simulation 
        for j in np.linspace(-np.pi,np.pi,args.omegas) + np.random.normal(0,np.pi/args.omegas,args.omegas): 

            objects = copy.deepcopy(og_objects)
            objects[2]['omega'] = j 
            sim_data = generate(objects, 
                Ndays=objects[1]['P']*args.periods, 
                Noutputs=round(objects[1]['P']*args.periods*24)
            ) 

            #ttv_data = analyze(sim_data)
            Tc = transit_times( sim_data['pdata'][0]['x'], sim_data['star']['x'], sim_data['times'] ) # in days

            xn = [
                objects[0]['m'],
                objects[1]['P'],
                objects[1]['m']*msun/mearth,
                objects[2]['P'],
                objects[2]['m']*msun/mearth,
                objects[2]['omega'],
                objects[2]['e']
            ]

            X.append( xn )
            # save transit times for each simulation
            y.append( Tc[:args.periods] )

            if ii % 100 == 0:
                pickle.dump([X,y],open(args.file,'wb') )

# combine simulations from mentor3/4, sna-dgx-n01, tethys2 into ttv.pkl
# does period ratio matter the most? along with mass and distance?

# import pickle
# import glob
# pkl_files = glob.glob('*.pkl')
# # combine all pickle files into one
# X,y = [],[]
# for pkl_file in pkl_files:
#     X_,y_ = pickle.load(open(pkl_file,'rb'))
#     X.extend(X_)
#     y.extend(y_)
# print(len(X),len(y))
# pickle.dump([X,y],open('ttv.pkl','wb') )