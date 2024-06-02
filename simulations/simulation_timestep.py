# create grid of simulations over various masses and period ratios
import copy
import pickle 
import argparse
import numpy as np
from tqdm import tqdm
import sys
sys.path.append('../')
from nbody.simulation import randomize, generate, analyze, report, transit_times
from astropy.constants import M_sun, M_earth, M_jup

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    help_ = "Name of pickle file to save simulations to"
    parser.add_argument("-f", "--file", help=help_, default="ttv.pkl")
    help_ = "Number of planet 1 orbital epochs/periods to integrate the simulation for"
    parser.add_argument("-e", "--epochs", help=help_, default=1000, type=int)

    args = parser.parse_args()

    try:
        # load from file if exists
        X,y = pickle.load(open(args.file,'rb'))
    except:
        X,y = [],[]

    # convert to SI units
    mearth = M_earth.value
    msun = M_sun.value
    mjup = M_jup.value

    # define stellar and planetary parameters
    objects = [
        # stellar parameters
        {
            'm': np.random.uniform(0.75,1.15), # star mass [msun]
        },

        # planet 1
        {
            'm': np.random.uniform(0.66*mearth/msun, 450*mearth/msun),
            'P': np.random.uniform(0.8,20),
            'inc': np.deg2rad(90)
        },

        # planet 2
        {
            'm': np.random.uniform(mearth/msun, 5*mearth/msun),
            # P - conditional, must be beyond hill radius of planet 1
            'omega': np.random.uniform(0,2*np.pi),
            'e': np.abs(np.random.normal(0,0.01)),
            'inc': np.deg2rad(90)
        }
    ]

    # randomize second planet period as ratio between 1.25 - 5 * P1
    objects[2]['P'] = objects[1]['P'] * 2.05 #np.random.uniform(1.25, 5)

    # seed simulation 
    sim_data = generate(objects, 
        Ndays=objects[1]['P']*args.epochs, 
        Noutputs=round(objects[1]['P']*args.epochs*24) # dt ~= 1 hr
    )
    ttv_data = analyze(sim_data)
    report(ttv_data)


    # rerun simulation with smaller time steps
    sim_data = generate(objects, 
        Ndays=objects[1]['P']*args.epochs, 
        Noutputs=round(objects[1]['P']*args.epochs*48) # dt ~ 30 min
    )
    ttv_data = analyze(sim_data)
    report(ttv_data)

    