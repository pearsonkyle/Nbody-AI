import pandas 
import numpy as np
import matplotlib.pyplot as plt

from nbody.simulation import generate, integrate, analyze, report
from nbody.tools import mjup,msun,mearth

# download data from NASA exoplanet archive 
def get_data(keys=['pl_name','pl_hostname','pl_pnum','pl_tranflag','pl_orbper','pl_bmassj','pl_orbeccen','st_mass','st_optmag','pl_def_refname','pl_disc_refname','pl_ttvflag']):
    # query the api for the newest data
    api_url = 'https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=exoplanets&format=json&select='
    query_url = api_url+','.join(keys)
    return pandas.read_json( query_url )


if __name__ == "__main__":
    
    data = get_data()

    # remove if data is null
    nulls = ~data.isnull().any(axis=1)
    data = data[nulls]

    # only deal with multiplanet systems 
    pmask = (data['pl_pnum'].values>1) & (data['pl_tranflag']==1) # check to see if companions are non-transiting 
    data = data[pmask]
    data['ttv'] = 0.0
    
    for name in np.unique(data['pl_hostname']):

        #ni = (data['pl_hostname'] == name ).index[0]
        #import pdb; pdb.set_trace() 


        # extract the first two planets in the system 
        nmask = data['pl_name'] == name
        bmask = data['pl_name'] == name+' b'
        cmask = data['pl_name'] == name+' c'

        if bmask.sum() > 0 and cmask.sum()>0:
            pass
        else:
            continue 

        objects = [
            {'m':float(data[bmask]['st_mass'])},
            {'m':float(data[bmask]['pl_bmassj'])*mjup/msun, 'P':float(data[bmask]['pl_orbper']), 'inc':3.14159/2, 'e':float(data[bmask]['pl_orbeccen']), 'omega':0  }, 
            {'m':float(data[cmask]['pl_bmassj'])*mjup/msun, 'P':float(data[cmask]['pl_orbper']), 'inc':3.14159/2, 'e':float(data[cmask]['pl_orbeccen']), 'omega':0  }, 
        ]

        if objects[1]['P'] < 30 and objects[2]['P'] < 90:
            # create REBOUND simulation
            sim = generate(objects)

            print(objects)
            # year long integrations, timestep = 1 hour
            sim_data = integrate(sim, objects, 20*objects[1]['P'], int(20*objects[1]['P']*24) ) 

            # collect the analytics from the simulation
            ttv_data = analyze(sim_data)

            # save data 
            data.at[data[bmask].index[0],'ttv'] = np.max(np.abs( ttv_data['planets'][0]['ttv'] )*24*60)
            data.at[data[cmask].index[0],'ttv'] = np.max(np.abs( ttv_data['planets'][1]['ttv'] )*24*60)

    print(data[data.ttv>1].sort_values('ttv',ascending=False).get(['pl_name','pl_orbper','pl_orbeccen','st_mass','st_optmag','ttv','pl_disc_refname','pl_def_refname','pl_ttvflag']))
    