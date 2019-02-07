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
    # TODO see if some values can be set to default
    nulls = ~data.isnull().any(axis=1)
    data = data[nulls]

    # only deal with multiplanet systems 
    pmask = (data['pl_pnum'].values>1) & (data['pl_orbper']<100)
    data = data[pmask]

    data = data.reset_index(drop=True)

    #initialize new column later
    data['ttv'] = 0.0
    
    for name in np.unique(data['pl_hostname']):

        # find all planets with this host name
        pmask = data.pl_hostname.values==name
        
        if pmask.sum() >= 2:
            objects = [
                { 'm':float(data[pmask]['st_mass'].iloc[0]) }
            ]

            # loop over planets and add to simulation
            for i in range(pmask.sum()):
                objects.append( {
                    'm':float(data[pmask]['pl_bmassj'].iloc[i])*mjup/msun, 
                    'P':float(data[pmask]['pl_orbper'].iloc[i]), 
                    'inc':3.14159/2, 
                    'e':float(data[pmask]['pl_orbeccen'].iloc[i]), 
                    'omega':0  
                } )
             

            # create REBOUND simulation
            sim = generate(objects)

            print(objects)
            # year long integrations, timestep = 1 hour
            sim_data = integrate(sim, objects, 20*objects[-1]['P'], int(20*objects[-1]['P']*24) ) 

            # collect the analytics from the simulation
            ttv_data = analyze(sim_data)

            #if name=='Kepler-30':
            #    import pdb; pdb.set_trace()

            # save simulation data
            for i in range(pmask.sum()):
                iname = data[pmask].iloc[i].name
                data.at[iname,'ttv'] = np.percentile(np.abs( ttv_data['planets'][i]['ttv'] )*24*60,90)

    mask = (data.ttv>1) & (data.pl_ttvflag==1) 
    print(data[mask].sort_values('ttv',ascending=False).get(['pl_name','pl_orbper','pl_orbeccen','st_mass','st_optmag','ttv','pl_disc_refname','pl_def_refname','pl_ttvflag']))
    