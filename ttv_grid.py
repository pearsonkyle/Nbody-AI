from nbody.simulation import generate, analyze, report
from nbody.tools import mjup,msun,mearth,mjup

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import cm

if __name__ == "__main__":
    
    # units: Msun, Days, au
    objects = [
        {'m':1.1933},
        {'m':0.43*mjup/msun, 'P':3.875, 'inc':3.14159/2, 'e':0, 'omega':0  }, 
        {'m':mearth/msun, 'P':7.2, 'inc':3.14159/2, 'e':0,  'omega':0  }, 
    ]
    # median planet value from TESS
    #In [10]: m['1d'].predict( np.array([[9.06573*rearth/rjup]]) )
    #Out[10]: (array([[-0.36545262]]), array([[0.17203385]]))


    # create REBOUND simulation
    # year long integrations, timestep = 1 hour
    #sim_data = generate(objects, 180, 180*24)

    # collect the analytics of interest from the simulation
    #ttv_data = analyze(sim_data)

    # plot the results 
    #report(ttv_data)

    periods = np.linspace(1.25*objects[1]['P'], 2.2*objects[1]['P'], 50 )
    #masses = np.linspace( objects[1]['m']*0.25, objects[1]['m']*6, 100 ) 
    masses = np.linspace( mearth/msun, 20*mearth/msun, 50 ) 
    
    ttvs = np.zeros((periods.shape[0],masses.shape[0]))

    for i in range(periods.shape[0]):
        print(i)
        for j in range(masses.shape[0]):
            objects[2]['P'] = periods[i]
            objects[2]['m'] = masses[j]

            sim_data = generate(objects, 180, 180*24)

            ttv_data = analyze(sim_data)

            ttvs[i,j] = ttv_data['planets'][0]['max']*24*60


    top = cm.get_cmap('viridis', 200)
    newcolors = top(np.linspace(0., 1, 200))
    newcolors[int(200* 2./30):int(200* 5./30), 3] = np.linspace(0,1,int(200* 5./30)-int(200* 2./30) ) # faded 1-3
    newcolors[:int(200* 2./30), 3] = 0 # white under 1 minutes
    newcmp = ListedColormap(newcolors)


    plt.imshow(ttvs, vmin=0, vmax=30, cmap=newcmp, origin='lower',
                extent=[ min(masses)*msun/mearth, max(masses)*msun/mearth, min(periods)/objects[1]['P'],max(periods)/objects[1]['P']],
                aspect=20 )

    plt.ylabel(r'Period Ratio [P$_{outer}$/P$_{inner}$]')
    plt.xlabel('Mass of Outer Planet [Earth]')
    plt.title("TTV Estimates for an Inner Planet")

    plt.ylim([1.25,2.2])
    cbar = plt.colorbar()
    cbar.set_label('Max TTV [min]', rotation=270, labelpad=10)
    plt.savefig('ttv_grid.pdf',bbox_inches='tight')
    plt.close()
