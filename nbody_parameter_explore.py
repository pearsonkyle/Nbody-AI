import matplotlib.pyplot as plt
import numpy as np
import pickle 
import copy

from nbody.simulation import randomize, generate, integrate, analyze, report
from nbody.tools import mjup,msun,mearth,G,au,rearth,sa

if __name__ == "__main__":

    # generate a random simulation from a uniform distribution
    objects = [
        {'m':1.12},
        {'m':0.28*mjup/msun, 'P':3.2888, 'inc':np.pi/2,'e':0, 'omega':0  }, 
        {'m':1*mjup/msun, 'P':8.25, 'inc':np.pi/2,'e':0,  'omega':0  }, 
    ]

    # compute other simulations with these parameters
    pbounds = [
        {'m':[1,1.5]},
        {
            'm':[0.5*mjup/msun, 1*mjup/msun],
            'P':[1.5,2.25],
            'inc':[np.pi/2*0.8,np.pi/2*0.9],
            'e':[0.025,0.05],
            'omega':[np.pi/4,np.pi/2],
            # add longitude of ascending node
        },
        {
            'm':[2*mjup/msun, 10*mjup/msun],
            'P':[6,9],
            'inc':[np.pi/2*0.8,np.pi/2*0.9],
            'e':[0.05,0.075],
            'omega':[np.pi/4,np.pi/2],
        },
    ]

    # run original simulation 
    sim = generate(objects)
    sim_data = integrate(sim, objects, 180, 180*24) # year long integrations, timestep = 1 minute 
    ttv_data_og = analyze(sim_data)
    
    # create a plot 
    f = plt.figure( figsize=(9,13) ) 
    plt.subplots_adjust()
    ax = [  plt.subplot2grid( (6,2), (0,0) ), 
            plt.subplot2grid( (6,2), (1,0) ), 
            plt.subplot2grid( (6,2), (2,0) ), 
            plt.subplot2grid( (6,2), (3,0) ), 
            plt.subplot2grid( (6,2), (4,0) ), 
            plt.subplot2grid( (6,2), (5,0) ), 

            plt.subplot2grid( (6,2), (1,1) ), 
            plt.subplot2grid( (6,2), (2,1) ), 
            plt.subplot2grid( (6,2), (3,1) ), 
            plt.subplot2grid( (6,2), (4,1) ), 
            plt.subplot2grid( (6,2), (5,1) ), 


            plt.subplot2grid( (6,2), (0,1) ), 
    ]
    plt.subplots_adjust(top=0.96, bottom=0.07, left=0.10, right=0.96, hspace=0.27, wspace=0.27)
    pc = 0 # plot counter

    labels = [
        "M*: {:.2f} (Msun)",
        "Mp 1: {:.1f} (M earth)",
        "P 1: {:.2f} (day)",
        "inc 1: {:.1f} (deg)",
        "ecc 1: {:.3f}",
        "omega 1: {:.1f}",


        "Mp 2: {:.1f} (M earth)",
        "P 2: {:.2f} (day)",
        "inc 2: {:.1f} (deg)",
        "ecc 2: {:.3f}",
        "omega 2: {:.1f}",

    ]

    for i in range( len(pbounds) ): # for each object

        for k in pbounds[i].keys(): # for each parameter
            tobjects = copy.deepcopy(objects)   

            if k == 'inc':
                ax[pc].plot( ttv_data_og['planets'][0]['ttv']*24*60, label=labels[pc].format(np.rad2deg(objects[i][k]) ) )
            elif k[0] == 'm' and i != 0:
                ax[pc].plot( ttv_data_og['planets'][0]['ttv']*24*60, label=labels[pc].format( msun*objects[i][k]/mearth ) )
            else:
                ax[pc].plot( ttv_data_og['planets'][0]['ttv']*24*60, label=labels[pc].format(objects[i][k]) )

            ax[pc].set_xlim([0,30])
            ax[pc].set_xlabel('Transit Epoch')
            ax[pc].set_ylabel('Planet 1 O-C (min)')

            for j in range(len(pbounds[i][k])): # parameter values

                tobjects[i][k] = pbounds[i][k][j]
                
                sim = generate(tobjects)
                sim_data = integrate(sim, tobjects, 180, 180*24) # year long integration, dt=1 hour
                ttv_data = analyze(sim_data)

                if k == 'inc':
                    ax[pc].plot( ttv_data['planets'][0]['ttv']*24*60, label=labels[pc].format(np.rad2deg(tobjects[i][k]) ) )
                elif k[0] == 'm' and i != 0:
                    ax[pc].plot( ttv_data['planets'][0]['ttv']*24*60, label=labels[pc].format( msun*tobjects[i][k]/mearth ) )
                else:
                    ax[pc].plot( ttv_data['planets'][0]['ttv']*24*60, label=labels[pc].format(tobjects[i][k]) )



            ax[pc].legend(loc='best')
            pc += 1


    # populate table 
    table = ax[-1].table(
        cellText= [ [objects[0]['m']], [objects[1]['m']], [2.5], [0], [90], [5], [4], [0], [90] ],
        rowLabels = [ "M* (Msun)",
            "Mp 1 (M earth)",
            "P 1 (day)",
            "ecc 1",
            "inc 1 (deg)",
            
            "Mp 2 (M earth)",
            "P 2 (day)",
            "ecc 2",
            "inc 2 (deg)",
        ],
        colWidths=[0.2,0.2],
        loc='center'
    )

    table.scale(1.5,1.5)
    ax[-1].axis('tight')
    ax[-1].axis('off')

    plt.show()