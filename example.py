from nbody.simulation import generate, integrate, analyze, report
from nbody.tools import mjup,msun,mearth

if __name__ == "__main__":
    
    # units: Msun, Days, au
    objects = [
        {'m':1.12},
        {'m':0.28*mjup/msun, 'P':3.2888, 'inc':3.14159/2,'e':0, 'omega':0  }, 
        {'m':0.988*mjup/msun, 'P':7, 'inc':3.14159/2,'e':0,  'omega':0  }, 
        {'m':0.432*mjup/msun, 'P':10, 'inc':3.14159/2,'e':0,  'omega':0  }, 
        {'m':0.256*mjup/msun, 'P':15, 'inc':3.14159/2,'e':0,  'omega':0  }, 
    ]

    # create REBOUND simulation
    sim = generate(objects)

    # year long integrations, timestep = 1 hour
    sim_data = integrate(sim, objects, 365, 365*24*2) 
    
    # collect the analytics of interest from the simulation
    ttv_data = analyze(sim_data)

    # plot the results 
    report(ttv_data)#, savefile='report.png')