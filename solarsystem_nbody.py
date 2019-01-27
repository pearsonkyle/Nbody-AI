from nbody.simulation import generate, integrate, analyze, report
from nbody.tools import mjup,msun,mearth

if __name__ == "__main__":
    
    # https://nssdc.gsfc.nasa.gov/planetary/factsheet/ 
    objects = [
        {'m':1.},
        {'m': 0.0553*mearth/msun, 'P':87.969,    'inc':3.14159/2, 'e':0.205 },  # Mercury
        {'m': 0.815*mearth/msun,  'P':224.701,   'inc':3.14159/2, 'e':0.007 },  # Venus
        {'m': mearth/msun,        'P':365.24,    'inc':3.14159/2, 'e':0.017 },  # Earth
        {'m': 0.107*mearth/msun,  'P':686.980,   'inc':3.14159/2, 'e':0.094 },  # Mars
        {'m': 317.83*mearth/msun, 'P':4332.589,  'inc':3.14159/2, 'e':0.049 },  # Jupiter
        {'m': 95.16*mearth/msun,  'P':10759.22,  'inc':3.14159/2, 'e':0.056 },  # Saturn
        {'m': 14.54*mearth/msun,  'P':30588.740, 'inc':3.14159/2, 'e':0.0457 }, # Uranus
        {'m': 17.15*mearth/msun,  'P':59799.9,   'inc':3.14159/2, 'e':0.0113 }, # Neptune      
    ]
    # inclination does not have a strong effect on TTVs so ignore 

    # create REBOUND simulation
    sim = generate(objects)

    # integrate the simulation (Ndays, Noutputs) so dt = Ndays/Noutputs 
    sim_data = integrate(sim, objects, 1000*365, 1000*365*24) 
    
    # collect the analytics of interest from the simulation
    ttv_data = analyze(sim_data)

    # plot the results 
    report(ttv_data)#, savefile='report.png')

    # TTV results:
    # Mercury ~ 0-1 min
    # Venus   ~ 3-5 min 
    # Earth   ~ 4-6 min
    # Mars    ~ 20-40 minutes
    # Jupiter ~ 500-1000 minutes
    # Saturn  ~ 1000-1500 minutes

    # Average Amplitude of TTV signal 
    # Average of top 75% +- standard deviation of Top 75% TTV measurements
    # the signal is not perfectly periodic so the amplitude varies over time to best quantify the variability
    # of the peak TTV signal we report it as the average of the top 75% of measurements and standard deviation of top 75%