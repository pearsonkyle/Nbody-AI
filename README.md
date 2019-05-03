# N-Body TTV Retrieval
A python 3 package for generating N-body simulations, computing transit timing variations (TTV) and retrieving orbit parameters and uncertainties from TTV measurements within a Bayesian framework. Machine learning is used to estimate the orbit parameters and constrain priors before running a retrieval to model orbital perturbations. 

## Background
Transiting exoplanets in multiplanet systems exhibit non-Keplerian orbits as a result of the graviational influence from companions which can cause the times and durations of transits to vary (TTV/TDV). The amplitude and periodicity of transit time variations are characteristic of the perturbing planet's mass and orbit. Astronomers back in the day were able to measure orbital perturbations of Uranus in order to indirectly discover Neptune before it was seen through a telescope. A similar analysis can find additional exoplanets that  

![](figures/exoplanet_ttv.gif)

Video credit: [NASA](https://www.youtube.com/watch?v=rqQ1xKsNIQE)

One of the primary objectives for the [TESS mission](https://tess.mit.edu/) is to measure the masses of at least 50 planets with radii less than 4 R_Earth, which typically requires radial velocity measurements. The average radial velocity semi-amplitude for a 1-10 M_Earth planet at an orbital period of 7 days around a Sun-like star is ~1-5 m/s. Earth-sized planets push the limits of current radial velocity instruments making it difficult to measure their mass. However, a planet's mass can also be measured with transit timing variations in photometric transit data.

## Generate an N-body simulation 
The n-body simulations in this research make use of the [REBOUND](https://rebound.readthedocs.io) code. To generate a random simulation follow the code below: 
```python
from nbody.simulation import generate, analyze, report
from nbody.tools import mjup,msun,mearth

if __name__ == "__main__":
    
    # units: Msun, Days, au
    objects = [
        {'m':1.12},
        {'m':0.28*mjup/msun, 'P':3.2888, 'inc':3.14159/2, 'e':0, 'omega':0  }, 
        {'m':0.988*mjup/msun, 'P':7, 'inc':3.14159/2, 'e':0,  'omega':0  }, 
        {'m':0.432*mjup/msun, 'P':12, 'inc':3.14159/2, 'e':0,  'omega':0  }, 
    ]

    # create REBOUND simulation
    # year long integrations, timestep = 1 hour
    sim_data = generate(objects, 365, 365*24)

    # collect the analytics of interest from the simulation
    ttv_data = analyze(sim_data)

    # plot the results 
    report(ttv_data, savefile='report.png')
```
A simulation report looks like this: 
![](figures/report_simulation.png)
**Top Left** plots of the orbit positions for each object. **Top Middle** Radial velocity semi-amplitude (m/s) for the star. **Top Right** Periodogram of RV semi-amplitude signal. **Bottom Left** Table of simulation parameters. **Bottom Middle** The difference between the observed transit time and calculated linear ephemeris (O-C). **Bottom Right** Periodogram of O-C signal for each planet. 

The data product for each simulation returned by `analyze(...)` will look like: 
```python
ttv_data = {
    'mstar': float,     # mass of star (msun)
    'objects': dict,    # dictionary passed to generate_simulation(...) method 
    'times': ndarray,   # array of each time in simulation 

    'RV':{
        'signal': ndarray, # RV semi-amplitude signal as a function of time (m/s)
        'max': float        # average between maximum and 75% percentile of RV semi-amplitude (m/s)

        'freq': ndarray,    # frequencys in periodogram (1./day)
        'power': ndarray,   # periodogram power from Lomb Scargle routine
    },

    'planets':[  # list of planet parameter dictionaries 
        {
            # orbit parameters
            'm': float,    # mass of planet in msun
            'a': float,    # average semi-major axis of simulation orbits (au)
            'e': float,    # average eccentricity of simulation orbits
            'inc': float,  # average inclination from simulation orbits
            'x': ndarray,  # downsampled orbit positions (1/10 resolution, au)
            'y': ndarray,  # downsampled orbit positions (1/10 resolution, au)

            # transit time data 
            'P': float,      # period derived from linear fit to mid transit times (day)
            't0': float,     # 0th epoch transit time from linear fit to mid transit times (day)
            'tt': ndarray,   # mid-transit times (day)
            'ttv': ndarray,  # o-c values (linear ephemeris subtracted from tt) (day)
            'max': float     # average between maximum and 75% percentile of O-C signal (min)

            # periodogram data of o-c signal 
            'freq': ndarray,    # frequencys in periodogram (1./epoch)
            'power': ndarray,   # periodogram power from Lomb Scargle routine
        }
    ]
}
```

## Estimating Planet Orbits
The presence of additional planets or even moons in an exoplanet system can be inferred by measuring perturbations in the orbit of a transiting exoplanet. The gravitational influence from the campanion, even if it is non-transiting, can perturb the transiting planet in a manner characteristic to the orbit of the perturbing planet. 

![](figures/nested_nbody_fit.png)

See documentation for [N-body retrieval](Retrieval.md). 

## Machine learning
In order to derive robust uncertainties or even compare the significance of a perturbation the Bayesian evidence is used. Retrieving parameters typically takes ~5000 N-body simulations, which can take up to 10 hours or more depending on the size of the prior and how long each N-body integration is. This is where we leverage machine learning in order to estimate our priors and expedite the retrieval. Machine learning is used to estimate the parameters of a new planet based on the O-C data and the known system parameters; M*, M1 and P1. The estimate from a neural network is used to constrain the priors in a parameter retrieval using nested sampling. 

![](figures/nn_prior.png)

See documentation for [machine learning](simulations/)

## Science
The plot below is a time series analysis for the object WASP-18 b using the data from the [Transiting Exoplanet Survey Satellite](https://www.nasa.gov/tess-transiting-exoplanet-survey-satellite/). The top subplot shows a full timeseries of data from sectors 1–2 with the mid transit of each light curve plotted as a green triangle. The bottom left subplot shows a phase folded light curve that has been fit with a transit model to derive the planetary parameters shown in the table on the far right. The green data points are phase folded and binned to a cadence of 2 minutes. The dotted line in the O-C plot represents one sigma uncertainties on the linear ephemeris. The middle right subplot shows a transit periodogram for the PDC Flux and for the residuals of the time series, after each light curve has been removed

![](figures/timeseries_100100827.png)

The residuals of a linear ephemeris are plotted and then compared against a non-linear ephemeris computed from an N-body simulation in the figure below. The values in the legend indicate the Bayesian evidence output from MultiNest. There is a degenerate solution set with two possible modes; a new planet with 45.1 M_Earth, 1.52 day period, 0 eccentricity and a 58.3 M_Earth planet at 1.65 day period with an eccentricity of 0.03. The red shaded region indicates the variability in the TTV signal as a result of the uncertainties in the derived parameters.

![](figures/wasp18_ttv_fit.png)

## Citation 
This work has been submitted for publication in Nature Astronomy and is currently under review. A preprint is available [here](https://www.overleaf.com/read/mfqvfxjbfrwh) and comments from the community are welcome. 

If you use any of these algorithms in your work please include Kyle A. Pearson as a coauthor. Current institution: Lunar and Planetary Laboratory, University of Arizona, 1629 East University Boulevard, Tucson, AZ, 85721, USA