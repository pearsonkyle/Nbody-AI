from scipy.optimize import brentq as findzero 
from scipy.optimize import least_squares
from scipy.interpolate import interp1d
from astropy.stats import LombScargle
from scipy.signal import find_peaks

import numpy as np

import copy

rsun = 6.955e8 # m
msun = 1.989e30 # kg
mjup = 1.898e27 # kg 
rjup = 7.1492e7 # m
mearth = 5.972e24 # kg
rearth = 6.3781e6 # m
au=1.496e11 # m 
G = 0.00029591220828559104 # day, AU, Msun

# logg[cm/s2], rs[rsun]
stellar_mass = lambda logg,rs: ((rs*6.955e10)**2 * 10**logg / 6.674e-8)/1000./msun
maxavg = lambda x: (np.percentile(np.abs(x),75)+np.max(np.abs(x))  )*0.5

# keplerian semi-major axis (au)
sa = lambda m,P : (G*m*P**2/(4*np.pi**2) )**(1./3) 

# approximate radial velocity semi-amplitude 
rv_semi = lambda M,m,a : (G*m*m/(M*a))**0.5

# rv_semi( 1.22, 90*mearth/msun, sa(1.22,2.155)) * au/(24*60*60)

# assume a relationship between stellar mass and radius from 
# (http://ads.harvard.edu/books/hsaa/toc.html)
mms = [40,18, 6.5,3.2,2.1,1.7,1.3,1.1 ,1,0.93,0.78,0.69,0.47,0.21,0.1]  # M/Msun
rrs = [18,7.4,3.8,2.5,1.7,1.3,1.2,1.05,1,0.93,0.85,0.74,0.63,0.32,0.13] # R/Rsun
m2r_star = interp1d(mms,rrs) # convert solar mass to solar radius

def hill_sphere(objs,i=1):
    '''
    Compute the hill sphere radius between the star and i^th planet
    '''
    objects = copy.deepcopy(objs)

    if 'a' not in objects[i]:
        objects[i]['a'] = ( G*objects[0]['m']*objects[i]['P']**2/(4*np.pi**2) )**(1./3)

    velk = lambda M,r : np.sqrt(G*M/r)
    hill_init = objects[i]['a']*(objects[i]['m']/(3*objects[0]['m']))**(1./3)

    def hill_radius(r):
        a_cent = velk(objects[0]['m'], objects[i]['a'])**2/objects[i]['a']
        a_star = G*objects[0]['m']/(objects[i]['a']+r)**2
        a_planet = G*objects[i]['m']/r**2
        return (a_star+a_planet-a_cent)

    hr = findzero(hill_radius, hill_init*0.5,hill_init*1.5)

    del objects
    return hr

def find_zero(t1,dx1, t2,dx2):
    # find zero with linear interpolation
    m = (dx2-dx1)/(t2-t1)
    t0 = -dx1/m + t1
    return t0

def TTV(epochs, tt):
    N = len(epochs)
    # linear fit to transit times 
    A = np.vstack([np.ones(N), epochs]).T
    try:
        b, m = np.linalg.lstsq(A, tt, rcond=None)[0]
    except:
        b, m = np.linalg.lstsq(A, tt)[0]
    ttv = (tt-m*np.array(epochs)-b)
    return [ttv,m,b]

def make_sin(t, pars, freqs=-1):
    fn = 0 

    # use fixed frequencies for ls fitting of amp and phase
    if isinstance(freqs,list) or isinstance(freqs,np.ndarray):
        if pars.ndim == 1: pars = pars.reshape(-1,2) 
        
        for i in range( pars.shape[0] ):
            fn += pars[i,0]*np.sin(t*2*np.pi*freqs[i] + pars[i,1])

    else:
        # reshape from 1d to 2d
        if pars.ndim == 1: pars = pars.reshape(-1,3)
        
        for i in range( pars.shape[0] ):
            fn += pars[i,1]*np.sin(t*2*np.pi*pars[i,0] + pars[i,2])

    return fn 

def lomb_scargle(t,y,dy=None, minfreq=1./365, maxfreq=1/2, npeaks=0, peaktol=0.05):
    # periodogram
    if isinstance(dy,np.ndarray):
        ls = LombScargle(t,y,dy)
    else:
        ls = LombScargle(t,y)

    frequency,power = ls.autopower(minimum_frequency=minfreq, maximum_frequency=maxfreq, samples_per_peak=10)
    probabilities = [0.1, 0.05, 0.01]
    try:
        pp = ls.false_alarm_level(probabilities)  
    except:
        pp = [-1,-1,-1]
    # power probabilities 
    # This tells us that to attain a 10% false alarm probability requires the highest periodogram peak to be approximately XX; 5% requires XX, and 1% requires XX.
    if npeaks > 0:

        # find peaks in periodogram 
        peaks,amps = find_peaks(power,height=peaktol)
        Nterms = npeaks#min(npeaks,len(peaks))
        fdata = np.zeros( (Nterms,3) ) # frequency, amplitude, shift  

        # fit amplitudes to each peak frequency 
        if Nterms > 0 and npeaks > 0:

            # sort high to low 
            peaks = peaks[ np.argsort(amps['peak_heights'])[::-1] ] 
            
            # estimate priors 
            for i in range(min(npeaks,len(peaks))):
                fdata[i,0] = frequency[ int(peaks[i]) ]
                fdata[i,1] = np.sort(amps['peak_heights'])[::-1][i] * maxavg(y) #amplitude estimate
                fdata[i,2] = 0 # phase shift estimate
        
            # ignore fitting frequencies 
            priors = fdata.flatten()
            bounds = np.array( [ [0,1], [0, max(y)*1.5], [0,2*np.pi] ]*Nterms ).T

            def fit_wave(pars):
                wave = make_sin(t, pars)
                return (y-wave)/y
            
            res = least_squares(fit_wave, x0=priors, bounds=bounds)

            fdata = res.x.reshape(Nterms,-1)

        return frequency,power, fdata    
    else:
        return frequency,power