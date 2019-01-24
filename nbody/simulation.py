# Kyle A. Pearson (2018)
# Methods to generate nbody simulations of transiting exoplanets 
import pickle 

import numpy as np
import matplotlib.pyplot as plt

import rebound

from nbody.tools import lomb_scargle,TTV,find_zero,hill_sphere,maxavg,sa,m2r_star
from nbody.tools import msun,mjup,mearth,G,au,rearth,rsun

def empty_data(N): # orbit parameters in timeseries
    return {
        'x':np.zeros(N),
        'y':np.zeros(N),
        'z':np.zeros(N),
        'P':np.zeros(N),
        'a':np.zeros(N),
        'e':np.zeros(N),
        'inc':np.zeros(N),
        'Omega':np.zeros(N),
        'omega':np.zeros(N),
        'M':np.zeros(N),
    }

def generate(objects):
    # create rebound simulation 
    # for object parameters see: 
    # https://rebound.readthedocs.io/en/latest/_modules/rebound/particle.html
    sim = rebound.Simulation()
    sim.units = ('day', 'AU', 'Msun')
    for i in range(len(objects)):
        sim.add( **objects[i] ) 
    sim.move_to_com() 
    return sim

def randomize():

    objects = [
        # stellar parameters
        {
            'm': np.random.uniform(0.4,1.5), # star mass [msun]
        },

        # planet 1
        {
            'm': np.random.uniform(0.66*mearth/msun, 100*mearth/msun),
            'P': np.random.uniform(1,10),
            # inc - conditional based on transiting inclination limit 
        },

        # planet 2
        {
            'm': np.random.uniform(0.1,4),  # ratio with planet 1   
            # P - conditional, must be beyond hill radius of planet 1
            # inc - conditional based on transiting inclination limit
        }
    ]

    # set mass of planet 2
    objects[2]['m'] *= objects[1]['m']

    # compute hill sphere for planet 1
    hr1 = hill_sphere(objects,i=1)
    a1 = sa(objects[0]['m'], objects[1]['P'] )
    a2,hr2 = 0,0

    # compute period for planet 2 beyond hill spheres interacting
    while a1+hr1 > a2-hr2:

        # check if hill spheres interact
        objects[2]['P'] = objects[1]['P']*np.random.uniform(1.25,4)
        a2 = sa(objects[0]['m'], objects[2]['P'] )
        hr2 = hill_sphere(objects,i=2)

    # compute (approx) inclination limit for visible transit geometry
    rs = (m2r_star(objects[0]['m'])*rsun-rearth/rsun) /au  # convert to au
    inc_lim1 = np.arctan(rs/a1)
    inc_lim2 = np.arctan(rs/a2)

    # chance for planet 1 to be non-transiting 
    objects[1]['inc'] = np.random.uniform(np.pi/2-inc_lim1*2, np.pi/2)

    # planet 1 is not transiting
    if objects[1]['inc'] < (np.pi/2-inc_lim1):

        # make sure planet 2 is transiting
        objects[2]['inc'] = np.random.uniform(np.pi/2-inc_lim2, np.pi/2)

    # planet 1 is transiting 
    else:
        # planet 2 can transit or not 
        objects[2]['inc'] = np.random.uniform(np.pi/2-inc_lim2*2, np.pi/2)

    # TODO potential bias for ML 
    # increase inclination limits? real orbits may subtend larger distribution in inc
    # further planets will have preferentially smaller inclinations

    # limits 
    limits = [
        {},
        {'hr':hr1,'inc_lim':np.pi/2-inc_lim1},
        {'hr':hr2,'inc_lim':np.pi/2-inc_lim2},
    ]

    return objects #, limits

def integrate(sim, objects, Ndays, Noutputs):
    # ps is now an array of pointers and will change as the simulation runs
    ps = sim.particles
    
    times = np.linspace(0., Ndays, Noutputs) # units of days 

    pdata = [empty_data(Noutputs) for i in range(len(objects)-1)]
    star = {'x':np.zeros(Noutputs),'y':np.zeros(Noutputs),'z':np.zeros(Noutputs) }

    # run simulation time steps 
    for i,time in enumerate(times):
        sim.integrate(time)

        # record data 
        for k in star.keys():
            star[k][i] = getattr(ps[0], k)

        for j in range(1,len(objects)):
            for k in pdata[j-1].keys():
                pdata[j-1][k][i] = getattr(ps[j],k)

    sim_data = ({
        'pdata':pdata,
        'star':star,
        'times':times,
        'objects':objects,
        'dt': Noutputs/(365*24*60*60) # conversion factor to get seconds for RV semi-amp
    })

    return sim_data 

def transit_times(xp,xs,times):
    # check for sign change in position difference
    dx = xp-xs 
    tt = []
    for i in range(1,len(dx)):
        if dx[i-1] >= 0 and dx[i] <= 0:
            tt.append( find_zero(times[i-1],dx[i-1], times[i],dx[i]) )
    return np.array(tt)

def analyze(m, ttvfast=False):

    if ttvfast:
        tt = transit_times( m['pdata'][0]['x'], m['star']['x'], m['times'] )
        ttv,per,t0 = TTV(np.arange(len(tt)),tt )
        return np.arange(len(ttv)),ttv

    RV = np.diff(m['star']['x'])*1.496e11*m['dt'] # measurements -> seconds
    
    freq,power,fdata,pp = lomb_scargle( m['times'][1:], RV, npeaks=3)

    data = {
        'times':m['times'],
        'RV':{'freq':freq, 'power':power, 'signal':RV, 'fdata':fdata, 'max':maxavg(RV),'pp':pp },
        'mstar': m['objects'][0]['m'],
        'planets':[],
        'objects':m['objects']
    }

    # parse planet data 
    for j in range(1,len(m['objects'])):
        key = label='Planet {}'.format(j)
        pdata = {}

        # compute transit times 
        tt = transit_times( m['pdata'][j-1]['x'], m['star']['x'], m['times'] )
        ttv,per,t0 = TTV(np.arange(len(tt)),tt )

        # periodogram of o-c                
        freq,power = lomb_scargle( np.arange(len(ttv)),ttv, minfreq=1./180,maxfreq=1./2,npeaks=3)

        # parameterize periodogram of o-c with 3 
        z = np.zeros((3,3))
        fdata = np.r_[fdata, z][:3,:3]

        # save data 
        for k in ['e','inc','a']:
            pdata[k] = np.mean( m['pdata'][j-1][k] )
        pdata['mass'] = m['objects'][j]['m']
        pdata['P'] = per
        pdata['t0'] = t0
        pdata['tt'] = tt
        pdata['ttv'] = ttv
        pdata['max'] = maxavg(ttv)
        pdata['freq'] = freq
        pdata['power'] = power
        
        pdata['pp'] = pp
        pdata['x'] = m['pdata'][j-1]['x'][::10]
        pdata['y'] = m['pdata'][j-1]['z'][::10]
        data['planets'].append(pdata)

    return data

def report(data, savefile=None):

    # set up simulation summary report 
    f = plt.figure( figsize=(16,8) ) 
    plt.subplots_adjust()
    ax = [ plt.subplot2grid( (2,3), (0,0) ), # x,y plot
            plt.subplot2grid( (2,3), (1,0) ), # table data 
            plt.subplot2grid( (2,3), (0,1) ), # # RV semi-amplitude plot  #include theoretical limit for 2body system
            plt.subplot2grid( (2,3), (1,1) ), # O-C plot # TODO add second axis for time (day)
            plt.subplot2grid( (2,3), (0,2) ), # lomb scargle for RV semi-amplitude
            plt.subplot2grid( (2,3), (1,2) ) # lomb scargle for o-c
        ]
    plt.subplots_adjust(top=0.96, bottom=0.07, left=0.10, right=0.96, hspace=0.27, wspace=0.27)

    ax[2].plot(data['times'][1:],data['RV']['signal'],'k-' )
    ax[2].set_xlim([0, 4*data['planets'][-1]['P'] ])
    ax[2].set_ylabel('RV semi-amplitude (m/s)')
    ax[2].set_xlabel('time (day)')

    ax[4].plot(1./data['RV']['freq'],data['RV']['power'] )
    # plot reconstructed signal from sin series 

    ax[4].set_xlabel('Period (day)')
    ax[4].set_ylabel('|RV semi-amplitude| Power')
    ax[4].set_xlim([0, 1.5*data['planets'][-1]['P'] ])
    #u = (m['objects'][1]['m']/m['objects'][0]['m']) * np.sqrt( G*(m['objects'][0]['m']+m['objects'][1]['m'])/m['objects'][1]['a']  )*1.496e11/(24*60*60)
    #print('expected RV:',u)

    # create table stuff 
    keys = ['mass','a','P','inc','e','max']
    units = [msun/mearth,1,1,1,1,24*60]
    rounds = [2,3,2,2,3,1,1]

    # make 2d list for table 
    tdata=[]
    for i in range(len(units)):
        tdata.append( [0 for i in range(len(data['planets']))] )
    
    tinfo = { 
        'mass':'Mass (Mearth)',
        'a':'Semi-major Axis (au)',
        'P':'Period (day)', 
        'inc':'Inclination (deg)',
        'e':'Eccentricity',
        'max':'TTV Max Avg (min)',
        # TODO add a transiting flag b.c hard to see from plots if each planet transits
    }
    row_labels = [ tinfo[k] for k in keys ]
    col_labels = [ "Planet {}".format(i+1) for i in range(len(data['planets']))]

    # for each planet in the system 
    for j in range(1,len(data['objects'])):
        
        # plot orbits
        ax[0].plot( data['planets'][j-1]['x'], data['planets'][j-1]['y'],label='Planet {}'.format(j),lw=0.5,alpha=0.5 )
        ax[3].plot(data['planets'][j-1]['ttv']*24*60,label='Planet {}'.format(j) )  # convert days to minutes

        ax[5].plot( 1./data['planets'][j-1]['freq'], data['planets'][j-1]['power'], label='Planet {}'.format(j) )

        # populate table data 
        for i,k in enumerate(keys):
            tdata[i][j-1] = np.round( data['planets'][j-1][k] * units[i], rounds[i])

            if k == 'inc':
                tdata[i][j-1] = np.round( np.rad2deg(data['planets'][j-1][k]) * units[i], rounds[i])


    table = ax[1].table(cellText=tdata, 
                        colLabels=col_labels,
                        rowLabels=row_labels,
                        colWidths=[0.2,0.2,0.2],
                        loc='center')

    ax[1].set_title('Stellar mass: {:.2f} Msun'.format(data['mstar']) )
    table.scale(1.5,1.5)
    
    ax[1].axis('tight')
    ax[1].axis('off')

    ax[0].set_ylabel('(au)')
    ax[0].set_xlabel('(au)')
    ax[0].legend(loc='best')

    ax[1].set_ylabel('position (au)')
    ax[1].set_xlabel('time (day)')

    ax[3].set_ylabel('O-C (minutes)')
    ax[3].set_xlabel('transit epoch')
    ax[3].legend(loc='best')
    ax[3].grid(True)
        
    ax[5].set_xlabel('Period (epoch)')
    ax[5].set_ylabel('|O-C| Power' )
    ax[5].set_xlim([1,30])
    ax[5].legend(loc='best')

    if savefile:
        plt.savefig(savefile )
        plt.close()
    else:
        plt.show()