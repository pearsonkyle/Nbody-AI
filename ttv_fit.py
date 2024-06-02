
import copy
import queue
import pickle
import numpy as np
import matplotlib.pyplot as plt
from ultranest import ReactiveNestedSampler
from nbody.simulation import transit_times
from nbody.simulation import generate, analyze, TTV
from nbody.tools import msun, mearth, mjup
from exotic.api.plotting import corner

class nbody_fitter():

    def __init__(self, data, prior=None, bounds=None, verbose=True):
        self.data = data
        self.bounds = bounds
        self.prior = prior
        self.verbose = verbose
        self.fit_nested()

    def fit_nested(self):

        # set up some arrays for mapping sampler output
        freekeys = []
        boundarray = []
        for i,planet in enumerate(self.bounds):
          for bound in planet:
            freekeys.append(f"{i}_{bound}")
            boundarray.append(planet[bound])

        # find min and max time for simulation
        min_time = np.min(self.data[1]['Tc'])
        max_time = np.max(self.data[1]['Tc'])
        sim_time = max_time-min_time+self.prior[1]['P']*5
        # TODO extend for multiplanet systems
        Tc_norm = self.data[1]['Tc'] - min_time  # normalize the data to the first observation
        self.orbit = np.rint(Tc_norm / self.prior[1]['P']).astype(int)  # number of orbits since first observation (rounded to nearest integer)

        # numpify
        boundarray = np.array(boundarray)
        bounddiff = np.diff(boundarray,1).reshape(-1)

        # create queue and save simulation to 
        self.queue = queue.Queue()

        def loglike(pars):

            # set parameters
            for i,par in enumerate(pars):
                idx,key = freekeys[i].split('_')
                idx = int(idx)
                if key == 'tmid':
                    continue
                # this dict goes to REBOUND and needs specific keys
                self.prior[idx][key] = par

            # run N-body simulation
            sim_data = generate(self.prior, sim_time, int(sim_time*24)) # uses linspace behind the scenes

            # json structure with analytics of interest from the simulation
            # ttv_data = analyze(sim_data) # slow
            chi2 = 0

            sim_shift = 0
            # loop over planets, check for data 
            for i,planet in enumerate(self.prior):
                if self.data[i]:
                    # compute transit times from N-body simulation
                    Tc_sim = transit_times( sim_data['pdata'][i-1]['x'], sim_data['star']['x'], sim_data['times'] )

                    # derive an offset in time from the first planet
                    if i-1==0: 
                        sim_shift = Tc_sim.min()

                    # shift the first mid-transit in simulation to observation
                    Tc_sim -= sim_shift # first orbit has tmid at 0
 
                    # scale Tc_sim to data
                    residual = self.data[i]['Tc'] - Tc_sim[self.orbit]
                    Tc_sim += residual.mean()

                    # add to queue
                    self.queue.put((i,self.prior.copy(),residual.mean(),Tc_sim))

                    # take difference between data and simulation
                    try:
                        chi2 += -0.5*np.sum(((self.data[i]['Tc'] - Tc_sim[self.orbit])/self.data[i]['Tc_err'])**2)
                    except:
                        chi2 += -1e6
                        print(self.prior)
                        # usually unstable orbit 

            return chi2

        def prior_transform(upars):
            return (boundarray[:,0] + bounddiff*upars)

        if self.verbose:
            self.results = ReactiveNestedSampler(freekeys, loglike, prior_transform).run(max_ncalls=4e4)
        else:
            self.results = ReactiveNestedSampler(freekeys, loglike, prior_transform).run(max_ncalls=4e4, show_status=self.verbose, viz_callback=self.verbose)

        self.pars = {}
        self.errors = {}
        self.quantiles = {}
        self.parameters = copy.deepcopy(self.prior)

        for i, key in enumerate(freekeys):
            idx, key = key.split('_')
            self.parameters[int(idx)][key] = self.results['posterior']['median'][i]
            self.pars[key] = self.results['maximum_likelihood']['point'][i] # TODO fix this
            self.errors[key] = self.results['posterior']['stdev'][i]
            self.quantiles[key] = [
                self.results['posterior']['errlo'][i],
                self.results['posterior']['errup'][i]]

    def plot_triangle(self):
        ranges = []
        mask1 = np.ones(len(self.results['weighted_samples']['logl']),dtype=bool)
        mask2 = np.ones(len(self.results['weighted_samples']['logl']),dtype=bool)
        mask3 = np.ones(len(self.results['weighted_samples']['logl']),dtype=bool)
        titles = []
        labels= []
        flabels = {
            'm':'Period [day]',
            'b':'T_mid [JD]',
        }
        for i, key in enumerate(self.quantiles):
            labels.append(flabels.get(key, key))
            titles.append(f"{self.pars[key]:.7f} +-\n {self.errors[key]:.7f}")
            ranges.append([
                self.pars[key] - 5*self.errors[key],
                self.pars[key] + 5*self.errors[key]
            ])

            # mask3 = mask3 & (self.results['weighted_samples']['points'][:,i] > (self.pars[key] - 3*self.errors[key]) ) & \
            #     (self.results['weighted_samples']['points'][:,i] < (self.pars[key] + 3*self.errors[key]) )

            # mask1 = mask1 & (self.results['weighted_samples']['points'][:,i] > (self.pars[key] - self.errors[key]) ) & \
            #     (self.results['weighted_samples']['points'][:,i] < (self.pars[key] + self.errors[key]) )

            # mask2 = mask2 & (self.results['weighted_samples']['points'][:,i] > (self.pars[key] - 2*self.errors[key]) ) & \
            #     (self.results['weighted_samples']['points'][:,i] < (self.pars[key] + 2*self.errors[key]) )


        chi2 = self.results['weighted_samples']['logl']*-2
        fig = corner(self.results['weighted_samples']['points'],
            #labels= labels,
            bins=int(np.sqrt(self.results['samples'].shape[0])),
            #range= ranges,
            figsize=(10,10),
            #quantiles=(0.1, 0.84),
            #plot_contours=True,
            #levels=[ np.percentile(chi2[mask1],95), np.percentile(chi2[mask2],95), np.percentile(chi2[mask3],95)],
            plot_density=False,
            titles=titles,
            data_kwargs={
                'c':chi2,
                'vmin':np.percentile(chi2,1),
                'vmax':np.percentile(chi2,95),
                'cmap':'viridis'
            },
            label_kwargs={
                'labelpad':50,
            },
            hist_kwargs={
                'color':'black',
            }
        )
        return fig



if __name__ == "__main__":
    
    # measured mid-transit times
    Tc = np.array([ 
        2456666.0885, 2456820.3127, 2456675.9249, 2456646.3907,
        2456643.1046, 2456748.1087, 2456892.4882, 2456882.6523,
        2456754.6769, 2456928.5953, 2456941.7105, 2456751.3949,
        2456862.9673, 2456918.7523, 2456777.6495, 2456721.8694,
        2456669.3677, 2456833.4292, 2456859.6809, 2456849.8310,
        2456866.2488, 2456807.1814
    ])

    Tc_error = np.array([
        0.0016, 0.0029, 0.004, 0.0025, 
        0.0027, 0.0022, 0.0018, 0.0033, 
        0.0022, 0.0027, 0.0029, 0.0025, 
        0.0019, 0.0037, 0.0022, 0.0022, 
        0.0032, 0.0031, 0.0045, 0.0032, 
        0.003, 0.0022
    ])

    # Parameters for N-body Retrieval

    # read more about adding particles: https://rebound.readthedocs.io/en/latest/addingparticles/ 
    nbody_prior = [
        # star
        {'m':1.12}, # mass [msun]

        # inner planet
        {'m':0.28*mjup/msun, 
        'P':3.2888,
        'inc':3.14159/2,
        'e':0,
        'omega':0},

        # outer planet
        {'m':0.988*mjup/msun,
        'P':7,
        'inc':3.14159/2,
        'e':0,
        'omega':0 },
    ]

    # specify data to fit
    data = [
        {},                                 # data for star (e.g. RV)
        {'Tc':Tc, 'Tc_err':Tc_error},       # data for inner planet (e.g. Mid-transit times)
        {}                                  # data for outer planet (e.g. Mid-transit times)
    ]

    # set up where to look for a solution
    bounds = [ 

        {}, # no bounds on star parameters

        { # bounds for inner planet
            'P': [nbody_prior[1]['P']-0.025, nbody_prior[1]['P']+0.025],  # based on solution from linear fit
            # mid-transit is automatically adjusted in the fitter class
        },

        { # bounds for outer planet
            'P':[6.5,8.5],                      # orbital period [day]
            'm':[0.5*mjup/msun,1.5*mjup/msun],  # mass [msun]
            #'e':[0,0.05],                       # eccentricity
            #'omega':[-20*np.pi/180, 20*np.pi/180]           # arg of periastron [radians]
        }
    ]

    # fit the data with nested sampling
    ttvfit = nbody_fitter(data, nbody_prior, bounds)

    # set up some arrays for mapping sampler output
    freekeys = []
    for i,planet in enumerate(ttvfit.bounds):
        for bound in planet:
            freekeys.append(f"{i}_{bound}")

    samples = ttvfit.results['weighted_samples']['points'].copy()
    samples[:,1] /= samples[:,0] # convert to period ratio
    samples[:,2] *= msun/mjup # convert to jupiter mass ratio

    chi2 = np.log10(ttvfit.results['weighted_samples']['logl']*-2)
    mask = chi2 < np.percentile(chi2, 50)

    fig = corner(samples[mask],
        labels= ['','','',''],
        bins=int(np.sqrt(mask.sum())),
        #range= ranges,
        figsize=(10,10),
        #quantiles=(0.1, 0.84),
        plot_contours=False,
        #levels=[ np.percentile(chi2[mask1],95), np.percentile(chi2[mask2],95), np.percentile(chi2[mask3],95)],
        plot_density=False,
        titles=['Inner Planet Period [day]', 'Outer Planet Period Ratio', 'Outer Planet Mass [mjup]',''],
        data_kwargs={
            'c':chi2[mask],
            'vmin':np.percentile(chi2[mask],1),
            'vmax':np.percentile(chi2[mask],99),
            'cmap':'jet_r'
        },
        label_kwargs={
            'labelpad':50,
        },
        hist_kwargs={
            'color':'black',
        }
    )
    plt.show()

    # TODO monte carlo for Tmid and best-fit period
