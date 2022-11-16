import copy
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas.util.testing as tm
import matplotlib.pyplot as plt
from exotic.api.plotting import corner
from ultranest import ReactiveNestedSampler
from astropy.timeseries import LombScargle


class linear_fitter(object):

    def __init__(self, data, dataerr, bounds=None, prior=None):
        self.data = data
        self.dataerr = dataerr
        self.bounds = bounds
        self.prior = prior # dict {'m':(0.1,0.5), 'b':(0,1)}
        if bounds is None:
            # use +- 3 sigma prior as bounds
            self.bounds = {
                'm':[prior['m'][0]-3*prior['m'][1],prior['m'][0]+3*prior['m'][1]],
                'b':[prior['b'][0]-3*prior['b'][1],prior['b'][0]+3*prior['b'][1]]
            }
        self.fit_nested()

    def fit_nested(self):
        freekeys = list(self.bounds.keys())
        boundarray = np.array([self.bounds[k] for k in freekeys])
        bounddiff = np.diff(boundarray,1).reshape(-1)
        self.epochs = np.round((self.data - np.mean(self.bounds['b']))/np.mean(self.bounds['m']))

        def loglike(pars):
            # chi-squared
            model = pars[0]*self.epochs + pars[1]
            return -0.5 * np.sum( ((self.data-model)/self.dataerr)**2 )
        
        def prior_transform(upars):
            # transform unit cube to prior volume
            return (boundarray[:,0] + bounddiff*upars)

        # estimate slope and intercept
        self.results = ReactiveNestedSampler(freekeys, loglike, prior_transform).run(max_ncalls=4e5,min_num_live_points=420, show_status=False)

        # alloc data for best fit + error
        self.errors = {}
        self.quantiles = {}
        self.parameters = {}

        for i, key in enumerate(freekeys):
            self.parameters[key] = self.results['maximum_likelihood']['point'][i]
            self.errors[key] = self.results['posterior']['stdev'][i]
            self.quantiles[key] = [
                self.results['posterior']['errlo'][i],
                self.results['posterior']['errup'][i]]

        # final model
        self.model = self.epochs * self.parameters['m'] + self.parameters['b']
        self.residuals = self.data - self.model

    def plot_oc(self, savefile=None, ylim='none', show_2sigma=False):
        # O-C plot
        fig,ax = plt.subplots(1, figsize=(9,6))

        ax.errorbar(self.epochs, self.residuals*24*60, yerr=self.dataerr*24*60, ls='none', marker='o',color='black')
        ylower = (self.residuals.mean()-3*np.std(self.residuals))*24*60
        yupper = (self.residuals.mean()+3*np.std(self.residuals))*24*60

        # upsample data
        epochs = (np.linspace(self.data.min()-7, self.data.max()+7, 1000) - self.parameters['b'])/self.parameters['m']

        depoch = self.epochs.max() - self.epochs.min()
        ax.set_xlim([self.epochs.min()-depoch*0.01, self.epochs.max()+depoch*0.01])

        # best fit solution
        model = epochs*self.parameters['m'] + self.parameters['b']
    
        # MonteCarlo the new ephemeris for uncertainty
        mc_m = np.random.normal(self.parameters['m'], self.errors['m'], size=10000)
        mc_b = np.random.normal(self.parameters['b'], self.errors['b'], size=10000)
        mc_model = np.expand_dims(epochs,-1) * mc_m + mc_b

        # create a fill between area for uncertainty of new ephemeris
        diff = mc_model.T - model

        if show_2sigma:
            ax.fill_between(epochs, np.percentile(diff,2,axis=0)*24*60, np.percentile(diff,98,axis=0)*24*60, alpha=0.2, color='k', label=r'Uncertainty ($\pm$ 2$\sigma$)')
        else:
            # show 1 sigma
            ax.fill_between(epochs, np.percentile(diff,36,axis=0)*24*60, np.percentile(diff,64,axis=0)*24*60, alpha=0.2, color='k', label=r'Uncertainty ($\pm$ 1$\sigma$)')

        # duplicate axis and plot days since mid-transit
        ax2 = ax.twiny()
        ax2.set_xlabel(f"Time [BJD - {self.parameters['b']:.1f}]",fontsize=14)
        ax2.set_xlim(ax.get_xlim())
        xticks = ax.get_xticks()
        dt = np.round(xticks*self.parameters['m'],1)
        #ax2.set_xticks(dt)
        ax2.set_xticklabels(dt)

        if ylim == 'diff':
            ax.set_ylim([ min(np.percentile(diff,1,axis=0)*24*60),
                          max(np.percentile(diff,99,axis=0)*24*60)])

        if self.prior is not None:
            # create fill between area for uncertainty of old/prior ephemeris
            epochs_p = (np.linspace(self.data.min()-7, self.data.max()+7, 1000) - self.prior['b'][0])/self.prior['m'][0]
            prior = epochs_p*self.prior['m'][0] + self.prior['b'][0]
            mc_m_p = np.random.normal(self.prior['m'][0], self.prior['m'][1], size=10000)
            mc_b_p = np.random.normal(self.prior['b'][0], self.prior['b'][1], size=10000)
            mc_model_p = np.expand_dims(epochs_p,-1) * mc_m_p + mc_b_p
            diff_p = mc_model_p.T - model

            # plot an invisible line so the 2nd axes are happy
            ax2.plot(epochs, (model-prior)*24*60, ls='--', color='r', alpha=0)

            # why is this so small!?!?!?
            #ax.plot(epochs, (model-prior)*24*60, ls='--', color='r')

            if show_2sigma:
                ax.fill_between(epochs, np.percentile(diff_p,2,axis=0)*24*60, np.percentile(diff_p,98,axis=0)*24*60, alpha=0.1, color='r', label=r'Prior ($\pm$ 2$\sigma$)')
            else:
                # show ~1 sigma
                ax.fill_between(epochs, np.percentile(diff_p,36,axis=0)*24*60, np.percentile(diff_p,64,axis=0)*24*60, alpha=0.1, color='r', label=r'Prior ($\pm$ 1$\sigma$)')

            if ylim == 'prior':
                ax.set_ylim([ min(np.percentile(diff_p,1,axis=0)*24*60),
                            max(np.percentile(diff_p,99,axis=0)*24*60)])
            elif ylim == 'average':
                ax.set_ylim([ 0.5*(min(np.percentile(diff,1,axis=0)*24*60) + min(np.percentile(diff_p,1,axis=0)*24*60)),
                            0.5*(max(np.percentile(diff,99,axis=0)*24*60) + max(np.percentile(diff_p,99,axis=0)*24*60))])

        ax.axhline(0,color='black',alpha=0.5,ls='--',
                   label="Period: {:.7f}+-{:.7f} days\nT_mid: {:.7f}+-{:.7f} BJD".format(self.parameters['m'], self.errors['m'], self.parameters['b'], self.errors['b']))

        # TODO sig figs
        #lclabel2 = r"$T_{mid}$ = %s $\pm$ %s BJD$_{TDB}$" %(
        #    str(round_to_2(self.parameters['tmid'], self.errors.get('tmid',0))),
        #    str(round_to_2(self.errors.get('tmid',0)))
        #)

        ax.legend(loc='best')
        ax.set_xlabel("Epoch [number]",fontsize=14)
        ax.set_ylabel("Residuals [min]",fontsize=14)
        ax.grid(True, ls='--')
        return fig, ax

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
            titles.append(f"{self.parameters[key]:.7f} +-\n {self.errors[key]:.7f}")
            ranges.append([
                self.parameters[key] - 5*self.errors[key],
                self.parameters[key] + 5*self.errors[key]
            ])

            if key == 'a2' or key == 'a1':
                continue

            mask3 = mask3 & (self.results['weighted_samples']['points'][:,i] > (self.parameters[key] - 3*self.errors[key]) ) & \
                (self.results['weighted_samples']['points'][:,i] < (self.parameters[key] + 3*self.errors[key]) )

            mask1 = mask1 & (self.results['weighted_samples']['points'][:,i] > (self.parameters[key] - self.errors[key]) ) & \
                (self.results['weighted_samples']['points'][:,i] < (self.parameters[key] + self.errors[key]) )

            mask2 = mask2 & (self.results['weighted_samples']['points'][:,i] > (self.parameters[key] - 2*self.errors[key]) ) & \
                (self.results['weighted_samples']['points'][:,i] < (self.parameters[key] + 2*self.errors[key]) )

        chi2 = self.results['weighted_samples']['logl']*-2
        fig = corner(self.results['weighted_samples']['points'],
            labels= labels,
            bins=int(np.sqrt(self.results['samples'].shape[0])),
            range= ranges,
            figsize=(10,10),
            #quantiles=(0.1, 0.84),
            plot_contours=True,
            levels=[ np.percentile(chi2[mask1],95), np.percentile(chi2[mask2],95), np.percentile(chi2[mask3],95)],
            plot_density=False,
            titles=titles,
            data_kwargs={
                'c':chi2,
                'vmin':np.percentile(chi2[mask3],1),
                'vmax':np.percentile(chi2[mask3],95),
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

    def plot_periodogram(self):

        # Search for periodic signals in residuals after linear fit
        freq,power = LombScargle(self.epochs, self.residuals).autopower(nyquist_factor=2)

        # change up the frequency grid a little
        maxper = np.max(self.epochs) - np.min(self.epochs)
        minper = (1./freq).min()

        # recompute on new grid
        freq,power = LombScargle(self.epochs, self.residuals).autopower(minimum_frequency=1./maxper, maximum_frequency=1./minper, nyquist_factor=2)

        # Phase fold data at max peak
        mi = np.argmax(power)
        per = 1./freq[mi]
        newphase = self.epochs/per % 1
        self.periods = 1./freq
        self.power = power

        # find best fit signal with 1 period
        # construct basis vectors with sin and cos
        basis = np.ones((3, len(self.epochs)))
        basis[0] = np.sin(2*np.pi*self.epochs/per)
        basis[1] = np.cos(2*np.pi*self.epochs/per)
        # fit for the coefficients
        coeffs = np.linalg.lstsq(basis.T, self.residuals, rcond=None)[0]
        # reconstruct signal
        y_bestfit = np.dot(basis.T, coeffs)
        # super sample fourier solution
        xnew = np.linspace(self.epochs.min(), self.epochs.max(), 1000)
        basis_new = np.ones((3, len(xnew)))
        basis_new[0] = np.sin(2*np.pi*xnew/per)
        basis_new[1] = np.cos(2*np.pi*xnew/per)
        y_bestfit_new = np.dot(basis_new.T, coeffs)


        # create plot
        fig, ax = plt.subplots(3, figsize=(10,12))

        # periodogram plot
        ax[0].semilogx(self.periods,self.power,'k-')
        ax[0].semilogx(self.periods,self.power,'k.')
        ax[0].set_xlabel("Period [epoch]",fontsize=14)
        ax[0].set_xlim([min(self.periods),max(self.periods)])
        ax[0].set_ylabel('Power',fontsize=14)
        ax[0].axvline(per,color='red')
        ax[0].set_title("Lomb-Scargle Periodogram")

        # o-c time series with fourier solution
        ax[1].errorbar(self.epochs,self.residuals*24*60,
                    yerr=self.dataerr*24*60,ls='none',
                    marker='o',color='black',
                    label=f'Data')
        ax[1].plot(xnew, y_bestfit_new*24*60, 'r-', label=f'Best fit (Period: {per:.2f})')
        ax[1].set_xlabel(f"Epochs",fontsize=14)
        ax[1].set_ylabel("O-C [min]",fontsize=14)
        ax[1].legend(loc='best')
        ax[1].grid(True,ls='--')

        # phase folded time series with fourier solution
        ax[2].errorbar(newphase,self.residuals*24*60,
                    yerr=self.dataerr*24*60,ls='none',
                    marker='o',color='black',
                    label=f'Data')

        xnewphase = xnew/per % 1
        ax[2].plot(xnewphase, y_bestfit_new*24*60, 'r.', ms=4, label=f'Best fit')

        # sort data in phase
        si = np.argsort(newphase)
        # bin data into 8 bins
        bins = np.linspace(0,1,8)
        binned = np.zeros(len(bins))
        binned_std = np.zeros(len(bins))

        for i in range(0,len(bins)):
            mask = np.digitize(newphase[si], bins)==i
            if mask.sum() > 1:
                binned[i] = np.mean(self.residuals[si][mask])
                binned_std[i] = np.std(self.residuals[si][mask])
            elif mask.sum() == 1:
                binned[i] = self.residuals[si][mask]
                binned_std[i] = self.dataerr[si][mask]
            else:
                binned[i] = np.nan
                binned_std[i] = np.nan

        ax[2].errorbar(bins-0.5/len(bins),binned*24*60,
                    yerr=binned_std*24*60,ls='none',
                    marker='o',color='limegreen',
                    label=f'Binned Data')

        ax[2].set_xlabel(f"Phase (Period: {per:.2f} epochs)",fontsize=14)
        ax[2].set_ylabel("O-C [min]",fontsize=14)
        ax[2].legend(loc='best')
        ax[2].grid(True,ls='--')
        return fig,ax

# Function that bins an array
def binner(arr, n, err=''):
    if len(err) == 0:
        ecks = np.pad(arr.astype(float), (0, ((n - arr.size % n) % n)), mode='constant',
                      constant_values=np.NaN).reshape(-1, n)
        arr = np.nanmean(ecks, axis=1)
        return arr
    else:
        ecks = np.pad(arr.astype(float), (0, ((n - arr.size % n) % n)), mode='constant',
                      constant_values=np.NaN).reshape(-1, n)
        why = np.pad(err.astype(float), (0, ((n - err.size % n) % n)), mode='constant', constant_values=np.NaN).reshape(
            -1, n)
        weights = 1. / (why ** 2.)
        # Calculate the weighted average
        arr = np.nansum(ecks * weights, axis=1) / np.nansum(weights, axis=1)
        err = np.array([np.sqrt(1. / np.nansum(1. / (np.array(i) ** 2.))) for i in why])
        return arr, err


def main():
    Tc = np.array([ # measured mid-transit times
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

    P = 3.2888  # orbital period for your target


    Tc_norm = Tc - Tc.min()  #normalize the data to the first observation
    #print(Tc_norm)
    orbit = np.rint(Tc_norm / P)  #number of orbits since first observation (rounded to nearest integer)
    #print(orbit)

    A = np.vstack([np.ones(len(Tc)), orbit]).T 
    #make a n x 2 matrix with 1's in the first column and values of orbit in the second

    res = sm.WLS(Tc, A, weights=1.0/Tc_error**2).fit() #perform the weighted least squares regression
    #pass in the T_c's, the new orbit matrix A, and the weights
    #use sm.WLS for weighted LS, sm.OLS for ordinary LS, or sm.GLS for general LS

    params = res.params #retrieve the slope and intercept of the fit from res
    std_dev = np.sqrt(np.diagonal(res.normalized_cov_params)) 

    slope = params[1]
    slope_std_dev = std_dev[1]
    intercept = params[0]
    intercept_std_dev = std_dev[0]

    #print(res.summary())
    #print("Params =",params)
    #print("Error matrix =",res.normalized_cov_params)
    #print("Standard Deviations =",std_dev)

    print("Weighted Linear Least Squares Solution")
    print("T0 =",intercept,"+-",intercept_std_dev)
    print("P =",slope,"+-",slope_std_dev)

    # min and max values to search between for fitting
    bounds = {
        'm':[P-0.1, P+0.1],                 # orbital period
        'b':[intercept-0.1, intercept+0.1]  # mid-transit time
    }

    # used to plot red overlay in O-C figure
    prior = {
        'm':[slope, slope_std_dev],         # value from WLS (replace with literature value)
        'b':[intercept, intercept_std_dev]  # value from WLS (replace with literature value)
    }   

    lf = linear_fitter( Tc, Tc_error, bounds, prior=prior )


    lf.plot_triangle()
    plt.subplots_adjust(top=0.9,hspace=0.2,wspace=0.2)
    plt.savefig("posterior.png")
    plt.close()
    print("image saved to: posterior.png")

    fig,ax = lf.plot_oc()
    plt.tight_layout()
    plt.savefig("oc.png")
    plt.close()
    print("image saved to: oc.png")

    fig,ax = lf.plot_periodogram()
    plt.tight_layout()
    plt.savefig("periodogram.png")
    plt.close()
    print("image saved to: periodogram.png")

if __name__ == "__main__":
    main()