# create grid of simulations over various masses and period ratios
import sys
sys.path.append('../')
from nbody.simulation import generate, analyze, report, transit_times, TTV
from nbody.tools import rv_semi, sa, au

import argparse
import numpy as np
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.signal import find_peaks
from astropy.timeseries import LombScargle
from astropy.constants import M_sun, M_earth, M_jup

from simulation_grid import generate_sim

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    help_ = "Number of planet 1 orbital epochs/periods to integrate the simulation for"
    parser.add_argument("-e", "--epochs", help=help_, default=200, type=int)
    help_ = "Number of grid points in mass and period ratio"
    parser.add_argument("-n", "--Npts", help=help_, default=25, type=int)
    help_ = "Number of harmonics to fit to TTVs"
    parser.add_argument("-o", "--n_order", help=help_, default=4, type=int)
    help_ = "Number of omegas to compute TTVs for"
    parser.add_argument("-w", "--n_omega", help=help_, default=16, type=int)
    args = parser.parse_args()

    # convert to SI units
    mearth = M_earth.value
    msun = M_sun.value
    mjup = M_jup.value

    # define stellar and planetary parameters
    objects = [
        # stellar parameters
        {
            'm': 1 # np.random.uniform(0.75,1.15), # star mass [msun]
        },

        # planet 1
        {
            'm': 10*mearth/msun, # don't mass this mass interfering with orbit of outer planet
            'P': 1.42,
            'inc': np.deg2rad(90),
            'e': 0,
            'omega': 0.01
        },

        # planet 2
        {
            'm':  10*mearth/msun,
            # P - conditional, must be beyond hill radius of planet 1
            # objects[2]['P'] = 1.5*objects[1]['P'] # set period of planet 2 to be 1.5x that of planet 
            'omega': 0.,
            'e': 0,
            'inc': np.deg2rad(90)
        }
    ]

    # double for loop over mass and period ratio for outer planet
    Npts = args.Npts
    mass_range = np.linspace(0.5, 420, Npts) * mearth/msun
    period_range = np.linspace(1.45, 3.11, Npts) * objects[1]['P']
    mass_grid, period_grid = np.meshgrid(mass_range, period_range)
    omega_range = np.linspace(0.0*np.pi, 2.0*np.pi, args.n_omega)

    # amplitude of O-C residuals
    ttv_grid = np.zeros((Npts,Npts))

    # grids for reconstructing TTVs
    per_epoch_grid = np.zeros((args.n_order,Npts,Npts))
    per_omega_grid = np.zeros((args.n_order,Npts,Npts))
    amplitude_grid = np.zeros((args.n_order*4,Npts,Npts))

    # TTV(w) = sin(2pi/P_ni * n)[a_i*sin(w/P_wi) + b_i*cos(w/P_wi)] + 
    #          cos(2pi/P_ni * n)[c_i*sin(w/P_wi) + d_i*cos(w/P_wi)] + ...

    # average error per data point in reconstructed TTV
    error_grid = np.zeros((Npts,Npts))

    # radial velocity for outer planet
    rv_grid = np.zeros((Npts,Npts))

    for i in tqdm(range(Npts)):
        for j in range(Npts):

            # update planet 2 parameters
            objects[2]['m'] = mass_grid[i][j]
            objects[2]['P'] = period_grid[i][j]

            # mini-grids for omega
            ttvs = np.zeros(args.n_omega)
            errs = np.zeros(args.n_omega)
            pers_epoch = np.zeros((args.n_omega, args.n_order))
            amps = np.zeros((args.n_omega, args.n_order*2))
            # Asin(2pi/P_n * n) + Bcos(2pi/P_n * n)

            # loop over omega and compute TTV
            for k,w in enumerate(omega_range):
                objects[2]['omega'] = w

                # generate simulation
                pers_e, coeffs, ttvs[k], errs[k] = generate_sim(objects, epochs=args.epochs, n_order=args.n_order)

                # store period and amplitude of each harmonic
                pers_epoch[k,:len(pers_e)] = pers_e
                import pdb; pdb.set_trace()
                amps[k,:len(amps)] = coeffs[2:]

            # for each order find the period of the coefficient
            pers_omega = np.zeros(args.n_omega, args.n_order)
            for k in range(args.n_order):

                ls = LombScargle(omega_range, amps[2*k])
                freq,power = ls.autopower(maximum_frequency=1./1.1, minimum_frequency=1./20, 
                                          nyquist_factor=2, samples_per_peak=20, method='cython')
                # find peaks in periodogram
                peaks,amps = find_peaks(power, height=0.05)
                peaks = peaks[np.argsort(amps['peak_heights'])[::-1]]
                peak_periods = 1./freq[peaks]

                # limit to nth order
                peak_periods = peak_periods[:args.n_order]

                # add periods to grid
                pers_omega[:len(peak_periods)] = peak_periods

            # set up basis functions for omega based TTV signal
            # TTV(w) = sin(2pi/P_ni * n)[a_i*sin(w/P_wi) + b_i*cos(w/P_wi)] + 
            #          cos(2pi/P_ni * n)[c_i*sin(w/P_wi) + d_i*cos(w/P_wi)] + ...
            # P_ni = per_epochs
            # P_wi = per_omega
            import pdb; pdb.set_trace()

            basis_list = []
            for k in range(args.n_order):
                if pers_epoch[k] > 0:
                    basis_list.append(np.sin(2*np.pi/pers_epoch[k] * np.arange(args.epochs))*np.sin(omega_range/pers_omega[k]))
                    basis_list.append(np.sin(2*np.pi/pers_epoch[k] * np.arange(args.epochs))*np.cos(omega_range/pers_omega[k]))
                    basis_list.append(np.cos(2*np.pi/pers_epoch[k] * np.arange(args.epochs))*np.sin(omega_range/pers_omega[k]))
                    basis_list.append(np.cos(2*np.pi/pers_epoch[k] * np.arange(args.epochs))*np.cos(omega_range/pers_omega[k]))

            # create design matrix
            design_matrix = np.vstack(basis_list).T

            # solve for coefficients
            coeffs = np.linalg.lstsq(design_matrix, ttvs, rcond=None)[0]

            # reconstruct TTVs
            ttv_recon = np.dot(design_matrix, coeffs)


            # compute average error per data point
            error_grid[i,j] = np.mean(errs)

            # estimate radial velocity amplitude of outer planet in m/s
            rv_grid[i][j] = rv_semi( objects[0]['m'], objects[2]['m'], sa(objects[0]['m'],objects[2]['P'])) * au/(24*60*60)



    # create custom color bar with 0-2 min as white and 2-30 min as jet
    amp_minutes = ttv_grid*24*60
    cmap = plt.cm.gist_rainbow
    cmaplist = [cmap(i) for i in range(cmap.N)][::-1]
    cmaplist[0] = (1,1,1,1.0)
    cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)

    # create 2D imshow plots of amplitude and periodicity
    fig, ax = plt.subplots(2,2,figsize=(11,10))
    im = ax[0,0].imshow(ttv_grid*24*60, origin='lower', aspect='auto', vmin=2, vmax=30, cmap=cmap,
                 extent=[
                    mass_grid.min()*msun/mearth, mass_grid.max()*msun/mearth,
                    period_grid.min()/objects[1]['P'], period_grid.max()/objects[1]['P']])
    cbar = fig.colorbar(im, ax=ax[0,0])
    cbar.set_label('TTV Amplitude [min]')
    ax[0,0].set_xlabel('Mass [Mearth]')
    ax[0,0].set_ylabel('Period Ratio')

    periodicity1_grid = periodicity_grid[0]
    periodicity2_grid = periodicity_grid[1]
    mask = ((ttv_grid*24*60) > 0.5)
    periodicity1_grid[~mask] = np.nan
    periodicity2_grid[~mask] = np.nan

    im = ax[1,0].imshow(periodicity1_grid, origin='lower', aspect='auto', vmin=0, vmax=args.epochs,cmap=cmap,
                 extent=[
                    mass_grid.min()*msun/mearth, mass_grid.max()*msun/mearth,
                    period_grid.min()/objects[1]['P'], period_grid.max()/objects[1]['P']])
    cbar = fig.colorbar(im, ax=ax[1,0])

    cbar.set_label('Periodicity [epochs]')
    ax[1,0].set_title('O-C Periodicity 1st')
    ax[1,0].set_xlabel('Mass [Mearth]')
    ax[1,0].set_ylabel('Period Ratio')

    im = ax[1,1].imshow(periodicity2_grid, origin='lower', aspect='auto', vmin=0, vmax=args.epochs,cmap=cmap,
                extent=[
                mass_grid.min()*msun/mearth, mass_grid.max()*msun/mearth,
                period_grid.min()/objects[1]['P'], period_grid.max()/objects[1]['P']])
    cbar = fig.colorbar(im, ax=ax[1,1])

    cbar.set_label('Periodicity [epochs]')
    ax[1,1].set_title('O-C Periodicity 2nd')
    ax[1,1].set_xlabel('Mass [Mearth]')
    ax[1,1].set_ylabel('Period Ratio')

    #rv_grid[~mask] = np.nan
    # create radial velocity amplitude plot
    im = ax[0,1].imshow(rv_grid, origin='lower', aspect='auto', vmin=1, vmax=50, cmap='gist_rainbow_r',
                    extent=[
                        mass_grid.min()*msun/mearth, mass_grid.max()*msun/mearth,
                        period_grid.min()/objects[1]['P'], period_grid.max()/objects[1]['P']])
    cbar = fig.colorbar(im, ax=ax[0,1])
    cbar.set_label('Radial Velocity Amplitude [m/s]')
    ax[0,1].set_title('Radial Velocity Amplitude')
    ax[0,1].set_xlabel('Mass [Mearth]')
    ax[0,1].set_ylabel('Period Ratio')

    plt.tight_layout()
    fig_name = f'grid_{args.epochs}_{args.Npts}.png'
    plt.savefig(fig_name)
    plt.show()

    dude()
    # TODO fix bottom

    # create a FITS image with multiple fields
    hdu_amplitude = fits.PrimaryHDU(ttv_grid*24*60)
    hdu_amplitude.header['BUNIT'] = 'minutes'
    hdu_amplitude.header['COMMENT'] = 'TTV amplitude in minutes'
    # x-axis values
    hdu_amplitude.header['CRPIX1'] = 1
    hdu_amplitude.header['CRVAL1'] = mass_grid.min()*msun/mearth
    hdu_amplitude.header['CDELT1'] = (mass_grid.max()*msun/mearth - mass_grid.min()*msun/mearth)/Npts
    # y-axis values
    hdu_amplitude.header['CRPIX2'] = 1
    hdu_amplitude.header['CRVAL2'] = period_grid.min()/objects[1]['P']
    hdu_amplitude.header['CDELT2'] = (period_grid.max()/objects[1]['P'] - period_grid.min()/objects[1]['P'])/Npts

    hdu_periodicity1 = fits.ImageHDU(periodicity1_grid)
    hdu_periodicity1.header['BUNIT'] = 'epochs'
    hdu_periodicity1.header['COMMENT'] = 'Periodicity of O-C in epochs'
    # x-axis values
    hdu_periodicity1.header['CRVAL1'] = mass_grid.min()*msun/mearth
    hdu_periodicity1.header['CRPIX1'] = 1
    hdu_periodicity1.header['CDELT1'] = (mass_grid.max()*msun/mearth - mass_grid.min()*msun/mearth)/Npts
    # y-axis values
    hdu_periodicity1.header['CRVAL2'] = period_grid.min()/objects[1]['P']
    hdu_periodicity1.header['CRPIX2'] = 1
    hdu_periodicity1.header['CDELT2'] = (period_grid.max()/objects[1]['P'] - period_grid.min()/objects[1]['P'])/Npts

    hdu_periodicity2 = fits.ImageHDU(periodicity2_grid)
    hdu_periodicity2.header['BUNIT'] = 'epochs'
    hdu_periodicity2.header['COMMENT'] = 'Periodicity of O-C in epochs'
    # x-axis values
    hdu_periodicity2.header['CRVAL1'] = mass_grid.min()*msun/mearth
    hdu_periodicity2.header['CRPIX1'] = 1
    hdu_periodicity2.header['CDELT1'] = (mass_grid.max()*msun/mearth - mass_grid.min()*msun/mearth)/Npts
    # y-axis values
    hdu_periodicity2.header['CRVAL2'] = period_grid.min()/objects[1]['P']
    hdu_periodicity2.header['CRPIX2'] = 1
    hdu_periodicity2.header['CDELT2'] = (period_grid.max()/objects[1]['P'] - period_grid.min()/objects[1]['P'])/Npts

    # radial velocity amplitude
    hdu_rv = fits.ImageHDU(rv_grid)
    hdu_rv.header['BUNIT'] = 'm/s'
    hdu_rv.header['COMMENT'] = 'Radial velocity amplitude in m/s'
    # x-axis values
    hdu_rv.header['CRVAL1'] = mass_grid.min()*msun/mearth
    hdu_rv.header['CRPIX1'] = 1
    hdu_rv.header['CDELT1'] = (mass_grid.max()*msun/mearth - mass_grid.min()*msun/mearth)/Npts
    # y-axis values
    hdu_rv.header['CRVAL2'] = period_grid.min()/objects[1]['P']
    hdu_rv.header['CRPIX2'] = 1
    hdu_rv.header['CDELT2'] = (period_grid.max()/objects[1]['P'] - period_grid.min()/objects[1]['P'])/Npts

    def add_objects_to_header(hdu_header):
        hdu_header.header['STARMASS'] = objects[0]['m']
        hdu_header.header['P1_MASS'] = objects[1]['m']
        hdu_header.header['P1_PER'] = objects[1]['P']
        hdu_header.header['P1_INC'] = objects[1]['inc']
        hdu_header.header['P1_ECC'] = objects[1]['e']
        hdu_header.header['P1_OMEGA'] = objects[1]['omega']
        hdu_header.header['P2_MASS'] = objects[2]['m']
        hdu_header.header['P2_PER'] = objects[2]['P']
        hdu_header.header['P2_INC'] = objects[2]['inc']
        hdu_header.header['P2_ECC'] = objects[2]['e']
        hdu_header.header['P2_OMEGA'] = objects[2]['omega']

    # add objects to header
    add_objects_to_header(hdu_amplitude)
    add_objects_to_header(hdu_periodicity1)
    add_objects_to_header(hdu_periodicity2)
    add_objects_to_header(hdu_rv)

    # give name to extensions
    hdu_amplitude.header['EXTNAME'] = 'TTV_AMPLITUDE'
    hdu_periodicity1.header['EXTNAME'] = 'TTV_PERIODICITY1'
    hdu_periodicity2.header['EXTNAME'] = 'TTV_PERIODICITY2'
    hdu_rv.header['EXTNAME'] = 'RV_AMPLITUDE'

    # save to FITS file
    hdul = fits.HDUList([hdu_amplitude, hdu_periodicity1, hdu_periodicity2, hdu_rv])
    hdul.writeto('ttv_grid.fits', overwrite=True)