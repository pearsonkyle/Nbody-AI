from astropy.constants import M_sun, M_earth, M_jup
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np

# convert to SI units
mearth = M_earth.value
msun = M_sun.value
mjup = M_jup.value


with fits.open('ttv_grid.fits') as ttv_grid_list:

    # The grid is stored in multiple extensions
    Npts = ttv_grid_list[0].header['NAXIS1']

    # in Earth
    mass_grid = np.linspace(ttv_grid_list[0].header['CRVAL1'], ttv_grid_list[0].header['CRVAL1'] + Npts*ttv_grid_list[0].header['CDELT1'], Npts)
    mass_grid_2 = np.linspace(ttv_grid_list[0].header['CRVAL1'], ttv_grid_list[0].header['CRVAL1'] + Npts*ttv_grid_list[0].header['CDELT1'], int(0.5*Npts))
    period_grid = np.linspace(ttv_grid_list[0].header['CRVAL2'], ttv_grid_list[0].header['CRVAL2'] + Npts*ttv_grid_list[0].header['CDELT2'], Npts)
    period_grid_2 = np.linspace(ttv_grid_list[0].header['CRVAL2'], ttv_grid_list[0].header['CRVAL2'] + Npts*ttv_grid_list[0].header['CDELT2'], int(0.5*Npts))

    # extract data
    pg, mg = np.meshgrid(period_grid, mass_grid)
    amplitude_grid = ttv_grid_list[0].data
    periodicity1_grid = ttv_grid_list[1].data
    periodicity2_grid = ttv_grid_list[2].data
    # combine periods by min
    #periodicity_grid = np.minimum(periodicity1_grid, periodicity2_grid)
    rv_grid = ttv_grid_list[3].data

    # amplitude grid
    fig, ax = plt.subplots(3, 2, figsize=(9, 11))
    fig.suptitle("N-body Estimates from Transit Timing Variations", fontsize=16)
    plt.subplots_adjust(left=0.0485, right=0.985, bottom=0.15, top=0.9, wspace=0.3)
    im = ax[0,0].imshow(amplitude_grid, origin='lower', extent=[mass_grid[0], mass_grid[-1],period_grid[0], period_grid[-1]], vmin=0,vmax=30, aspect='auto', cmap='jet', interpolation='none')
    ax[0,0].set_ylabel('Period Ratio')
    ax[0,0].set_xlabel('Mass [Earth]')
    cbar = fig.colorbar(im, ax=ax[0,0])
    cbar.set_label('TTV Amplitude [min]')

    # periodoicity 1 grid
    im = ax[0,1].imshow(periodicity1_grid, origin='lower', extent=[mass_grid[0], mass_grid[-1],period_grid[0], period_grid[-1]], vmin=0,vmax=0.5*np.nanmax(ttv_grid_list[1].data), aspect='auto', cmap='jet', interpolation='none')
    ax[0,1].set_ylabel('Period Ratio')
    ax[0,1].set_xlabel('Mass [Earth]')
    cbar = fig.colorbar(im, ax=ax[0,1])
    cbar.set_label('TTV Periodicity 1st Order [epoch]')

    # simulate some results based on the periodogram
    mask = (amplitude_grid > 1) & (amplitude_grid < 5) & (((periodicity1_grid > 15) & (periodicity1_grid < 25)) | ((periodicity2_grid > 15) & (periodicity2_grid < 25)))
    # plot mask
    im = ax[1,0].imshow(mask, origin='lower', extent=[mass_grid[0], mass_grid[-1],period_grid[0], period_grid[-1]], vmin=0,vmax=1, aspect='auto', cmap='binary_r', interpolation='none')
    ax[1,0].set_ylabel('Period Ratio')
    ax[1,0].set_xlabel('Mass [Earth]')
    cbar = fig.colorbar(im, ax=ax[1,0])
    cbar.set_label('N-body Prior from O-C Data')

    # compare to original
    # TODO find why the last bin has double the number of points
    masses = mg.T[mask].flatten()
    ax[1,1].hist(masses, bins=mass_grid_2, alpha=0.5)
    ax[1,1].set_xlabel('Mass [Earth]')
    ax[1,1].set_ylabel('Prior Probability')
    ax[1,1].axes.yaxis.set_ticklabels([])
    ax[1,1].set_xlim([mg.min(), mg.max()-10])

    # compare to original
    periods = pg.T[mask].flatten()
    ax[2,0].hist(periods, bins=period_grid, density=True, alpha=0.5)
    ax[2,0].set_xlabel('Period Ratio')
    ax[2,0].set_ylabel('Prior Probability')
    ax[2,0].axes.yaxis.set_ticklabels([])
    ax[2,0].set_xlim([pg.min(), pg.max()])

    # plot histogram for rv data
    rvs = rv_grid.T[mask].flatten()
    ax[2,1].hist(rvs, bins=50, density=True, alpha=0.5)
    ax[2,1].set_xlabel('RV Semi-Amplitude [m/s]')
    ax[2,1].set_ylabel('Prior Probability')
    ax[2,1].axes.yaxis.set_ticklabels([])
    plt.tight_layout()
    plt.savefig('ttv_grid.png')
    plt.show()

"""
# distribution similar to masses
nbins = 50
heights, edges = np.histogram(masses, bins=mass_grid, density=True)
edge_center = (edges[:-1] + edges[1:]) / 2

# Step 2: Normalize the histogram
bin_widths = np.diff(edges)
total_area = np.sum(bin_widths * heights)
normalized_heights = heights / total_area

# Step 3: Interpolate to create a continuous PDF
interp_func_mass = interp1d(edge_center, normalized_heights, kind='linear', bounds_error=False, fill_value=0)

# Step 4: Sample from the continuous distribution
new_x = np.sort(np.random.uniform(edge_center[0], edge_center[-1], size=1000))
samples = interp_func_mass(new_x)
#ax[1,1].plot(new_x, samples, 'k-')
"""