from nbody.tools import mjup,msun,mearth, sa, rv_semi, au

import matplotlib.pyplot as plt
import numpy as np

# sa = lambda m,P : (G*m*P**2/(4*np.pi**2) )**(1./3) 

if __name__ == "__main__":
    objects = [
        {'m':1.1933},
        {'m':0.43*mjup/msun, 'P':3.875, 'inc':3.14159/2, 'e':0, 'omega':0  }, 
        {'m':mearth/msun, 'P':7.2, 'inc':3.14159/2, 'e':0,  'omega':0  }, 
    ]

    periods = np.linspace(1.25*objects[1]['P'], 2.2*objects[1]['P'], 50 )
    masses = np.linspace( mearth/msun, 20*mearth/msun, 50 ) 

    rvs = np.zeros( (periods.shape[0],masses.shape[0]) )

    for i in range(periods.shape[0]):
        for j in range(masses.shape[0]):
            objects[2]['P'] = periods[i]
            objects[2]['m'] = masses[j]

            rvs[i,j] = rv_semi(objects[0]['m'], objects[2]['m'], sa(objects[0]['m'],objects[2]['P']) ) * au / (24*60*60)


            # formula from : https://arxiv.org/pdf/astro-ph/0412028v1.pdf
            #a1 = sa(objects[1]['m'], objects[1]['P'])
            #a2 = sa(objects[2]['m'], objects[2]['P'])
            #ae = a1/(a2*(1-objects[2]['e']))

            # terrible formula
            #dt = (objects[2]['m']/objects[0]['m'])*objects[1]['P']*ae**3 *(1-np.sqrt(2)*ae**(3./2))**-2
            #amin = objects[2]['m']/ ( objects[0]['m']**(2./3) * (objects[2]['P']**(2./3) - objects[1]['P']**(2./3))**2 )


    plt.imshow(rvs, cmap='gist_rainbow', origin='lower',
                extent=[ min(masses)*msun/mearth, max(masses)*msun/mearth, min(periods)/objects[1]['P'],max(periods)/objects[1]['P']],
                aspect=20, vmin=0, vmax=6)

    plt.ylabel(r'Period Ratio [P$_{outer}$/P$_{inner}$]')
    plt.xlabel('Mass of Outer Planet [Earth]')
    #plt.title('M* = {:.2f} [Sun], P1 = {:.2f} [day]'.format(objects[0]['m'], objects[1]['P']) )
    plt.title("RV Semi-amplitude for an Outer Planet")
    plt.ylim([1.25,2.2])
    cbar = plt.colorbar()
    cbar.set_label('RV Semi-amplitude [m/s]', rotation=270, labelpad=10)
    plt.savefig('rv_grid.pdf',bbox_inches='tight')
    plt.close()


    '''
    __1___
    |   /
   10  /
    | /
    |/
    '''