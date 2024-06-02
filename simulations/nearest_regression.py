import sys
sys.path.append('../')
from nbody.tools import msun, mearth, mjup
from nbody.simulation import randomize, generate, analyze, report, transit_times

import glob
import pickle
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from exotic.api.plotting import corner

import faiss

class FaissKNeighbors:
    def __init__(self, k=5):
        self.index = None
        self.y = None
        self.k = k

    def fit(self, X, y):
        self.index = faiss.IndexFlatL2(X.shape[1])
        self.index.add(X.astype(np.float32))
        self.y = y

    def predict(self, X):
        distances, indices = self.index.search(X.astype(np.float32), k=self.k)
        return self.y[indices]

    def transform(self, X):
        distances, indices = self.index.search(X.astype(np.float32), k=self.k)
        return indices

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    help_ = "Input pickle file of simulations"
    parser.add_argument("-i", "--input", help=help_)
    args = parser.parse_args()

    # Load N-body simulations
    Xsim,ysim = pickle.load(open(args.input,'rb'))
    # Xsim: mstar [msun], P1 [day], m1 [mearth], P2 [day], m2 [mearth], omega [rad], ecc
    # ysim: simulation time of mid-transit [min]

    # get random index
    idx = np.random.randint(0,len(Xsim))

    # generate random test data
    Xtest = Xsim[idx]

    # get dictionary for Nbody simulation
    objects = randomize()

    objects[0]['m'] = Xtest[0]
    objects[1]['P'] = Xtest[1]
    objects[1]['m'] = Xtest[2]*mearth/msun
    objects[1]['inc'] = np.pi/2

    objects[2]['P'] = Xtest[3]
    objects[2]['m'] = Xtest[4]*mearth/msun
    objects[2]['omega'] = Xtest[5]
    objects[2]['e'] = Xtest[6]
    objects[2]['inc'] = np.pi/2

    # generate Nbody simulation
    sim_data = generate(objects,
        Ndays=objects[1]['P']*50,
        Noutputs=round(objects[1]['P']*50))

    # generate some fake transit times
    Tc_sim_og = transit_times( sim_data['pdata'][0]['x'], sim_data['star']['x'], sim_data['times'] ) # in days
    epochs_og = np.arange(len(Tc_sim_og))

    # add gaussian noise of order 1 minute + randomly downsample to 66% of data
    Tc_sim = Tc_sim_og + np.random.normal(0,1/60./24.,len(Tc_sim_og))
    epochs = np.random.choice(np.arange(len(Tc_sim)),int(len(Tc_sim)*0.66),replace=False)
    Tc_sim = Tc_sim[epochs]

    # convert to O-C graph
    A = np.vstack([epochs,np.ones(len(Tc_sim))]).T
    m,b = np.linalg.lstsq(A,Tc_sim,rcond=None)[0]
    oc_sim = Tc_sim - (m*epochs+b)

    # plot O-C graph
    fig,ax = plt.subplots(1,1,figsize=(8,6))
    ax.plot(epochs_og, Tc_sim_og-m*epochs_og-b, 'g-', label='N-body Simulation')
    ax.plot(epochs_og, (ysim[idx]/24/60)-m*epochs_og-b, 'r-', label='Data Base Simulation')
    ax.scatter(epochs,oc_sim,s=10,c='k',marker='o',label='Simulated+Noise')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('O-C [days]')
    ax.legend()
    plt.show()

    dude()


    # find nearest neighbors

    # plot corner of nearest neighbors



    # find lengths of each y vector (transit times)
    lengths = np.array([len(yi) for yi in ysim])

    # find median length
    median_length = np.median(lengths)
    print('median length:',median_length)

    # loop over list and extract same length vectors
    y = np.zeros((len(X),int(median_length)))
    mask = np.zeros(len(X),dtype=bool)
    for i in range(len(X)):
        if lengths[i] >= median_length:
            y[i,:] = ysim[i][:int(median_length)]
            mask[i] = True
        else:
            # simulation too short
            mask[i] = False

    # mask out bad simulations
    X = X[mask]
    y = y[mask]

    yt = np.zeros(y.shape) # will be residuals (similar to O-C in minutes)
    ym = np.zeros(y.shape) # subtract first transit time from each

    linear_coeffs = np.zeros((y.shape[0],2))
    # loop over each simulation and subtract out a linear fit
    for i in range(y.shape[0]):
        linear_coeffs[i], epochs = np.polyfit(np.arange(y.shape[1]),y[i,:],1),np.arange(y.shape[1])
        yt[i,:] = y[i,:] - np.polyval(linear_coeffs[i],epochs)
        ym[i,:] = y[i,:] - y[i,0]

    print('X shape:',X.shape)
    print('y shape:',y.shape)

    # filter out simulations with variations larger than 60 minutes
    mask = (np.abs(yt).max(1) < 60) & (np.abs(yt).max(1) > 0.5)
    X = X[mask]
    y = y[mask]
    yt = yt[mask]
    ym = ym[mask]

    # mask star within 1+-0.01 solar masses
    # mask = (X[:,0] > 0.95) & (X[:,0] < 1.05)
    # X = X[mask]
    # y = y[mask]
    # yt = yt[mask]
    # ym = ym[mask]

    print(f"Filtered out {len(Xsim)-len(X)} simulations with variations larger than 60 minutes")

    # set up nearest neighbor regression
    knn = FaissKNeighbors(k=15)

    knn.fit(X,yt)
    Xtest = X[0] * np.random.uniform(0.99,1.01,size=X[0].shape)
    distances, indices = knn.index.search(Xtest.reshape(1,-1), k=8)

    # fit knn with preprocessed data 
    Xnorm = (X - X.mean(0)) / X.std(0)
    knn.fit(Xnorm,yt)
    Xtestnorm = (Xtest - X.mean(0)) / X.std(0)
    distances, indices = knn.index.search(Xtestnorm.reshape(1,-1), k=8)
    
    # yinterp = yt[0].copy() # alloc 
    # # try a linear interpolate over neighbors
    # # loop over each sample + interpolate
    # A = np.vstack([X[indices[0]].T, np.ones(len(indices[0]))]).T
    # for row in range(yt.shape[1]): # interpolate each point in O
    #     # fit system of linear equations to simplex
    #     # solves: Ab = y, where b = (A.T A)^-1 A.T y
    #     b = np.linalg.lstsq(A, yt[indices[0],row], rcond=None)[0]
    #     yinterp[row] = np.sum(Xtest*b[:-1]) + b[-1]



    # plot
    fig,ax = plt.subplots(2,1,figsize=(10,9))
    for i,idx in enumerate(indices[0]):
        ax[0].plot(yt[idx,:],alpha=1-0.02*i,color=plt.cm.inferno(0.02*i),zorder=100-i)
        print(f'pars {X[idx]}')
    #ax[0].plot(yinterp,color='cyan',zorder=1000,lw=3,label='interpolated')
    ax[0].set_title(f'Nearest neighbors to {Xtest}')
    ax[0].plot(yt[0,:],color='limegreen',ls='--',label='test',zorder=1000,lw=3)
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('O-C [min]')
    #ax[0].legend()


    # try NN on OC data
    knn_oc = FaissKNeighbors(k=10)

    # normalize yt
    yt_norm = (yt - yt.min(0)) / (yt.max(0) - yt.min(0))

    # fit knn
    knn_oc.fit(yt, X)

    # test query
    yt_test = yt[0] + np.random.uniform(0,0.5,size=yt[0].shape)
    yt_test_norm = (yt_test - yt.min(0)) / (yt.max(0) - yt.min(0))
    distances, indices = knn_oc.index.search(yt_test.reshape(1,-1), k=1000)

    for i,idx in enumerate(indices[0]):
        if i%5==0:
            # set color based on distance
            vmin = np.percentile(distances[0],5)
            vmax = np.percentile(distances[0],95)
            color = plt.cm.viridis((distances[0][i]-vmin)/(vmax-vmin))
            ax[1].plot(yt[idx,:],label=f'pars {X[idx]}',alpha=1-0.001*i,c=color,zorder=1000-i)
    ax[1].set_title(f'Nearest neighbors to {yt_test}')
    ax[1].plot(yt[0,:],color='cyan',ls='--',label='test',zorder=1010,linewidth=3)
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('O-C [min]')

    plt.tight_layout()
    plt.show()

    # TODO try linear interpolation over neighbors


    # period ratio
    col_ratio = X[:,3] / X[:,1]

    Xstack = np.column_stack((X,col_ratio))

    # mass ratio
    col_ratio = X[:,2] / X[:,4]
    Xstack = np.column_stack((Xstack,col_ratio))

    keys = [
        'M*',
        'P1 [day]',
        'M1 [earth]',
        'P2 [day]',
        'M2 [earth]',
        'omega2 [rad]',
        'e2',
        'Period Ratio',
        'Mass Ratio'
    ]
    loglike = np.exp(-distances/yt.shape[1])[0]

    fig = corner(Xstack[indices[0]],
        labels= keys,
        bins=int(np.sqrt(len(indices[0]))),
        plot_density=False,
        data_kwargs={
            'c':distances,
            'vmin':np.percentile(distances,5),
            'vmax':np.percentile(distances,95),
            'cmap':'viridis',
        },
        label_kwargs={
            'labelpad':15,
        },
        hist_kwargs={
            'color':'black',
        }
    )
    plt.savefig('corner.png',dpi=300)
    plt.show()

