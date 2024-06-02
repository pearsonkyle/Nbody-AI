import sys
sys.path.append('../')
from nbody.tools import msun, mearth, mjup
import glob
import pickle
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

import torch
from torch import nn
import torch.functional as F
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, TensorDataset

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    help_ = "Input pickle file of simulations"
    parser.add_argument("-i", "--input", help=help_)
    args = parser.parse_args()

    # Load N-body simulations
    Xsim,ysim = pickle.load(open(args.input,'rb'))
    # Xsim: mstar [msun], P1 [day], m1 [mearth], P2 [day], m2 [mearth], omega [rad], ecc
    # ysim: simulation time of mid-transit [min]

    # find lengths of each y vector
    lengths = np.array([len(yi) for yi in ysim])

    # find median length
    median_length = np.median(lengths)

    # loop over list and extract same length vectors
    y = np.zeros((len(ysim),int(median_length)))
    mask = np.zeros(len(ysim),dtype=bool)
    for i in range(len(ysim)):
        if lengths[i] >= median_length:
            y[i,:] = ysim[i][:int(median_length)]
            mask[i] = True
        else:
            mask[i] = False

    # mask out bad simulations
    X = np.array(Xsim)
    X = X[mask]
    y = y[mask]

    print('X shape:',X.shape)
    print('y shape:',y.shape)

    # compute ephemeris residuals from simulation (similar to O-C in minutes)
    yt = np.zeros(y.shape) 
    linear_coeffs = np.zeros((y.shape[0],2))
    # loop over each simulation and subtract out a linear fit
    print('Subtracting linear fit...')
    for i in tqdm(range(y.shape[0])):
        linear_coeffs[i], epochs = np.polyfit(np.arange(y.shape[1]),y[i,:],1),np.arange(y.shape[1])
        yt[i,:] = y[i,:] - np.polyval(linear_coeffs[i],epochs)

    # filter out simulations with variations larger than 60 minutes
    mask = (np.abs(yt).max(1) < 10) & (np.abs(yt).max(1) > 0.75)
    X = X[mask]
    y = y[mask]

    print(f"Filtered {len(mask)-np.sum(mask)} simulations with variations larger than 60 minutes")

    # preprocess inputs by scaling each feature between min and max
    feature_min = np.min(X,axis=0)
    feature_max = np.max(X,axis=0)
    for i in range(X.shape[1]):
        X[:,i] = (X[:,i] - feature_min[i])/(feature_max[i] - feature_min[i])


    # compute the Lomb-Scargle periodogram + amplitude
    sin_coeffs = np.zeros((y.shape[0],5))
    max_periods = np.zeros((y.shape[0],2))
    err = np.zeros(y.shape[0]) # average error per measurement
    sin_approx = np.zeros(y.shape)
    
    for i in tqdm(range(y.shape[0])):
        # search for periodic signals in O-C residuals
        freq,power = LombScargle(epochs, yt[i]).autopower(nyquist_factor=.5)

        # period of highest power
        mi = np.argmax(power)
        per = 1./freq[mi]
        max_periods[i,0] = per
            
        # find best fit signal with 1 period
        # construct basis vectors with sin and cos
        basis = np.ones((3, len(epochs)))
        basis[0] = np.sin(2*np.pi*epochs/per)
        basis[1] = np.cos(2*np.pi*epochs/per)
        # fit for the coefficients
        sin_coeffs_1 = np.linalg.lstsq(basis.T, yt[i], rcond=None)[0]

        # reconstruct signal
        y_bestfit = np.dot(basis.T, sin_coeffs_1)
        # plt.plot(epochs, yt[i],'ko'); plt.plot(epochs, y_bestfit,'r-'); plt.show()
        y_residual = yt[i] - y_bestfit

        # search for another periodic signal
        freq2,power2 = LombScargle(epochs, y_residual).autopower(nyquist_factor=.5)
        # plt.semilogx(1./freq, power,'r-'); plt.semilogx(1./freq2, power2,'g-'); plt.show()

        # period of highest power
        mi2 = np.argmax(power2)
        per2 = 1./freq2[mi2]
        max_periods[i,1] = per2

        # find best fit signal with 2 periods
        # construct basis vectors with sin and cos
        basis2 = np.ones((5, len(epochs)))
        basis2[0] = np.sin(2*np.pi*epochs/per)
        basis2[1] = np.cos(2*np.pi*epochs/per)
        basis2[2] = np.sin(2*np.pi*epochs/per2)
        basis2[3] = np.cos(2*np.pi*epochs/per2)
        # fit for the coefficients
        sin_coeffs[i] = np.linalg.lstsq(basis2.T, yt[i], rcond=None)[0]

        # reconstruct signal
        sin_approx[i] = np.dot(basis2.T, sin_coeffs[i])

        # compute err
        err[i] = np.mean(np.abs(yt[i] - sin_approx[i]))

    # print average error
    print('Median Error: {:.2f} min'.format(np.median(err)))
    print('Lower Quartile Error: {:.2f} min'.format(np.percentile(err,25)))
    print('Upper Quartile Error: {:.2f} min'.format(np.percentile(err,75)))
    # something like the inaccuracy of the sin approximation
    Xog = X.copy()
    # scale X back to original range
    for i in range(X.shape[1]):
        Xog[:,i] = X[:,i]*(feature_max[i]-feature_min[i])+feature_min[i]

    period_ratio = Xog[:,3]/Xog[:,1]

    dude()

    # combine periods and sin_coeffs
    ycombine = np.hstack((max_periods,sin_coeffs))

    # alloc new scaled array of y
    ynew = np.zeros(ycombine.shape)

    # scale each column by min and max
    reverse_scale_fn = []
    for i in range(ynew.shape[1]):
        ynew[:,i] = (ycombine[:,i]-np.min(ycombine[:,i]))/(np.max(ycombine[:,i])-np.min(ycombine[:,i]))
        # lambda function to reverse scaling
        reverse_scale_fn.append(lambda x, i=i: x*(np.max(ycombine[:,i])-np.min(ycombine[:,i]))+np.min(ycombine[:,i]))

    # assert reverse scaling works
    assert np.allclose(ycombine, np.array([reverse_scale_fn[i](ynew[:,i]) for i in range(ynew.shape[1])]).T)

    # transform X to fourier series basis with random frequencies   
    nfreq = 10
    Xnew = np.zeros((X.shape[0],2*nfreq*X.shape[1]))
    rperiods = np.random.random(nfreq)*10+1
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Xnew[i,2*nfreq*j:2*nfreq*(j+1)] = np.hstack((np.sin(2*np.pi*X[i,j]/rperiods),np.cos(2*np.pi*X[i,j]/rperiods)))

    # create test and train data [input_pars, sin_pars -> O-C data]
    X_train, X_test, y_train, y_test = train_test_split(Xnew, np.hstack([ynew, yt]), test_size=0.2, random_state=42)

    # set up DataLoaders
    train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    test_dataset = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # create pytorch MLP model
    class MLP(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(MLP, self).__init__()
            self.linear_layers = [
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, output_dim),
                nn.Sigmoid(),
            ]
            self.layers = nn.Sequential(*self.linear_layers)

        def forward(self, x):
            return self.layers(x)


    # create model
    model = MLP(X_train.shape[1], ynew.shape[1])
    loss_fn = nn.MSELoss()
    # percent loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def train(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        model.train()
        
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            pars = y[:,:ycombine.shape[1]].clone()
            spars = y[:,:ycombine.shape[1]].clone()
            oc = y[:,ycombine.shape[1]:]

            # Compute prediction error
            pred = model(X) # sin coefficients

            # scale pars back to original space
            for i in range(pars.shape[1]):
                spars[:,i] = reverse_scale_fn[i](pars[:,i])

            per = spars[:,0]
            per2 = spars[:,1]
            # construct basis vectors with sin and cos
            basis = torch.ones((X.shape[0], 5, len(epochs)))
            basis[:,0] = torch.sin(2*np.pi*epochs/per.reshape(-1,1))
            basis[:,1] = torch.cos(2*np.pi*epochs/per.reshape(-1,1))
            basis[:,2] = torch.sin(2*np.pi*epochs/per2.reshape(-1,1))
            basis[:,3] = torch.cos(2*np.pi*epochs/per2.reshape(-1,1))

            # reconstruct signal
            sin_approx = torch.zeros((X.shape[0], len(epochs)))
            for i in range(X.shape[0]):
                sin_approx[i] = torch.sum(basis[i]*spars[i,2:].reshape(-1,1), axis=0)

            # create OC loss
            #loss = loss_fn(sin_approx, oc)
            loss = loss_fn(pred, pars) #+ loss_fn(sin_approx, oc)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 80 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test(dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():

            for batch, (X, y) in enumerate(dataloader):
                X, y = X.to(device), y.to(device)
                pars = y[:,:ycombine.shape[1]].clone()
                spars = y[:,:ycombine.shape[1]].clone()
                oc = y[:,ycombine.shape[1]:]

                # Compute prediction error
                pred = model(X) # sin coefficients

                # scale pars back to original space
                for i in range(pars.shape[1]):
                    spars[:,i] = reverse_scale_fn[i](pars[:,i])

                per = spars[:,0]
                per2 = spars[:,1]
                # construct basis vectors with sin and cos
                basis = torch.ones((X.shape[0], 5, len(epochs)))
                basis[:,0] = torch.sin(2*np.pi*epochs/per.reshape(-1,1))
                basis[:,1] = torch.cos(2*np.pi*epochs/per.reshape(-1,1))
                basis[:,2] = torch.sin(2*np.pi*epochs/per2.reshape(-1,1))
                basis[:,3] = torch.cos(2*np.pi*epochs/per2.reshape(-1,1))

                # reconstruct signal
                sin_approx = torch.zeros((X.shape[0], len(epochs)))
                for i in range(X.shape[0]):
                    sin_approx[i] = torch.sum(basis[i]*spars[i,2:].reshape(-1,1), axis=0)

                # create OC loss
                #test_loss = loss_fn(sin_approx, oc)
                test_loss = loss_fn(pred, pars)

        test_loss /= num_batches
        #correct /= size
        print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")


    for t in range(15):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_loader, model, loss_fn, optimizer)
        test(test_loader, model, loss_fn)
    print("Done!")

    # predict on test data
    y_pred = model(torch.from_numpy(X_test).float().to(device)).cpu().detach().numpy()

    # import randomforest
    # reg = RandomForestRegressor(max_depth=10, random_state=0)
    # reg.fit(X, ynew)
    # y_pred = reg.predict(X_test)

    # scale back to original values
    y_pred_orig = np.zeros(y_pred.shape)
    y_pred_sine = np.zeros((y_pred.shape[0], len(epochs)))
    pred_err = np.zeros(y_pred_orig.shape[0])

    for i in range(y_pred.shape[1]):
        y_pred_orig[:,i] = reverse_scale_fn[i](y_pred[:,i])

    # loop over each sample and calc sine approximation
    for i in range(y_pred_orig.shape[0]):
        per = y_pred_orig[i,0]
        per2 = y_pred_orig[i,1]
        # construct basis vectors with sin and cos
        basis = np.ones((5, len(epochs)))
        basis[0] = np.sin(2*np.pi*epochs/per)
        basis[1] = np.cos(2*np.pi*epochs/per)
        basis[2] = np.sin(2*np.pi*epochs/per2)
        basis[3] = np.cos(2*np.pi*epochs/per2)

        # reconstruct signal
        y_pred_sine[i] = np.dot(basis.T, y_pred_orig[i,2:])

        # calc prediction error
        pred_err[i] = np.mean(np.abs(y_pred_sine[i] - y_test[i,7:]))

    # compute error
    print('Median Error: {:.2f} min'.format(np.median(pred_err)))
    print('Lower Quartile Error: {:.2f} min'.format(np.percentile(pred_err,25)))
    print('Upper Quartile Error: {:.2f} min'.format(np.percentile(pred_err,75)))

    # plot results
    fig,ax = plt.subplots(2,figsize=(10,10))
    #random index
    ax[0].plot(epochs, sin_approx[0],'r-',label='sin fit')
    ax[0].plot(epochs, y_pred_sine[0],'g-',label='prediction')
    ax[0].plot(epochs, yt[0],'ko',label='data')
    idx = np.random.randint(0,y_test.shape[0])
    ax[1].plot(epochs, sin_approx[idx],'r-',label='sin fit')
    ax[1].plot(epochs, y_pred_sine[idx],'g-',label='prediction')
    ax[1].plot(epochs, yt[idx],'ko',label='data')
    plt.show()