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

    # preprocess inputs by scaling each feature between min and max
    X = np.array(Xsim)
    for i in range(X.shape[1]):
        X[:,i] = (X[:,i]-np.min(X[:,i]))/(np.max(X[:,i])-np.min(X[:,i]))
    
    # find lengths of each y vector
    lengths = np.array([len(yi) for yi in ysim])

    # find median length
    median_length = np.median(lengths)

    # loop over list and extract same length vectors
    y = np.zeros((len(X),int(median_length)))
    mask = np.zeros(len(X),dtype=bool)
    for i in range(len(X)):
        if lengths[i] >= median_length:
            y[i,:] = ysim[i][:int(median_length)]
            mask[i] = True
        else:
            mask[i] = False

    # mask out bad simulations
    X = X[mask]
    y = y[mask]

    yt = np.zeros(y.shape) # will be residuals (similar to O-C in minutes)
    linear_coeffs = np.zeros((y.shape[0],2))
    # loop over each simulation and subtract out a linear fit
    for i in range(y.shape[0]):
        linear_coeffs[i], epochs = np.polyfit(np.arange(y.shape[1]),y[i,:],1),np.arange(y.shape[1])
        yt[i,:] = y[i,:] - np.polyval(linear_coeffs[i],epochs)

    print('X shape:',X.shape)
    print('y shape:',y.shape)

    # filter out simulations with variations larger than 60 minutes
    mask = (np.abs(yt).max(1) < 60) & (np.abs(yt).max(1) > 0.5)
    X = X[mask]
    y = y[mask]
    yt = yt[mask]

    print(f"Filtered out {len(mask)-np.sum(mask)} simulations with variations larger than 60 minutes")

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
        # TODO implement regularization

        # reconstruct signal
        sin_approx[i] = np.dot(basis2.T, sin_coeffs[i])

        # compute err
        err[i] = np.mean(np.abs(yt[i] - sin_approx[i]))

    # print average error
    print('Median Error: {:.2f} min'.format(np.median(err)))
    print('Lower Quartile Error: {:.2f} min'.format(np.percentile(err,25)))
    print('Upper Quartile Error: {:.2f} min'.format(np.percentile(err,75)))
    # something like the inaccuracy of the sin approximation

    # combine periods and sin_coeffs
    ycombine = np.hstack((max_periods,sin_coeffs))

    # clamp periods between 0 and 50
    ycombine[:,0] = np.clip(ycombine[:,0],0,50)
    ycombine[:,1] = np.clip(ycombine[:,1],0,50)

    # alloc new scaled array of y
    ynew = np.zeros(ycombine.shape)

    # scale each column to have 0 mean and unit variance
    reverse_scale_fn = []
    for i in range(ynew.shape[1]):
        ynew[:,i] = (ycombine[:,i] - ycombine[:,i].mean())/ycombine[:,i].std()
        reverse_scale_fn.append(lambda x, i=i: x*ycombine[:,i].std() + ycombine[:,i].mean())

        #ynew[:,i] = (ycombine[:,i]-np.min(ycombine[:,i]))/(np.max(ycombine[:,i])-np.min(ycombine[:,i]))
        # lambda function to reverse scaling
        #reverse_scale_fn.append(lambda x, i=i: x*(np.max(ycombine[:,i])-np.min(ycombine[:,i]))+np.min(ycombine[:,i]))


    # # assert reverse scaling works
    # assert np.allclose(ycombine, np.array([reverse_scale_fn[i](ynew[:,i]) for i in range(ynew.shape[1])]).T)

    # transform X to fourier series basis with random frequencies   
    nfreq = 100
    Xnew = np.zeros((X.shape[0],2*(nfreq)*X.shape[1]))
    rperiods = np.random.random(nfreq)*20
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Xnew[i,2*j] = np.sin(2*np.pi*epochs[j]/rperiods).sum()
            Xnew[i,2*j+1] = np.cos(2*np.pi*epochs[j]/rperiods).sum()

    # add X to Xnew
    Xnew = np.hstack((Xnew,X))

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
        def __init__(self, input_dim, output_dim, rescale_fn, epochs):
            super(MLP, self).__init__()
            self.linear_layers = [
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, output_dim),
                nn.Sigmoid(),
            ]
            self.layers = nn.Sequential(*self.linear_layers)
            self.rescale_fn = rescale_fn
            # convert epochs to tensor
            self.epochs = torch.from_numpy(epochs).float()

        def forward(self, x):
            pars = self.layers(x)
            
            # rescale output
            spars = torch.zeros(pars.shape)
            for i in range(pars.shape[1]):
                spars[:,i] = self.rescale_fn[i](pars[:,i])

            per = spars[:,0]
            per2 = spars[:,1]

            # construct basis vectors with sin and cos
            basis = torch.ones((x.shape[0], 5, len(self.epochs)))
            basis[:,0] = torch.sin(2*np.pi*self.epochs/per.reshape(-1,1))
            basis[:,1] = torch.cos(2*np.pi*self.epochs/per.reshape(-1,1))
            basis[:,2] = torch.sin(2*np.pi*self.epochs/per2.reshape(-1,1))
            basis[:,3] = torch.cos(2*np.pi*self.epochs/per2.reshape(-1,1))

            # reconstruct signal
            sin_approx = torch.zeros((x.shape[0], len(self.epochs)))
            for i in range(x.shape[0]):
                sin_approx[i] = torch.matmul(basis[i].T, spars[i,2:])

            return sin_approx

        def backward(self, x):
            pars = self.layers(x)
            
            # rescale output
            spars = torch.zeros(pars.shape)
            for i in range(pars.shape[1]):
                spars[:,i] = self.rescale_fn[i](pars[:,i])

            per = spars[:,0]
            per2 = spars[:,1]

            # construct basis vectors with sin and cos
            basis = torch.ones((x.shape[0], 5, len(self.epochs)))
            basis[:,0] = torch.cos(2*np.pi*self.epochs/per.reshape(-1,1))
            basis[:,1] = -torch.sin(2*np.pi*self.epochs/per.reshape(-1,1))
            basis[:,2] = torch.cos(2*np.pi*self.epochs/per2.reshape(-1,1))
            basis[:,3] = -torch.sin(2*np.pi*self.epochs/per2.reshape(-1,1))

            # reconstruct signal
            sin_approx = torch.zeros((x.shape[0], len(self.epochs)))
            for i in range(x.shape[0]):
                sin_approx[i] = torch.matmul(basis[i].T, spars[i,2:])

            return sin_approx

    # create model
    model = MLP(X_train.shape[1], ynew.shape[1], reverse_scale_fn, epochs)
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

            # estimate O-C
            pred = model(X)
            pred_pars = model.layers(X)

            # create OC loss
            loss = loss_fn(pred, oc) + loss_fn(pred_pars, pars)
            #loss =  #+ loss_fn(sin_approx, oc)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 10 == 0:
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

                # estimate O-C
                pred = model(X)

                # create OC loss
                test_loss = loss_fn(pred, oc)

        test_loss /= num_batches
        #correct /= size
        print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")


    for t in range(15):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_loader, model, loss_fn, optimizer)
        test(test_loader, model, loss_fn)
    print("Done!")

    # predict on test data
    y_pred = model.forward(torch.from_numpy(X_test).float().to(device)).cpu().detach().numpy()

    pred_err = np.zeros(y_pred.shape[0])
    for i in range(y_pred.shape[0]):
        pred_err[i] = np.mean(np.abs((y_pred[i] - y_test[i,7:])))

    # reconstruct sin fit from parameters in y_test
    test_sin = np.zeros((y_test.shape[0], len(epochs)))
    test_pars = np.zeros((y_test.shape[0], 7))
    for i in range(y_test.shape[0]):
        # scale test parameters to original
        for j in range(7):
            test_pars[i,j] = reverse_scale_fn[j](y_test[i,j])
        # reconstruct sin fit
        per = test_pars[i,0]
        per2 = test_pars[i,1]
        basis = np.ones((5, len(epochs)))
        basis[0] = np.sin(2*np.pi*epochs/per)
        basis[1] = np.cos(2*np.pi*epochs/per)
        basis[2] = np.sin(2*np.pi*epochs/per2)
        basis[3] = np.cos(2*np.pi*epochs/per2)
        test_sin[i] = np.matmul(basis.T, test_pars[i,2:])


    # compute error
    print('Median Error: {:.2f} min'.format(np.median(pred_err)))
    print('Lower Quartile Error: {:.2f} min'.format(np.percentile(pred_err,25)))
    print('Upper Quartile Error: {:.2f} min'.format(np.percentile(pred_err,75)))

    # plot results
    fig,ax = plt.subplots(2,figsize=(10,10))
    #random index
    ax[0].plot(epochs, test_sin[0],'r-',label='sin fit to data')
    ax[0].plot(epochs, y_pred[0],'g-',label='NN prediction')
    ax[0].plot(epochs, y_test[0,7:],'ko',label='data')
    ax[0].legend()
    idx = np.random.randint(0,y_test.shape[0])
    ax[1].plot(epochs, test_sin[idx],'r-',label='sin fit to data')
    ax[1].plot(epochs, y_pred[idx],'g-',label='NN prediction')
    ax[1].plot(epochs, y_test[idx,7:],'ko',label='data')
    ax[1].legend()
    plt.show()