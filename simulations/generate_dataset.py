import glob
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

from nbody.tools import lomb_scargle, make_sin, msun, mearth, mjup

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    help_ = "Input pickle file of simulations"
    parser.add_argument("-i", "--input", help=help_)
    help_ = "Pickle file to write X,y data to"
    parser.add_argument("-o", "--output", help=help_)
    args = parser.parse_args()


    #data = pickle.load(open('ttv100e.pkl','rb'))
    data = pickle.load(open(args.input,'rb'))

    X, y, z = [], [], []

    for i in range(len(data)):
        ttv_data = data[i]
        if ttv_data['planets'][0]['ttv'].shape[0] < 30 or ttv_data['planets'][0]['max']*24*60 > 10 or ttv_data['planets'][0]['max']*24*60 < 0.5:
            #print(i)
            continue 

        if i%100 ==0:
            print(i)
        xn = [] 
        xn.append( ttv_data['mstar'] )
        xn.append( ttv_data['planets'][0]['P'] )
        xn.append( ttv_data['planets'][0]['mass']*msun/mearth )
        xn.append( ttv_data['planets'][1]['P'] )
        xn.append( ttv_data['planets'][1]['mass']*msun/mearth )
        xn.append( ttv_data['objects'][2]['omega'] )
        xn.append( ttv_data['planets'][1]['e'] )

        X.append( xn )
        z.append( (ttv_data['planets'][0]['ttv']*24*60)[:30] )

        #Y = np.fft.fft( z[-1][:100] )
        #max2min = np.argsort(np.abs(Y[:50]))[::-1]

        #Ys = np.copy(Y)
        #Ys[max2min[10:]] = 0
        #Ys[-1*max2min[10:]] = 0

        # fit sin waves to data 
        #t = np.arange( ttv_data['planets'][0]['ttv'].shape[0] )
        #frequency, power, fdata = lomb_scargle(t, ttv_data['planets'][0]['ttv']*24*60, npeaks=3, minfreq=1./100)

        #plt.plot(np.fft.ifft(Ys),'g-' ); 
        #plt.plot(ttv_data['planets'][0]['ttv']*24*60,'ko'); 
        #plt.plot( t, make_sin(t,fdata),'r-' )

        #freq = max2min[:4]
        #real = np.hstack([ Y[freq].real, Y[-1*freq].real ])
        #imag = np.hstack([ Y[freq].imag, Y[-1*freq].imag ])

        #y.append( np.hstack([freq,real,imag]) ) 
        
        #y.append(fdata.flatten())
    
    X = np.array(X)
    z = np.array(z)
    #y = np.array(y)

    pickle.dump([X,z], open(args.output,'wb'))