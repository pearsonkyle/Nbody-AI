import numpy as np
import pickle
import matplotlib.pyplot as plt

from nbody.tools import make_sin 

if __name__ == "__main__":
    X,y,z = pickle.load(open('X7_y20_z.pkl','rb'))

    error = np.zeros(X.shape[0])
    
    for i in range(X.shape[0]):

        Y = np.fft.fft( z[i][:100] )

        py = np.copy(Y)
        py[:] = 0 
        n = 4
        freq = y[i][:n].astype(int)
        py[freq] = y[i][n:2*n] + 1j*y[i][3*n:4*n]
        py[-1*freq] = y[i][2*n:3*n] + 1j*y[i][4*n:5*n]

        ym = np.fft.ifft(py)

        res = z[i][:100]-ym
        error[i] = np.mean(np.abs(res))
    
    plt.hist(error, bins=int(np.sqrt(error.shape[0])) )
    plt.xlabel("Average Error |OC - 3D Sin series| [min]")
    plt.show()

    plt.plot(z[i],'k-'); plt.plot(ym,'r-');plt.show()
    plt.scatter(X[:,4],X[:,3]/X[:,1],c=error); plt.colorbar(); plt.show()