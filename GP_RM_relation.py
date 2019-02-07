import matplotlib.pyplot as plt
import numpy as np
import pickle 
import pandas

import GPy

#G = 6.678e-11 # [m3 kg-1 s-2]
#mjup = 1.898e27 # kg
#rjup = 69.911e6 # m 
#msun = 1.989e30 # kg
#rsun = 695.508e6 # m 
#AU = 1.496e11 # m 
#day2sec = 24*60*60 # s

def get_data(keys=['pl_pnum','pl_radj','pl_orbper','pl_ratdor','st_mass', 'pl_bmassj']):
    # query the api for the newest data
    api_url = 'https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=exoplanets&format=json&select='
    query_url = api_url+','.join(keys)
    return pandas.read_json( query_url )

def filter_data(data,filters):
    if isinstance(filters,list):
        mask = filters[0](data)
        for i in range(1,len(filters)):
            mask = np.logical_and(mask, filters[i](data) )
        return data[mask]

if __name__ == "__main__":
    
    data = get_data(keys=['pl_pnum','pl_radj','pl_orbper','pl_ratdor','st_mass', 'pl_bmassj','pl_radjerr1','pl_radjerr2'])

    filters = [
        lambda x: ~x.isnull().any(axis=1),
        lambda x: x.st_mass<2,
        lambda x: x.pl_orbper<100,
        lambda x: x.pl_orbper>1,
    ]

    data = filter_data(data, filters)

    X = {
        '1d':data.get(['pl_radj']).values,
        '2d':data.get(['pl_radj','pl_orbper']).values,
    }

    y = np.log10(data['pl_bmassj']).values.reshape(-1,1)


    # create simple GP model
    m = { 
        '1d':GPy.models.GPRegression(X['1d'],y, GPy.kern.RBF(X['1d'].shape[1],ARD=True) ),
        '2d':GPy.models.GPRegression(X['2d'],y, GPy.kern.RBF(X['2d'].shape[1],ARD=True) ),
    }

    # train the GP models with data
    for k in m.keys():
        m[k].optimize(messages=True,max_f_eval = 1000)

    # compute error in model 
    pred = {}; cov={}
    for k in m.keys():
        pred[k],cov[k] = m[k].predict(X[k])
        pred[k] = pred[k].reshape(-1)
        cov[k] = cov[k].reshape(-1)**0.5

    std = 0.5*(data['pl_radjerr1'].values + np.abs(data['pl_radjerr2']).values)

    # deviation from prediction in terms of prediction uncertainty
    sigma={}
    for k in pred.keys():
        sigma[k] = (pred[k]-y.reshape(-1))/cov[k]
    
        print('mean sigma :{:.2f}'.format(np.abs(sigma[k]).mean()),k )

    plt.subplots_adjust(top=0.95,bottom=0.06,left=0.07,right=0.96)
    ax = [
        plt.subplot2grid((3, 4), (0, 0), colspan=3,rowspan=3),
        plt.subplot2grid((3, 4), (0, 3)),
        plt.subplot2grid((3, 4), (1, 3)),
        plt.subplot2grid((3, 4), (2, 3))
    ]

    for k in pred.keys():

        #ax[0].errorbar( pred[less1],data['pl_radj'][less1], fmt='.', xerr=cov[less1], label='Estimate (Err<1)',alpha=0.5,color='green',errorevery=10)
        ax[3].hist( (pred[k]-y.reshape(-1)),bins=np.linspace(-2,2,20),alpha=0.5,label=k )         
    
    
    #ax[0].plot( pred['forecaster'],data['pl_radj'], lw=0,marker='.', label='{}'.format('forecaster'),alpha=0.5)

    less1 = np.abs(sigma['2d']) < 1
    more1 = np.abs(sigma['2d']) > 1

    ax[0].errorbar( pred['2d'][less1],data['pl_radj'][less1], marker='.',lw=0, label='{}'.format('2d'),alpha=0.5)
    #ax[0].errorbar( pred['1d'],data['pl_radj'], xerr=cov['1d'], label='{} Estimate'.format('1d'),alpha=0.5)
    
    sidx = np.argsort(data['pl_radj'].values)
    ax[0].plot( pred['1d'][sidx],data['pl_radj'].values[sidx],lw=2, label='{}'.format('1d'),alpha=0.5)

    ax[0].plot( np.log10(data['pl_bmassj']), data['pl_radj'],'ko',label='Measured',alpha=0.5 )
    ax[1].plot( np.log10(data['pl_bmassj']), data['pl_radj'],'k.',label='Measured',alpha=0.5 )


    #ax[2].plot( pred['forecaster'],data['pl_radj'], lw=0,marker='.', label='{}'.format('forecaster'),alpha=0.5)
    ax[2].errorbar( pred['2d'][less1],data['pl_radj'][less1], marker='.',lw=0, label='{}'.format('2d'),alpha=0.5)
    sidx = np.argsort(data['pl_radj'].values)
    ax[2].plot( pred['1d'][sidx],data['pl_radj'].values[sidx],lw=2, label='{}'.format('1d'),alpha=0.5)
    
    ax[0].legend(loc='best')
    ax[3].legend(loc='best')
    ax[0].set_xlabel('Log10( Mass [Jupiter] )')
    ax[0].set_ylabel('Radius [Jupiter]')
    ax[0].set_xlim([-2,1.5])
    ax[0].set_ylim([0,2])
    
    ax[1].set_xlabel('Log10( Mass [Jupiter] )')
    ax[1].set_ylabel('Radius [Jupiter]')
    
    ax[2].set_xlabel('Log10( Mass [Jupiter] )')
    ax[2].set_ylabel('Radius [Jupiter]')
    

    plt.show()
    
    dude()
    
    


    ############################## PREDICT MASSES OF NEW PLANETS 


    # remove anything that does not a full set of data 
    nulls = ~data.isnull().any(axis=1)
    train_data = data[nulls]
    single = train_data['pl_pnum']==1
    lowmass = train_data.st_mass<3
    shortper = train_data.pl_orbper<15
    train_data = train_data[single].drop(['pl_pnum'],axis=1)[shortper]

    # find transits without mass measurements
    newdata = get_data(keys=['pl_pnum','pl_radj','pl_orbper','pl_ratdor','st_mass','st_teff'])
    nulls = ~newdata.isnull().any(axis=1)
    newdata = newdata[nulls]
    single = newdata['pl_pnum']==1
    shortper = newdata.pl_orbper<15
    lowmass = newdata.st_mass<3
    AND = np.logical_and(shortper,lowmass)
    newdata = newdata[single].drop(['pl_pnum'],axis=1)[AND]

    nomass = list(set(newdata.index.values) - set(train_data.index.values))
    newdata = newdata.loc[nomass]

    ############################################################
    pred,cov = m.predict(newdata.values)
    pred = pred.reshape(-1)
    cov = cov.reshape(-1)**0.5
    ax[0].errorbar( pred, np.log10(newdata['pl_radj']), fmt='.', xerr=cov, label='Estimate',alpha=0.5,color='black')

    f,ax = plt.subplots(2)    
    ax[0].errorbar( pred,newdata['pl_radj'], fmt='.', xerr=cov, label='Estimate',alpha=0.5,color='black')
    ax[0].set_xlabel('Log10( Mass [Jupiter] )')
    ax[0].set_ylabel('Radius [Jupiter]')
    ax[0].set_title('New mass estimates')

    ax[1].hist(cov,bins=np.linspace(0.3,0.5,10))
    ax[1].set_xlabel('Size of uncertainty [Log10( Mass [Jupiter]]')
    plt.show()



    f = plt.figure()
    figure = corner.corner(train_data.values,
                            labels=['Planet Mass (Jup)','Period (Day)','Radius (Jup)','a/R*','Star Mass (Solar)'],
                            range=[ (-2,1.5),(0,15),(0,2.25),(0,30),(0,3) ], 
                            plot_contours=False,
                            plot_density=False,
                            data_kwargs={'alpha':0.5},
                            fig=f
                            )
    plt.show()
