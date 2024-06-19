
import os
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from scipy.stats import rankdata, chi2, norm, gamma

COLORS = ["white","red","green","orange","purple","bisque","blue","darkblue","powderblue","steelblue"]
COLORS2 = ["black","red","green","orange","purple","bisque","blue","darkblue","powderblue","steelblue"]


# ---------------------------------------------------------------------------- #
#                                COMPUTE METRICS                               #
# ---------------------------------------------------------------------------- #

def rank_hist(forecast,true_obs):
    K = forecast.shape[-1]

    print("Minimum rank")
    rank_min = rankdata(
        np.concatenate([true_obs[...,np.newaxis],forecast],axis=-1),
        method="min",
        axis=-1
    )[...,0]
    print("Maximum rank")
    rank_max = rankdata(
        np.concatenate([true_obs[...,np.newaxis],forecast],axis=-1),
        method="max",
        axis=-1
    )[...,0]
    rank = np.random.randint(rank_min,rank_max+1)-1

    if K==107:
        rank = (rank/6).astype(np.int32)
        K=17
    Z = rank/(K+1)
    Z = Z.reshape((Z.shape[0],Z.shape[1]*Z.shape[2]))
    return Z

def JPZ_test(rankhist):
    n_i = rankhist
    N = np.sum(n_i,axis=0)
    J = n_i.shape[0]
    i = np.tile(np.arange(start=1,stop=J+1)[...,np.newaxis],reps=[1,n_i.shape[-1]])
    n_th = np.tile((N/J)[np.newaxis,:],reps=[n_i.shape[0],1])
    x_i = (n_i-n_th)/np.sqrt(n_th)
    
    # Pearson chi^2 test
    X2 = np.sum(np.square(x_i),axis=0)
    
    # Linear contrast
    a = 2*np.sqrt(3/(J**3-J))
    b = -(np.sqrt(3)*J+np.sqrt(3))/np.sqrt(J*(J+1)*(J-1))
    x_lin = a*i+b
    X2_lin = np.sum(x_i*x_lin,axis=0)**2
    
    # Squared contrast
    a = 6*np.sqrt(5/(J**5-5*J**3+4*J))
    b = -1/2*(np.sqrt(5)*J**2-np.sqrt(5))/(np.sqrt((J-2)*(J-1)*J*(J+1)*(J+2)))
    x_u = a*(i-(J+1)/2)**2+b
    X2_u = np.sum(x_i*x_u,axis=0)**2
    
    # Wave contrast
    x_wave = np.sin(2*np.pi*(i/(J+1)))
    X2_wave = np.sum(x_i*x_wave,axis=0)**2
    
    df = pd.DataFrame({"X2":X2,"pearson":1-chi2.cdf(x=X2,df=J-1),
                      "X2_lin":X2_lin,"linear":1-chi2.cdf(x=X2_lin,df=J-1),
                      "X2_u":X2_u,"squared":1-chi2.cdf(x=X2_u,df=J-1),
                      "X2_wave":X2_wave,"wave":1-chi2.cdf(x=X2_wave,df=J-1)})
    return df

def quantile_gtcnd(p,rho_L,mu,rho_sigma):
    L = 1 / (1 + np.exp(-rho_L))
    sigma = np.exp(rho_sigma)
    cdf = norm.cdf(x=-mu/sigma,loc=0,scale=1)
    return np.where(p<=L,
                    np.zeros_like(p),
                    mu+sigma*norm.ppf(q=(p-L)*(1-cdf)/(1-L)+cdf,loc=0,scale=1)
                )

def quantile_csgd(p,rho_k,rho_theta,rho_delta):
    k = np.exp(rho_k)
    theta = np.exp(rho_theta)
    delta = - k*theta / (1 + np.exp(-rho_delta))

    qf = gamma.ppf(q=p, a=k, loc=delta, scale=theta)
    return np.where(qf<=0, np.zeros_like(p), qf)


# ---------------------------------------------------------------------------- #
#                                 PLOT METRICS                                 #
# ---------------------------------------------------------------------------- #

def plot_grid_rank_hist(data_list,JPZ_list,name_list,path,file_name,r=3,c=3,K=17):
    
    fig, axs = plt.subplots(r,c,figsize=(c*5,r*5))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.2)
    
    for p,data in enumerate(data_list):
        bp = axs[p//c,p%c].boxplot(data,patch_artist=True,
                                widths=1.0,
                                showfliers=False)
        for patch in bp["boxes"]:
            patch.set(facecolor=COLORS[p%len(COLORS)])
        for median in bp['medians']:
            median.set_color('black')
        for whisker in bp['whiskers']:
            whisker.set_linestyle('--')
        axs[p//c,p%c].plot([.5,K+1.5],[1/(K+1)]*2,"k--")
        rate_JPZ = np.count_nonzero(JPZ_list[p]["pearson"]>0.05)/len(JPZ_list[p])*100
        axs[p//c,p%c].text((K+2)/2,0.15,f"JPZ test : {round(rate_JPZ)}%",ha='center', va='center',fontsize=15)
        axs[p//c,p%c].grid(False)
        axs[p//c,p%c].set_title(f"{name_list[p][0]}",fontsize=20,y=1.11,pad=-14)
        axs[p//c,p%c].text((K+2)/2,0.20625,name_list[p][1],ha='center', va='center',fontsize=9)
        if p==0 or p==1:
            axs[p//c,p%c].set_xticks([i+1 for i in range(K+1)])
            axs[p//c,p%c].set_xticklabels([str(i+1)  if i%2==0 else "" for i in range(K+1)])
        else:
            axs[p//c,p%c].set_xticks([i+1 for i in range(K+1)])
            axs[p//c,p%c].set_xticklabels([str(i+1)  if i%2==0 else "" for i in range(K+1)])
        axs[p//c,p%c].axis(ymin=0.,ymax=.2)

    plt.savefig(os.path.join(path,file_name))
    plt.close()
    return None
    

