import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm, gamma
from sklearn.metrics import roc_curve

COLORS = ["black","red","green","orange","purple","bisque","blue","darkblue","powderblue","steelblue"]

# ---------------------------------------------------------------------------- #
#                                COMPUTE METRICS                               #
# ---------------------------------------------------------------------------- #

def quantile_to_F(q_pred,y_obs,threshold):
    return np.mean(q_pred>threshold,axis=-1).reshape(-1), (y_obs>threshold).astype(np.float32).reshape(-1)

def obs_to_F(y_obs,threshold):
    return (y_obs>threshold).astype(np.float32).reshape(-1)

def cdf_gtcnd(z,rho_L,mu,rho_sigma):
    L = 1 / (1 + np.exp(-rho_L))
    sigma = np.exp(rho_sigma)
    return L*(z>=0)+(1-L)*truncnorm.cdf(x=z,loc=mu,scale=sigma,a=-mu/sigma,b=np.ones_like(mu)*np.inf)

def cdf_csgd(z,rho_k,rho_theta,rho_delta):
    k = np.exp(rho_k)
    theta = np.exp(rho_theta)
    delta = - k*theta / (1 + np.exp(-rho_delta))
    return (z>=0)*gamma.cdf(x=z,a=k,loc=delta,scale=theta)

def group_raw(X):
    return np.concatenate((X[:,:-2,:-2], X[:,1:-1,:-2],X[:,2:,:-2],
                           X[:,:-2,1:-1], X[:,1:-1,1:-1],X[:,2:,1:-1],
                           X[:,:-2,2:], X[:,1:-1,2:],X[:,2:,2:]),axis=-1)

# ---------------------------------------------------------------------------- #
#                                 PLOT METRICS                                 #
# ---------------------------------------------------------------------------- #

def plot_grid_roc(data,y_true,thresholds,path,file_name,r=3,c=3,lw=2):
    figs, axs = plt.subplots(r,c,figsize=(c*5,r*5))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.2)

    for p,t in enumerate(thresholds):
        print(f"Threshold t={t}mm")
        fig,ax = plt.subplots(figsize=(5,5)) # Individual plot
        i=0
        for k,v in data.items():
            if v[1][:5]=="U-Net":
                y_obs = obs_to_F(
                    y_obs=y_true,
                    threshold=t
                )
                F_pred = v[0][p]
                F_pred, y_obs = F_pred[~np.isnan(F_pred)], y_obs[~np.isnan(F_pred)]
            elif v[1][:3]=="Raw":
                F_pred, y_obs = quantile_to_F(
                    q_pred=group_raw(v[0]),
                    y_obs=y_true[:,1:-1,1:-1],
                    threshold=t
                )
            else:
                F_pred, y_obs = quantile_to_F(
                    q_pred=v[0],
                    y_obs=y_true,
                    threshold=t
                )
            fpr, tpr, thresholds = roc_curve(y_true=y_obs,y_score=F_pred)
            axs[p//c,p%c].plot(fpr,tpr,lw=lw,label=v[1],color=COLORS[i%len(COLORS)])
            ax.plot(fpr,tpr,lw=lw,label=v[1],color=COLORS[i%len(COLORS)])
            i += 1
        
        axs[p//c,p%c].plot([0, 1], [0, 1], color="k", lw=lw, linestyle="--")
        axs[p//c,p%c].axis(xmin=0.,xmax=1.,ymin=0.,ymax=1.)
        axs[p//c,p%c].set_aspect(1.0)
        axs[p//c,p%c].set_title(f"t={int(t)} mm",fontsize=15)
        axs[p//c,p%c].set_xlabel("False Positive Rate")
        axs[p//c,p%c].set_ylabel("True Positive Rate")
        axs[p//c,p%c].legend(loc="lower right")

        # Individual plots
        ax.plot([0, 1], [0, 1], color="k", lw=lw, linestyle="--")
        ax.axis(xmin=0.,xmax=1.,ymin=0.,ymax=1.)
        ax.set_aspect(1.0)
        ax.set_title(f"t={int(t)} mm",fontsize=15)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")
        fig.savefig(os.path.join(path,file_name+f"_t{t}mm.png"),bbox_inches="tight")

    figs.savefig(os.path.join(path,file_name+".png"),bbox_inches="tight")
    plt.close()
    return None