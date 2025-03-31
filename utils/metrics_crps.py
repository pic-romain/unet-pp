import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import cartopy.crs as ccrs
import cartopy.feature as cf

# Latitude and longitude (should be modified to match the data if different from the one used in the paper)
lats = np.arange(start=43.1,stop=45.9,step=.025)[::-1]
lons = np.arange(start=1.1,stop=5.9,step=.025)

# ---------------------------------------------------------------------------- #
#                                COMPUTE METRICS                               #
# ---------------------------------------------------------------------------- #

def CRPS_fair(ens_pred,Y_obs):
    CRPS = np.empty(shape=(0,*ens_pred.shape[1:-1]),dtype=np.float32)
    M = ens_pred.shape[-1]
    n = ens_pred.shape[0]
    for i in range(n):
        c = np.mean(np.abs(ens_pred[i:(i+1),:,:,:]-np.tile(Y_obs[i:(i+1),:,:][...,np.newaxis],reps=[1,1,1,M])),axis=-1)-1/(2*M*(M-1))*np.sum(np.sum(np.abs(ens_pred[i:(i+1),:,:,:,None]-ens_pred[i:(i+1),:,:,None,:]),axis=-1),axis=-1)
        CRPS = np.concatenate([CRPS,c],axis=0)
    return CRPS

def QS(quantile_pred, alpha, y):
    return 2*(np.less_equal(y,quantile_pred).astype(np.float32)-alpha)*(quantile_pred-y)

def CRPS_qs(quantile_pred,alpha,Y):
    CRPS = np.zeros(shape=quantile_pred.shape[:-1])
    n=quantile_pred.shape[-1]
    for i in range(n):
        CRPS = CRPS+QS(quantile_pred[:,:,:,i],alpha[:,:,:,i],Y)/(n+1)
    return CRPS

# ---------------------------------------------------------------------------- #
#                                 PLOT METRICS                                 #
# ---------------------------------------------------------------------------- #

def plot_crps(CRPS_mean,lons,lats,margin,path_save,alpha=.95):
    fig = plt.figure(figsize=(10,10))
    ax = plt.axes(projection=ccrs.Robinson())
    ax.use_sticky_edges = False
    ax.set_xmargin(margin)
    ax.set_ymargin(margin)
    
    cmap = plt.cm.get_cmap("coolwarm")
    levels = np.linspace(start=0,stop=1,num=21)
    cm = plt.pcolormesh(lons, lats, CRPS_mean,
                        transform=ccrs.PlateCarree(),alpha=alpha,
                        cmap=cmap, norm=colors.BoundaryNorm(levels,ncolors=cmap.N,clip=False),shading='auto')

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                    linewidth=1.5, color='gray', alpha=0.25, linestyle='--')
        
    ax.coastlines(resolution="10m")
    ax.add_feature(cf.BORDERS)
    plt.title(f"Mean CRPS = {np.mean(CRPS_mean):.4f}",fontsize=15)
    
    colorbar_axes = fig.add_axes([0, 0, 0.1, 0.1])
    posn = ax.get_position()
    colorbar_axes.set_position([posn.x0 + posn.width + 0.075, posn.y0,0.04, posn.height])
    cbar = plt.colorbar(cm,cax=colorbar_axes,extend='max',ticks=levels[::2])
    cbar.ax.tick_params(labelsize=15)

    plt.savefig(path_save,bbox_inches='tight')
    plt.close()
    return None

def plot_crpss_raw(CRPS_mean,CRPS_raw_mean,lons,lats,margin,path_save,alpha=.95):
    CRPSS_raw = 1-CRPS_mean/CRPS_raw_mean
    
    fig = plt.figure(figsize=(10,10))
    ax = plt.axes(projection=ccrs.Robinson())
    ax.use_sticky_edges = False
    ax.set_xmargin(margin)
    ax.set_ymargin(margin)
    
    cmap = plt.cm.get_cmap("RdBu_r")
    levels = [-100,-50,-30,-20,-10,-2.5,-1,1,2.5,10,20,30,50,100]
    cm = plt.pcolormesh(lons, lats, CRPSS_raw*100,
                        transform=ccrs.PlateCarree(),alpha=alpha,
                        cmap=cmap,norm=colors.BoundaryNorm(levels,ncolors=cmap.N,clip=False),shading='auto')
    
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                    linewidth=1.5, color='gray', alpha=0.25, linestyle='--')
    
    ax.coastlines(resolution="10m")
    ax.add_feature(cf.BORDERS)
    # plt.title(f"Mean CRPSS = {np.nanmean(CRPSS_raw)*100:.4f}%",fontsize=15)
    print(f"Mean CRPSS = {np.nanmean(CRPSS_raw)*100:.4f}%")
    colorbar_axes = fig.add_axes([0, 0, 0.1, 0.1])
    posn = ax.get_position()
    colorbar_axes.set_position([posn.x0 + posn.width + 0.075, posn.y0,0.04, posn.height])
    cbar = plt.colorbar(cm,cax=colorbar_axes,ticks=levels,extend="both")
    cbar.ax.tick_params(labelsize=15)
    # cbar.ax.set_ylabel('(%)', loc="top",rotation=0,fontsize=15)
    plt.savefig(path_save,bbox_inches='tight')
    plt.close()
    return None

def plot_crpss_qrf(CRPS_mean,CRPS_qrf_mean,lons,lats,margin,path_save,alpha=.95):
    CRPSS_qrf = 1-CRPS_mean/CRPS_qrf_mean
    
    fig = plt.figure(figsize=(10,10))
    ax = plt.axes(projection=ccrs.Robinson())
    ax.use_sticky_edges = False
    ax.set_xmargin(margin)
    ax.set_ymargin(margin)

    cmap = plt.cm.get_cmap("RdBu_r")
    levels = [-100,-50,-30,-20,-10,-2.5,-1,1,2.5,10,20,30,50,100]

    cm = plt.pcolormesh(lons, lats, CRPSS_qrf*100,
                        transform=ccrs.PlateCarree(),alpha=alpha,
                        cmap=cmap,norm=colors.BoundaryNorm(levels,ncolors=cmap.N,clip=False),shading='auto')
    
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                    linewidth=1.5, color='gray', alpha=0.25, linestyle='--')
    
    ax.coastlines(resolution="10m")
    ax.add_feature(cf.BORDERS)
    # plt.title(f"Mean CRPSS = {np.nanmean(CRPSS_qrf)*100:.4f}%",fontsize=15)
    print(f"Mean CRPSS = {np.nanmean(CRPSS_qrf)*100:.4f}%")
    colorbar_axes = fig.add_axes([0, 0, 0.1, 0.1])
    posn = ax.get_position()
    colorbar_axes.set_position([posn.x0 + posn.width + 0.075, posn.y0,0.04, posn.height])
    cbar = plt.colorbar(cm,cax=colorbar_axes,ticks=levels,extend="both")
    cbar.ax.tick_params(labelsize=15)
    
    # cbar.ax.set_ylabel('(%)', loc="top",rotation=0,fontsize=15)
    plt.savefig(path_save,bbox_inches='tight')
    plt.close()
    return None