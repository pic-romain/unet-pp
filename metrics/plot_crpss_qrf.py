import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-m","--margin",type=float,default=.025,help="Margins around the plot")
parser.add_argument("--methods", nargs="+",type=str,
                    default=["qrf+gtcnd","qrf+csgd","unet+gtcnd","unet+csgd"])
args = parser.parse_args()
print(args)

import os, sys
sys.path.insert(1,'../utils')
from metrics_crps import plot_crpss_qrf, lats, lons

import numpy as np

# ---------------------------------------------------------------------------- #

print("PLOT CRPSS/qrf MAPS ")

root_ref = "../output/reference_models/CRPS"
root_unet = "../output/unet_models/CRPS"
root_out = "../output/plots/CRPSS_qrf"

# --------------------------- QRF (reference model) -------------------------- #

print("Searching the best QRF model")
files = [f for f in os.listdir(root_ref) if f[:9]=="CRPS_qrf_"]
CRPS_qrf_trainval, CRPS_qrf_test = {}, {}
for f in files:
        params = '.'.join(f[9:].split('.')[:-1]).split("_")
        pred,q,ntree,mtry,nodesize = params[0],int(params[1][1:]),int(params[2][5:]),int(params[3][4:]),int(params[4][8:])
        
        CRPS_qrf = np.load(os.path.join(root_ref,f))
        CRPS_qrf_mean = np.nanmean(CRPS_qrf,axis=0)

        if pred == "trainval":
            CRPS_qrf_trainval[f] = np.nanmean(CRPS_qrf_mean)
        elif pred == "test":
            CRPS_qrf_test[f] = np.nanmean(CRPS_qrf_mean)

best_qrf_trainval = min(CRPS_qrf_trainval, key=CRPS_qrf_trainval.get)
best_qrf_test = min(CRPS_qrf_test, key=CRPS_qrf_test.get)
del CRPS_qrf_trainval, CRPS_qrf_test

CRPS_qrf_mean_trainval = np.mean(np.load(os.path.join(root_ref,best_qrf_trainval)),axis=0)
CRPS_qrf_mean_test = np.mean(np.load(os.path.join(root_ref,best_qrf_test)),axis=0)
print(f"Best trainval : {best_qrf_trainval}")
print(f"Best test : {best_qrf_test}")


# -------------------------- QRF with tail extension ------------------------- #

for distrib in ["gtcnd","csgd"]:
    if f"qrf+{distrib}" in args.methods:
        print(f"### PLOT CRPSS QRF+{distrib.upper()} ###")
        files = [f for f in os.listdir(root_ref) if f[:(10+len(distrib))]==f"CRPS_qrf+{distrib}_"]
        for f in files:
            params = '.'.join(f[(10+len(distrib)):].split('.')[:-1]).split("_")
            pred, q, ntree, mtry, nodesize = params[0], int(params[1][1:]), int(params[2][5:]), int(params[3][4:]), int(params[4][8:])

            print(f"QRF+{distrib.upper()} PRED={pred} Q={q} NTREE={ntree} MTRY={mtry} NODESIZEMIN={nodesize}")
            CRPS_qrftail = np.load(os.path.join(root_ref,f)).astype(np.float32)
            CRPS_qrftail_mean = np.nanmean(CRPS_qrftail,axis=0)

            if pred == "trainval":
                plot_crpss_qrf(
                    CRPS_mean=CRPS_qrftail_mean,
                    CRPS_qrf_mean=CRPS_qrf_mean_trainval,
                    lons=lons,
                    lats=lats,
                    margin=args.margin,
                    path_save=os.path.join(root_out,f'CRPSS_{pred}_qrf+{distrib}_q{q}_ntree{ntree}_mtry{mtry}_nodesize{nodesize}_qrf.png')
                )
            elif pred == "test":
                plot_crpss_qrf(
                    CRPS_mean=CRPS_qrftail_mean,
                    CRPS_qrf_mean=CRPS_qrf_mean_test,
                    lons=lons,
                    lats=lats,
                    margin=args.margin,
                    path_save=os.path.join(root_out,f'CRPSS_{pred}_qrf+{distrib}_q{q}_ntree{ntree}_mtry{mtry}_nodesize{nodesize}_qrf.png')
                )

        
# ----------------------------------- U-Net ---------------------------------- #

for distrib in ["gtcnd","csgd"]:
    if f"unet+{distrib}" in args.methods:
        print(f"### PLOT CRPSS U-Net+{distrib.upper()} ###")
        files = [f for f in os.listdir(root_unet) if f[:(11+len(distrib))]==f"CRPS_UNet_{distrib}_"]
        for f in files:
            params = '.'.join(f[10:].split('.')[:-1]).split("_")
            distrib, arch, pred, depth, nl, nc, non_separable, lr, batch_size, epochs, nreps = params[0], params[1], params[2], int(params[3][1:]), int(params[4][2:]), int(params[5][2:]), int(params[6][2:]), float(params[7][2:]), int(params[8][1:]), int(params[9][1:]), int(params[10][5:])

            print(f"U-Net+{distrib.upper()} {arch} PRED={pred} DEPTH={depth} NC={nc} NON_SEPARABLE={non_separable} LR={lr} BATCH_SIZE={batch_size} EPOCHS={epochs} nreps={nreps}")
            CRPS = np.load(os.path.join(root_unet,f))
            CRPS_mean = np.nanmean(CRPS,axis=0)
            
            if pred == "trainval":
                plot_crpss_qrf(
                    CRPS_mean=CRPS_mean,
                    CRPS_qrf_mean=CRPS_qrf_mean_trainval,
                    lons=lons,
                    lats=lats,
                    margin=args.margin,
                    path_save=os.path.join(root_out,f'CRPSS_{pred}_UNet_{distrib}_{arch}_d{depth}_nc{nc}_ns{non_separable}_lr{lr}_b{batch_size}_e{epochs}_nreps{nreps}_qrf.png')
                )
            elif pred == "test":
                plot_crpss_qrf(
                    CRPS_mean=CRPS_mean,
                    CRPS_qrf_mean=CRPS_qrf_mean_test,
                    lons=lons,
                    lats=lats,
                    margin=args.margin,
                    path_save=os.path.join(root_out,f'CRPSS_{pred}_UNet_{distrib}_{arch}_d{depth}_nc{nc}_ns{non_separable}_lr{lr}_b{batch_size}_e{epochs}_nreps{nreps}_qrf.png')
                )