# --------------------------------- ARGUMENTS -------------------------------- #

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--methods", nargs="+",type=str,
                    default=["raw","qrf","qrf+gtcnd","qrf+csgd","unet+gtcnd","unet+csgd"])
parser.add_argument("--K", type=int, default=17)
args = parser.parse_args()
print(args)

# ---------------------------------- IMPORTS --------------------------------- #

import os, sys
sys.path.insert(1,'../utils')
from metrics_rh import rank_hist, quantile_gtcnd, quantile_csgd
from learning import trainval_test_index

import numpy as np

# ---------------------------------------------------------------------------- #

K = args.K

root_obs = "../data"
root_ref = "../output/reference_models/models"
root_unet = "../output/unet_models/parameters"

root_ref_out = "../output/reference_models/RankHistograms"
root_unet_out = "../output/unet_models/RankHistograms"

# ------------------------------- OBSERVATIONS ------------------------------- #

Y_trainval = np.load(os.path.join(root_obs,"Y_trainval.npy")).astype(np.float32)
Y_test = np.load(os.path.join(root_obs,"Y_test.npy")).astype(np.float32)


# ------------------------------- RAW ENSEMBLE ------------------------------- #

if "raw" in args.methods:
    print("RAW ENSEMBLE")
    raw_precip_trainval = np.load(os.path.join(root_ref, "X_raw_trainval.npy")).astype(np.float32)
    raw_precip_test = np.load(os.path.join(root_ref, "X_raw_test.npy")).astype(np.float32)

    print("Training/validation set")
    Z_trainval = rank_hist(forecast = raw_precip_trainval, true_obs = Y_trainval)
    np.save(
        os.path.join(root_ref_out,'Z_raw_trainval.npy'),
        Z_trainval
    )

    print("Test set")
    Z_test = rank_hist(forecast = raw_precip_test, true_obs = Y_test)
    np.save(
        os.path.join(root_ref_out,'Z_raw_test.npy'),
        Z_test
    )


# ------------------------------------ QRF ----------------------------------- #

if "qrf" in args.methods:
    print("### Rank Histogram QRF ###")
    files = [f for f in os.listdir(root_ref) if f[:8]=="qrf_pred"]
    ### Filter QRF with certain parameters
    # files = [f for f in files if "ntree2000_mtry4_nodesize20" in f]

    for f in files:
        params = '.'.join(f[9:].split('.')[:-1]).split("_")
        pred,q,ntree,mtry,nodesize = params[0],int(params[1][1:]),int(params[2][5:]),int(params[3][4:]),int(params[4][8:])
        
        print(f"QRF PRED={pred} Q={q} NTREE={ntree} MTRY={mtry} NODESIZEMIN={nodesize}")
        qrf_forecast = np.load(os.path.join(root_ref,f)).astype(np.float32)
        if pred=="trainval":
            Z_qrf = rank_hist(
                forecast=qrf_forecast,
                true_obs=Y_trainval
            )
        elif pred=="test":
            Z_qrf = rank_hist(
                forecast=qrf_forecast,
                true_obs=Y_test
            )
        np.save(
            os.path.join(root_ref_out,f"Z_qrf_{pred}_q{q}_ntree{ntree}_mtry{mtry}_nodesize{nodesize}.npy"),
            Z_qrf
        )


# -------------------------- QRF with tail extension ------------------------- #

for distrib in ["gtcnd","csgd"]:
    if f"qrf+{distrib}" in args.methods:
        print(f"### Rank Histogram QRF+{distrib.upper()} ###")
        files = [f for f in os.listdir(root_ref) if f[:(10+len(distrib))]==f"qrf+{distrib}_pred_"]
        ### Filter QRF+tail with certain parameters
        # files = [f for f in files if "ntree2000_mtry4_nodesize20" in f] 

        for f in files:
            params = '.'.join(f[(10+len(distrib)):].split('.')[:-1]).split("_")
            pred,q,ntree,mtry,nodesize = params[0],int(params[1][1:]),int(params[2][5:]),int(params[3][4:]),int(params[4][8:])
            
            print(f"QRF+{distrib.upper()} PRED={pred} Q={q} NTREE={ntree} MTRY={mtry} NODESIZEMIN={nodesize}")
            qrf_forecast = np.load(os.path.join(root_ref,f)).astype(np.float32)
            if pred=="trainval":
                Z_qrf = rank_hist(forecast=qrf_forecast,true_obs=Y_trainval)
            elif pred=="test":
                Z_qrf = rank_hist(forecast=qrf_forecast,true_obs=Y_test)

            np.save(
                os.path.join(root_ref_out, f"Z_qrf+{distrib}_{pred}_q{q}_ntree{ntree}_mtry{mtry}_nodesize{nodesize}.npy"),
                Z_qrf
            )


# ----------------------------------- U-Net ---------------------------------- #

for distrib in ["gtcnd","csgd"]:
    if f"unet+{distrib}" in args.methods:
        print(f"### Rank Histogram U-Net+{distrib.upper()} ###")
        files = [f for f in os.listdir(root_unet) if f[:(13+len(distrib))]==f"params_UNet_{distrib}_"]
        ### Filter U-Net with certain parameters
        # if distrib == "gtcnd":
        #     files = [f for f in files if "d2_nl0_nc0_ns0_lr1e-05_b16_e170_nreps10" in f]
        # elif distrib == "csgd":
        #     files = [f for f in files if "d2_nl0_nc0_ns0_lr1e-05_b16_e140_nreps10" in f]

        for f in files:
            params = '.'.join(f[12:].split('.')[:-1]).split("_")
            distrib, arch, pred, depth, nl, nc, non_separable, lr, batch_size, epochs, nreps = params[0], params[1], params[2], int(params[3][1:]), int(params[4][2:]), int(params[5][2:]), int(params[6][2:]), float(params[7][2:]), int(params[8][1:]), int(params[9][1:]), int(params[10][5:])

            print(f"U-Net+{distrib.upper()} {arch} PRED={pred} DEPTH={depth} NL={nl} NC={nc} NON_SEPARABLE={non_separable} LR={lr} BATCH_SIZE={batch_size} EPOCHS={epochs} nreps={nreps}")

            params_pred = np.nanmean(np.load(os.path.join(root_unet,f)).astype(np.float32),axis=0)
            n,d1,d2 = params_pred.shape[:-1]
            quantiles = np.array([i/(K+1) for i in range(1,K+1)])
            if distrib == "gtcnd":
                unet_quantiles = quantile_gtcnd(
                    p=np.tile(quantiles[np.newaxis,np.newaxis,np.newaxis,...],reps=[n,d1,d2,1]),
                    rho_L=np.tile(params_pred[:,:,:,0][...,np.newaxis],reps=[1,1,1,K]),
                    mu=np.tile(params_pred[:,:,:,1][...,np.newaxis],reps=[1,1,1,K]),
                    rho_sigma=np.tile(params_pred[:,:,:,2][...,np.newaxis],reps=[1,1,1,K])
                )
            elif distrib == "csgd":
                unet_quantiles = quantile_csgd(
                    p=np.tile(quantiles[np.newaxis,np.newaxis,np.newaxis,...],reps=[n,d1,d2,1]),
                    rho_k=np.tile(params_pred[:,:,:,0][...,np.newaxis],reps=[1,1,1,K]),
                    rho_theta=np.tile(params_pred[:,:,:,1][...,np.newaxis],reps=[1,1,1,K]),
                    rho_delta=np.tile(params_pred[:,:,:,2][...,np.newaxis],reps=[1,1,1,K])
                )
            
            if pred=="trainval":
                rank_unet,Z_unet = rank_hist(
                    forecast=unet_quantiles,
                    true_obs=Y_trainval
                )
            elif pred=="test":
                rank_unet,Z_unet = rank_hist(
                    forecast=unet_quantiles,
                    true_obs=Y_test
                )

            np.save(
                os.path.join(root_unet_out, f"Z_UNet_{distrib}_{arch}_{pred}_d{depth}_nl{nl}_nc{nc}_ns{non_separable}_lr{lr}_b{batch_size}_e{epochs}_nreps{nreps}.npy"),
                Z_unet
            )