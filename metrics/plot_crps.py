# --------------------------------- ARGUMENTS -------------------------------- #

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--margin", type=float, default=.025, help="Margins around the plot")
parser.add_argument("--methods", nargs="+", type=str,
                    default=["raw","qrf","qrf+gtcnd","qrf+csgd","unet+gtcnd","unet+csgd"])
args = parser.parse_args()
print(args)

# ---------------------------------- IMPORTS --------------------------------- #

import os, sys
sys.path.insert(1,'../utils')
from metrics_crps import plot_crps, lats, lons

import numpy as np

# ---------------------------------------------------------------------------- #

print("PLOT CRPS MAPS")

root_ref = "../output/reference_models/CRPS"
root_unet = "../output/unet_models/CRPS"
root_out = "../output/plots/CRPS"


# ------------------------------- RAW ENSEMBLE ------------------------------- #

if "raw" in args.methods:
    print("### PLOT CRPS RAW ###")
    CRPS_raw_trainval = np.load(os.path.join(root_ref, 'CRPS_raw_trainval.npy'))
    CRPS_raw_test = np.load(os.path.join(root_ref, 'CRPS_raw_test.npy'))
    CRPS_raw_mean_trainval = np.nanmean(CRPS_raw_trainval, axis=0)
    CRPS_raw_mean_test = np.nanmean(CRPS_raw_test, axis=0)

    # Training/validation set
    plot_crps(
        CRPS_mean=CRPS_raw_mean_trainval,
        lons=lons,
        lats=lats,
        margin=args.margin,
        path_save=os.path.join(root_out, "CRPS_trainval_raw.png"),
    )
    # Test set
    plot_crps(
        CRPS_mean=CRPS_raw_mean_test,
        lons=lons,
        lats=lats,
        margin=args.margin,
        path_save=os.path.join(root_out, "CRPS_test_raw.png"),
    )


# ------------------------------------ QRF ----------------------------------- #

if "qrf" in args.methods:
    print("### PLOT CRPS QRF ###")
    files = [f for f in os.listdir(root_ref) if f[:9]=="CRPS_qrf_"]
    for f in files:
        params = ".".join(f[9:].split(".")[:-1]).split("_")
        pred, q, ntree, mtry, nodesize = params[0], int(params[1][1:]), int(params[2][5:]), int(params[3][4:]), int(params[4][8:])

        print(f"CRPS QRF PRED={pred} Q={q} NTREE={ntree} MTRY={mtry} NODESIZEMIN={nodesize}")
        CRPS_qrf = np.load(os.path.join(root_ref, f))
        CRPS_qrf_mean = np.nanmean(CRPS_qrf, axis=0)

        if pred == "trainval":
            plot_crps(
                CRPS_mean=CRPS_qrf_mean,
                lons=lons,
                lats=lats,
                margin=args.margin,
                path_save=os.path.join(root_out,f"CRPS_{pred}_qrf_q{q}_ntree{ntree}_mtry{mtry}_nodesize{nodesize}.png")
            )
        elif pred == "test":
            plot_crps(
                CRPS_mean=CRPS_qrf_mean,
                lons=lons,
                lats=lats,
                margin=args.margin,
                path_save=os.path.join(root_out,f"CRPS_{pred}_qrf_q{q}_ntree{ntree}_mtry{mtry}_nodesize{nodesize}.png")
            )


# -------------------------- QRF with tail extension ------------------------- #

for distrib in ["gtcnd","csgd"]:
    if f"qrf+{distrib}" in args.methods:
        print(f"CRPS QRF+{distrib.upper()}")
        files = [f for f in os.listdir(root_ref) if f[:(10+len(distrib))]==f"CRPS_qrf+{distrib}_"]
        for f in files:
            params = '.'.join(f[(10+len(distrib)):].split('.')[:-1]).split("_")
            pred, q, ntree, mtry, nodesize = params[0], int(params[1][1:]), int(params[2][5:]), int(params[3][4:]), int(params[4][8:])

            print(f"CRPS QRF+{distrib.upper()} PRED={pred} Q={q} NTREE={ntree} MTRY={mtry} NODESIZEMIN={nodesize}")
            CRPS_qrftail = np.load(os.path.join(root_ref,f)).astype(np.float32)
            CRPS_qrftail_mean = np.nanmean(CRPS_qrftail,axis=0)

            if pred == "trainval":
                plot_crps(
                    CRPS_mean=CRPS_qrftail_mean,
                    lons=lons,
                    lats=lats,
                    margin=args.margin,
                    path_save=os.path.join(root_out,f'CRPS_{pred}_qrf+{distrib}_q{q}_ntree{ntree}_mtry{mtry}_nodesize{nodesize}.png')
                )
            elif pred == "test":
                plot_crps(
                    CRPS_mean=CRPS_qrftail_mean,
                    lons=lons,
                    lats=lats,
                    margin=args.margin,
                    path_save=os.path.join(root_out,f'CRPS_{pred}_qrf+{distrib}_q{q}_ntree{ntree}_mtry{mtry}_nodesize{nodesize}.png')
                )


# ----------------------------------- U-Net ---------------------------------- #

for distrib in ["gtcnd","csgd"]:
    if f"unet+{distrib}" in args.methods:
        print(f"CRPS U-Net+{distrib.upper()}")
        files = [f for f in os.listdir(root_unet) if f[:(11+len(distrib))]==f"CRPS_UNet_{distrib}_"]
        for f in files:
            params = '.'.join(f[10:].split('.')[:-1]).split("_")
            distrib, arch, pred, depth, nl, nc, non_separable, lr, batch_size, epochs, nreps = params[0], params[1], params[2], int(params[3][1:]), int(params[4][2:]), int(params[5][2:]), int(params[6][2:]), float(params[7][2:]), int(params[8][1:]), int(params[9][1:]), int(params[10][5:])
            
            print(f"U-Net+{distrib.upper()} {arch} PRED={pred} DEPTH={depth} NL={nl} NC={nc} NON_SEPARABLE={non_separable} LR={lr} BATCH_SIZE={batch_size} EPOCHS={epochs} nreps={nreps}")
            CRPS = np.load(os.path.join(root_unet,f))
            CRPS_mean = np.nanmean(CRPS,axis=0)

            if pred == "trainval":
                plot_crps(
                    CRPS_mean=CRPS_mean,
                    lons=lons,
                    lats=lats,
                    margin=args.margin,
                    path_save=os.path.join(root_out,f'CRPS_{pred}_UNet_{distrib}_{arch}_d{depth}_nl{nl}_nc{nc}_ns{non_separable}_lr{lr}_b{batch_size}_e{epochs}_nreps{nreps}.png')
                )
            elif pred == "test":
                plot_crps(
                    CRPS_mean=CRPS_mean,
                    lons=lons,
                    lats=lats,
                    margin=args.margin,
                    path_save=os.path.join(root_out,f'/CRPS_{pred}_UNet_{distrib}_{arch}_d{depth}_nl{nl}_nc{nc}_ns{non_separable}_lr{lr}_b{batch_size}_e{epochs}_nreps{nreps}.png')
                )
