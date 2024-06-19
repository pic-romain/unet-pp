# --------------------------------- ARGUMENTS -------------------------------- #

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--methods", nargs="+", type=str, 
                    default=["raw", "qrf", "qrf+gtcnd", "qrf+csgd", "unet+gtcnd", "unet+csgd"])
args = parser.parse_args()
print(args)

# ---------------------------------- IMPORTS --------------------------------- #

import numpy as np
import os, sys

sys.path.insert(1, "../utils")
from metrics_crps import CRPS_fair, CRPS_qs
from unet import CRPS_gtcnd, CRPS_csgd

# ---------------------------------------------------------------------------- #

root = "../data"
root_ref = "../output/reference_models/models"
root_ref_out = "../output/reference_models/CRPS"
root_unet = "../output/unet_models/parameters"
root_unet_out = "../output/unet_models/CRPS"

# ------------------------------- Observations ------------------------------- #

Y_trainval = np.load(os.path.join(root, "Y_trainval.npy")).astype(np.float32)
Y_test = np.load(os.path.join(root, "Y_test.npy")).astype(np.float32)


# ------------------------------- RAW ENSEMBLE ------------------------------- #

if "raw" in args.methods:
    print("### CRPS RAW ###")
    raw_precip_trainval = np.load(os.path.join(root_ref, "X_raw_trainval.npy")).astype(np.float32)
    raw_precip_test = np.load(os.path.join(root_ref, "X_raw_test.npy")).astype(np.float32)
    np.save(
        file=os.path.join(root_ref_out, "CRPS_raw_trainval.npy"),
        arr=CRPS_fair(raw_precip_trainval, Y_trainval),
    )
    np.save(
        file=os.path.join(root_ref_out, "CRPS_raw_test.npy"),
        arr=CRPS_fair(raw_precip_test, Y_test),
    )

# ------------------------------------ QRF ----------------------------------- #

if "qrf" in args.methods:
    print("### CRPS QRF ###")
    files = [f for f in os.listdir(root_ref) if f[:8] == "qrf_pred"]
    for f in files:
        params = ".".join(f[9:].split(".")[:-1]).split("_")
        pred, q, ntree, mtry, nodesize = params[0], int(params[1][1:]), int(params[2][5:]), int(params[3][4:]), int(params[4][8:])
        
        print(f"CRPS QRF PRED={pred} Q={q} NTREE={ntree} MTRY={mtry} NODESIZEMIN={nodesize}")
        qrf_precip = np.load(os.path.join(root_ref, f)).astype(np.float32)
        n_quantiles = qrf_precip.shape[-1]
        quantiles = np.array([i / (n_quantiles + 1) for i in range(1, n_quantiles + 1)])
        
        if pred == "trainval":
            CRPS_qrf = CRPS_qs(
                qrf_precip,
                alpha=np.tile(
                    quantiles[np.newaxis, np.newaxis, np.newaxis, ...],
                    [*qrf_precip.shape[:-1], 1],
                ),
                Y=Y_trainval,
            )
        elif pred == "test":
            CRPS_qrf = CRPS_qs(
                qrf_precip,
                alpha=np.tile(
                    quantiles[np.newaxis, np.newaxis, np.newaxis, ...],
                    [*qrf_precip.shape[:-1], 1],
                ),
                Y=Y_test,
            )
        else:
            raise ValueError("pred should be trainval or test")
        
        np.save(
            os.path.join(root_ref, f"CRPS/CRPS_qrf_{pred}_q{q}_ntree{ntree}_mtry{mtry}_nodesize{nodesize}.npy"),
            CRPS_qrf
        )


# -------------------------- QRF with tail extension ------------------------- #

for distrib in ["gtcnd", "csgd"]:
    if f"qrf+{distrib}" in args.methods:
        print(f"### CRPS QRF+{distrib.upper()} ###")
        files = [f for f in os.listdir(root_ref) if f[: (10 + len(distrib))] == f"qrf+{distrib}_pred_"]
        for f in files:
                params = ".".join(f[(10 + len(distrib)) :].split(".")[:-1]).split("_")
                pred, q, ntree, mtry, nodesize = params[0], int(params[1][1:]), int(params[2][5:]), int(params[3][4:]), int(params[4][8:])
                
                print(f"CRPS QRF+{distrib.upper()} PRED={pred} Q={q} NTREE={ntree} MTRY={mtry} NODESIZEMIN={nodesize}")
                qrf_precip = np.load(os.path.join(root_ref, f)).astype(np.float32)
                n_quantiles = qrf_precip.shape[-1]
                quantiles = np.array([i / (n_quantiles + 1) for i in range(1, n_quantiles + 1)])

                if pred == "trainval":
                    CRPS_qrf = CRPS_qs(
                        qrf_precip,
                        alpha=np.tile(
                            quantiles[np.newaxis, np.newaxis, np.newaxis, ...],
                            [*qrf_precip.shape[:-1], 1],
                        ),
                        Y=Y_trainval,
                    )
                elif pred == "test":
                    CRPS_qrf = CRPS_qs(
                        qrf_precip,
                        alpha=np.tile(
                            quantiles[np.newaxis, np.newaxis, np.newaxis, ...],
                            [*qrf_precip.shape[:-1], 1],
                        ),
                        Y=Y_test,
                    )
                else:
                    raise ValueError("pred should be trainval or test")
                np.save(
                    os.path.join(root_ref, f"CRPS/CRPS_qrf+{distrib}_{pred}_q{q}_ntree{ntree}_mtry{mtry}_nodesize{nodesize}.npy"),
                    CRPS_qrf
                )


# ----------------------------------- U-Net ---------------------------------- #

for distrib in ["gtcnd", "csgd"]:
    if f"unet+{distrib}" in args.methods:
        print(f"### CRPS U-Net+{distrib.upper()} ###")
        files = [f for f in os.listdir(root_unet) if f[: (13 + len(distrib))] == f"params_UNet_{distrib}_"]
        for f in files:
            params = ".".join(f[12:].split(".")[:-1]).split("_")
            distrib, arch, pred, depth, nl, nc, non_separable, lr, batch_size, epochs, nreps = params[0], params[1], params[2], int(params[3][1:]), int(params[4][2:]), int(params[5][2:]), int(params[6][2:]), float(params[7][2:]), int(params[8][1:]), int(params[9][1:]), int(params[10][5:])
            print(f"U-Net+{distrib.upper()} {arch} PRED={pred} DEPTH={depth} NL={nl} NC={nc} NON_SEPARABLE={non_separable} LR={lr} BATCH_SIZE={batch_size} EPOCHS={epochs} nreps={nreps}")

            params_pred = np.load(os.path.join(root_unet, f)).astype(np.float32)
            n_nan = len(np.where(np.isnan(params_pred))[0])
            if n_nan > 0:
                print(f"Number of NaN values in the parameters : {n_nan}")
            params_pred = np.nanmean(params_pred, axis=0)

            Y = Y_trainval if pred == "trainval" else (Y_test if pred == "test" else None)

            if distrib == "gtcnd":
                CRPS_unet = CRPS_gtcnd(
                    rho_L=params_pred[..., 0],
                    mu=params_pred[..., 1],
                    rho_sigma=params_pred[..., 2],
                    y=Y,
                    fit=False,
                    mean=False,
                )
            elif distrib == "csgd":
                CRPS_unet = CRPS_csgd(
                    rho_k=params_pred[..., 0],
                    rho_theta=params_pred[..., 1],
                    rho_delta=params_pred[..., 2],
                    y=Y,
                    fit=False,
                    mean=False,
                )
            else:
                raise(f"ERROR {distrib} not available")
            np.save(
                os.path.join(root_unet_out, f"CRPS_UNet_{distrib}_{arch}_{pred}_d{depth}_nl{nl}_nc{nc}_ns{non_separable}_lr{lr}_b{batch_size}_e{epochs}_nreps{nreps}.npy"),
                CRPS_unet,
            )
