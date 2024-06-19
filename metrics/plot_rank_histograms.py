# --------------------------------- ARGUMENTS -------------------------------- #

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--methods", nargs="+",type=str,
                    default=["raw","qrf","qrf+gtcnd","qrf+csgd","unet+gtcnd","unet+csgd"])
parser.add_argument("-r",type=int,default=3)
parser.add_argument("-c",type=int,default=3)
parser.add_argument("--K", type=int, default=17)
args = parser.parse_args()
print(args)

# ---------------------------------- IMPORTS --------------------------------- #

import os
import numpy as np

import os, sys
sys.path.insert(1,'../utils')
from metrics_rh import plot_grid_rank_hist, JPZ_test

# ---------------------------------------------------------------------------- #

K = args.K

root_ref = "../output/reference_models"
root_unet = "../output/unet_models"
root_obs = "../data"
root_out = "../output/plots/RankHistograms"

print("Plotting the best models")
Z_trainval_list,Z_test_list = [],[]
JPZ_trainval_list,JPZ_test_list = [],[]
name_trainval_list,name_test_list = [],[]
data_trainval_list, data_test_list = [],[]


# ------------------------------- RAW ENSEMBLE ------------------------------- #

if "raw" in args.methods:
    print("RAW ENSEMBLE")
    Z_trainval_list.append(np.load(os.path.join(root_ref,'RankHistograms/Z_raw_trainval.npy')))
    Z_test_list.append(np.load(os.path.join(root_ref,'RankHistograms/Z_raw_test.npy')))
    name_trainval_list.append(("Raw Ensemble",""))
    name_test_list.append(("Raw Ensemble",""))


# ------------------------------------ QRF ----------------------------------- #

if "qrf" in args.methods:
    print("Searching the best QRF model")
    files = [f for f in os.listdir(os.path.join(root_ref,'CRPS')) if f[:9]=="CRPS_qrf_"]
    CRPS_qrf_trainval, CRPS_qrf_test = {}, {}
    for f in files:
        params = '.'.join(f[9:].split('.')[:-1]).split("_")
        pred,q,ntree,mtry,nodesize = params[0],int(params[1][1:]),int(params[2][5:]),int(params[3][4:]),int(params[4][8:])
        
        CRPS_qrf = np.load(os.path.join(os.path.join(root_ref,'CRPS'),f))
        if pred == "trainval":
            CRPS_qrf_trainval[f] = np.nanmean(np.nanmean(CRPS_qrf,axis=0))
        elif pred == "test":
            CRPS_qrf_test[f] = np.nanmean(np.nanmean(CRPS_qrf,axis=0))
    best_qrf_trainval = "_".join(min(CRPS_qrf_trainval, key=CRPS_qrf_trainval.get).split("_")[1:])
    best_qrf_test = "_".join(min(CRPS_qrf_test, key=CRPS_qrf_test.get).split("_")[1:])
    del CRPS_qrf_trainval, CRPS_qrf_test

    Z_trainval_list.append(np.load(os.path.join(root_ref,"RankHistograms/Z_"+best_qrf_trainval)))
    Z_test_list.append(np.load(os.path.join(root_ref,"RankHistograms/Z_"+best_qrf_test)))
    name_trainval_list.append(("QRF",".".join(best_qrf_trainval.split('.')[:-1])))
    name_test_list.append(("QRF",".".join(best_qrf_test.split('.')[:-1])))


# -------------------------- QRF with tail extension ------------------------- #

for distrib in ["gtcnd","csgd"]:
    if f"qrf+{distrib}" in args.methods:
        print(f"Searching the best QRF+{distrib.upper()} model")
        CRPS_qrftail_trainval, CRPS_qrftail_test = {}, {}
        files = [f for f in os.listdir(os.path.join(root_ref,'CRPS')) if f[:(10+len(distrib))]==f"CRPS_qrf+{distrib}_"]
        for f in files:
            params = '.'.join(f[(10+len(distrib)):].split('.')[:-1]).split("_")
            pred,q,ntree,mtry,nodesize = params[0],int(params[1][1:]),int(params[2][5:]),int(params[3][4:]),int(params[4][8:])
            
            CRPS_qrftail = np.load(os.path.join(root_ref,'CRPS',f)).astype(np.float32)
            if pred == "trainval":
                CRPS_qrftail_trainval[f] = np.nanmean(np.nanmean(CRPS_qrftail,axis=0))
            elif pred == "test":
                CRPS_qrftail_test[f] = np.nanmean(np.nanmean(CRPS_qrftail,axis=0))
        best_qrftail_trainval = "_".join(min(CRPS_qrftail_trainval, key=CRPS_qrftail_trainval.get).split("_")[1:])
        best_qrftail_test = "_".join(min(CRPS_qrftail_test, key=CRPS_qrftail_test.get).split("_")[1:])
        del CRPS_qrftail_trainval, CRPS_qrftail_test

        Z_trainval_list.append(np.load(os.path.join(root_ref,"RankHistograms/Z_"+best_qrftail_trainval)))
        Z_test_list.append(np.load(os.path.join(root_ref,"RankHistograms/Z_"+best_qrftail_test)))
        name_trainval_list.append((f"QRF+{distrib.upper()}",".".join(best_qrftail_trainval.split('.')[:-1])))
        name_test_list.append((f"QRF+{distrib.upper()}",".".join(best_qrftail_test.split('.')[:-1])))
        

# ----------------------------------- U-Net ---------------------------------- #

for distrib in ["gtcnd","csgd"]:
    if f"unet+{distrib}" in args.methods:
        print(f"Searching the best U-Net+{distrib.upper()} model")
        CRPS_unet_trainval, CRPS_unet_test = {}, {}
        files = [f for f in os.listdir(os.path.join(root_unet,"CRPS")) if f[:(11+len(distrib))]==f"CRPS_UNet_{distrib}_"]
        for f in files:
            params = '.'.join(f[10:].split('.')[:-1]).split("_")
            distrib, arch, pred, depth, nl, nc, non_separable, lr, batch_size, epochs, nreps = params[0], params[1], params[2], int(params[3][1:]), int(params[4][2:]), int(params[5][2:]), int(params[6][2:]), float(params[7][2:]), int(params[8][1:]), int(params[9][1:]), int(params[10][5:])

            CRPS_unet = np.load(os.path.join(os.path.join(root_unet,'CRPS'),f))
            if pred == "trainval":
                CRPS_unet_trainval[f] = np.nanmean(np.nanmean(CRPS_unet,axis=0))
            elif pred == "test":
                CRPS_unet_test[f] = np.nanmean(np.nanmean(CRPS_unet,axis=0))
        
        best_unet_trainval = "_".join(min(CRPS_unet_trainval, key=CRPS_unet_trainval.get).split("_")[1:])
        best_unet_test = "_".join(min(CRPS_unet_test, key=CRPS_unet_test.get).split("_")[1:])
        del CRPS_unet_trainval, CRPS_unet_test

        # -------------------------- Select specific models -------------------------- #
        # if distrib == "gtcnd":
        #     best_unet_trainval = "UNet_gtcnd_G_trainval_d2_nl0_nc0_ns0_lr1e-05_b16_e170_nreps10.npy"
        #     best_unet_test = "UNet_gtcnd_G_test_d2_nl0_nc0_ns0_lr1e-05_b16_e170_nreps10.npy"
        # elif distrib == "csgd":
        #     best_unet_trainval = "UNet_csgd_G_trainval_d2_nl0_nc0_ns0_lr1e-05_b16_e140_nreps10.npy"
        #     best_unet_test = "UNet_csgd_G_test_d2_nl0_nc0_ns0_lr1e-05_b16_e140_nreps10.npy"

        Z_trainval_list.append(np.load(os.path.join(root_unet,"RankHistograms/Z_"+best_unet_trainval)))
        Z_test_list.append(np.load(os.path.join(root_unet,"RankHistograms/Z_"+best_unet_test)))
        name_trainval_list.append((f"U-Net+{distrib.upper()}",".".join(best_unet_trainval.split('.')[:-1])))
        name_test_list.append((f"U-Net+{distrib.upper()}",".".join(best_unet_test.split('.')[:-1])))
    

print("### Compute JPZ test for training/validation set ###")
for Z in Z_trainval_list:
    data_trainval_list.append([np.count_nonzero(np.logical_and(i/(K+1)<=Z,Z<(i+1)/(K+1)),axis=0)/Z.shape[0] for i in range(K+1)])
    JPZ_trainval_list.append(JPZ_test(np.array([np.count_nonzero(np.logical_and(i/(K+1)<=Z,Z<(i+1)/(K+1)),axis=0) for i in range(K+1)])))
print("### Compute JPZ test for test set ###")
for Z in Z_test_list:
    data_test_list.append([np.count_nonzero(np.logical_and(i/(K+1)<=Z,Z<(i+1)/(K+1)),axis=0)/Z.shape[0] for i in range(K+1)])
    JPZ_test_list.append(JPZ_test(np.array([np.count_nonzero(np.logical_and(i/(K+1)<=Z,Z<(i+1)/(K+1)),axis=0) for i in range(K+1)])))

methods = "-".join(args.methods)
plot_grid_rank_hist(
    data_list=data_trainval_list,
    JPZ_list=JPZ_trainval_list,
    name_list=name_trainval_list,
    path=root_out,
    file_name=f"RankHistograms_trainval_models_{methods}.png",
    r=args.r,
    c=args.c
)
plot_grid_rank_hist(
    data_list=data_test_list,
    JPZ_list=JPZ_test_list,
    name_list=name_test_list,
    path=root_out,
    file_name=f"RankHistograms_test_models_{methods}.png",
    r=args.r,
    c=args.c
)