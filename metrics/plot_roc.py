# --------------------------------- ARGUMENTS -------------------------------- #

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--methods", nargs="+",type=str,
                    default=["raw","qrf","qrf+gtcnd","qrf+csgd","unet+gtcnd","unet+csgd"])
parser.add_argument("--pred", type=str, default="test")
parser.add_argument("--thresholds", nargs='+',type=float,default=[0.,5.,10.,20.])
parser.add_argument("-r",type=int,default=2)
parser.add_argument("-c",type=int,default=2)
parser.add_argument("--K",type=int,default=17)
args = parser.parse_args()
print(args)

# ---------------------------------- IMPORTS --------------------------------- #

import os, sys
sys.path.insert(1,'../utils')
from metrics_roc import plot_grid_roc, cdf_gtcnd, cdf_csgd

import numpy as np

# ---------------------------------------------------------------------------- #

K = args.K

root_obs = "../data"
root_raw = "../data"
root_ref = "../output/reference_models"
root_unet = "../output/unet_models"
root_out = "../output/plots/ROC"

# ------------------------------- Observations ------------------------------- #

if args.pred == "trainval":
    Y_trainval = np.load(os.path.join(root_obs,"Y_trainval.npy")).astype(np.float32)
    trainval_dict = {}
elif args.pred == "test":
    Y_test = np.load(os.path.join(root_obs,"Y_test.npy")).astype(np.float32)
    test_dict = {}



# ------------------------------- RAW ENSEMBLE ------------------------------- #

if "raw" in args.methods:
    print("RAW ENSEMBLE")
    if args.pred == "trainval":
        X_raw_trainval = np.load(os.path.join(root_raw, "X_raw_trainval.npy")).astype(np.float32)
        trainval_dict["raw_ensemble"] = [X_raw_trainval,"Raw Ensemble"]
        del X_raw_trainval
    elif args.pred == "test":
        X_raw_test = np.load(os.path.join(root_raw, "X_raw_test.npy")).astype(np.float32)
        test_dict["raw_ensemble"] = [X_raw_test,"Raw Ensemble"]
        del X_raw_test


# ------------------------------------ QRF ----------------------------------- #

if "qrf" in args.methods:
    print("Searching the best QRF model")
    files = [f for f in os.listdir(os.path.join(root_ref,'CRPS')) if f[:9]=="CRPS_qrf_"]
    CRPS_qrf_trainval, CRPS_qrf_test = {}, {}
    for f in files:
        params = '.'.join(f[9:].split('.')[:-1]).split("_")
        pred,q,ntree,mtry,nodesize = params[0],int(params[1][1:]),int(params[2][5:]),int(params[3][4:]),int(params[4][8:])
        
        CRPS_qrf = np.load(os.path.join(os.path.join(root_ref,'CRPS'),f)).astype(np.float32)
        if pred == "trainval":
            CRPS_qrf_trainval[f] = np.nanmean(np.nanmean(CRPS_qrf,axis=0))
        elif pred == "test":
            CRPS_qrf_test[f] = np.nanmean(np.nanmean(CRPS_qrf,axis=0))

    if args.pred == "trainval":
        best_qrf_trainval = "_".join(min(CRPS_qrf_trainval, key=CRPS_qrf_trainval.get).split("_")[1:])
        print("Training/validation set :", best_qrf_trainval)
        qrf_trainval = np.load(os.path.join(root_ref,'models',best_qrf_trainval[:3]+"_pred"+best_qrf_trainval[3:])).astype(np.float32)
        trainval_dict[".".join(best_qrf_trainval.split('.')[:-1])] = [qrf_trainval,"QRF"]
    elif args.pred == "test":
        best_qrf_test = "_".join(min(CRPS_qrf_test, key=CRPS_qrf_test.get).split("_")[1:])
        print("Test set :", best_qrf_test)
        qrf_test = np.load(os.path.join(root_ref,'models',best_qrf_test[:3]+"_pred"+best_qrf_test[3:])).astype(np.float32)
        test_dict[".".join(best_qrf_test.split('.')[:-1])] = [qrf_test,"QRF"]
    del CRPS_qrf_trainval, CRPS_qrf_test


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
        
        if args.pred == "trainval":
            best_qrftail_trainval = "_".join(min(CRPS_qrftail_trainval, key=CRPS_qrftail_trainval.get).split("_")[1:])
            print("Training/validation set :", best_qrftail_trainval)
            qrftail_trainval = np.load(os.path.join(root_ref,'models',best_qrftail_trainval[:(4+len(distrib))]+"_pred"+best_qrftail_trainval[(4+len(distrib)):])).astype(np.float32)
            trainval_dict[".".join(best_qrftail_trainval.split('.')[:-1])] = [qrftail_trainval,f"QRF+{distrib.upper()}"]
        elif args.pred == "test":
            best_qrftail_test = "_".join(min(CRPS_qrftail_test, key=CRPS_qrftail_test.get).split("_")[1:])
            print("Test set :", best_qrftail_test)
            qrftail_test = np.load(os.path.join(root_ref,'models',best_qrftail_test[:(4+len(distrib))]+"_pred"+best_qrftail_test[(4+len(distrib)):])).astype(np.float32)
            test_dict[".".join(best_qrftail_test.split('.')[:-1])] = [qrftail_test,f"QRF+{distrib.upper()}"]
        del CRPS_qrftail_trainval, CRPS_qrftail_test


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

        ### Select U-Net with certain parameters
        # if distrib == "gtcnd":
        #     best_unet_trainval = "UNet_gtcnd_G_trainval_d2_nl0_nc0_ns0_lr1e-05_b16_e170_nreps10.npy"
        #     best_unet_test = "UNet_gtcnd_G_test_d2_nl0_nc0_ns0_lr1e-05_b16_e170_nreps10.npy"
        # elif distrib == "csgd":
        #     best_unet_trainval = "UNet_csgd_G_trainval_d2_nl0_nc0_ns0_lr1e-05_b16_e140_nreps10.npy"
        #     best_unet_test = "UNet_csgd_G_test_d2_nl0_nc0_ns0_lr1e-05_b16_e140_nreps10.npy"
        
        if args.pred == "trainval":
            best_unet_trainval = "_".join(min(CRPS_unet_trainval, key=CRPS_unet_trainval.get).split("_")[1:])
            print("Training/validation set :", best_unet_trainval)
            del CRPS_unet_trainval, CRPS_unet_test
            params_trainval = np.nanmean(np.load(os.path.join(root_unet,"parameters","params_"+best_unet_trainval)).astype(np.float32),axis=0)
            n_trainval,d1,d2 = params_trainval.shape[:-1]
        elif args.pred == "test":
            best_unet_test = "_".join(min(CRPS_unet_test, key=CRPS_unet_test.get).split("_")[1:])
            print("Test set :", best_unet_test)
            params_test = np.nanmean(np.load(os.path.join(root_unet,"parameters","params_"+best_unet_test)).astype(np.float32),axis=0)
            n_test,d1,d2 = params_test.shape[:-1]
        del CRPS_unet_trainval, CRPS_unet_test


        thresholds = np.array(args.thresholds)
        n_thresholds = len(args.thresholds)
        if distrib=="gtcnd":
            if args.pred == "trainval":
                unet_trainval = 1-cdf_gtcnd(z=np.tile(thresholds[...,np.newaxis],reps=[1,n_trainval*d1*d2]),
                                            rho_L=np.tile(params_trainval[:,:,:,0].reshape(-1)[np.newaxis,...],reps=[n_thresholds,1]),
                                            mu=np.tile(params_trainval[:,:,:,1].reshape(-1)[np.newaxis,...],reps=[n_thresholds,1]),
                                            rho_sigma=np.tile(params_trainval[:,:,:,2].reshape(-1)[np.newaxis,...],reps=[n_thresholds,1]))
            elif args.pred == "test":
                unet_test = 1-cdf_gtcnd(z=np.tile(thresholds[...,np.newaxis],reps=[1,n_test*d1*d2]),
                                        rho_L=np.tile(params_test[:,:,:,0].reshape(-1)[np.newaxis,...],reps=[n_thresholds,1]),
                                        mu=np.tile(params_test[:,:,:,1].reshape(-1)[np.newaxis,...],reps=[n_thresholds,1]),
                                        rho_sigma=np.tile(params_test[:,:,:,2].reshape(-1)[np.newaxis,...],reps=[n_thresholds,1]))
        elif distrib=="csgd":
            if args.pred == "trainval":
                unet_trainval = 1-cdf_csgd(z=np.tile(thresholds[...,np.newaxis],reps=[1,n_trainval*d1*d2]),
                                            rho_k=np.tile(params_trainval[:,:,:,0].reshape(-1)[np.newaxis,...],reps=[n_thresholds,1]),
                                            rho_theta=np.tile(params_trainval[:,:,:,1].reshape(-1)[np.newaxis,...],reps=[n_thresholds,1]),
                                            rho_delta=np.tile(params_trainval[:,:,:,2].reshape(-1)[np.newaxis,...],reps=[n_thresholds,1]))
            elif args.pred == "test":
                unet_test = 1-cdf_csgd(z=np.tile(thresholds[...,np.newaxis],reps=[1,n_test*d1*d2]),
                                        rho_k=np.tile(params_test[:,:,:,0].reshape(-1)[np.newaxis,...],reps=[n_thresholds,1]),
                                        rho_theta=np.tile(params_test[:,:,:,1].reshape(-1)[np.newaxis,...],reps=[n_thresholds,1]),
                                        rho_delta=np.tile(params_test[:,:,:,2].reshape(-1)[np.newaxis,...],reps=[n_thresholds,1]))
        if args.pred == "trainval":
            trainval_dict[".".join(best_unet_trainval.split('.')[:-1])] = [unet_trainval,f"U-Net+{distrib.upper()}"]
        elif args.pred == "test":
            test_dict[".".join(best_unet_test.split('.')[:-1])] = [unet_test,f"U-Net+{distrib.upper()}"]

methods = "-".join(args.methods)
if args.pred == "trainval":
    print("PLOT ROC TRAINVAL")
    plot_grid_roc(
        data=trainval_dict,
        y_true=Y_trainval,
        thresholds=args.thresholds,
        path=root_out,
        file_name=f"ROC_trainval_models_{methods}",
        r=args.r,
        c=args.c
    )
elif args.pred == "test":
    print("PLOT ROC TEST")
    plot_grid_roc(
        data=test_dict,
        y_true=Y_test,
        thresholds=args.thresholds,
        path=root_out,
        file_name=f"ROC_test_models_{methods}",
        r=args.r,
        c=args.c
    )