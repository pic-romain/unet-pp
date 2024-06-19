# ---------------------------------- IMPORTS --------------------------------- #

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--methods", nargs="+",type=str, default=["unet+gtcnd","unet+csgd"])
parser.add_argument("--nreps", type=int, default=10, help="Number of repetitions")
parser.add_argument("--nfolds",type=int,default=7, help="Number of folds")
args = parser.parse_args()
print(args)

import os
import numpy as np

# ---------------------------------------------------------------------------- #


root = "../data"
root_unet = "../output/unet_models/parameters"

Y_trainval = np.load(os.path.join(root,"Y_trainval.npy"))
n_trainval, d1, d2 = Y_trainval.shape
del Y_trainval
trainval_dow = np.load(os.path.join(root,"trainval_dow.npy"))


for distrib in ["gtcnd","csgd"]:
    if f"unet+{distrib}" in args.methods:   
        unet_dict = {}
        files = [f for f in os.listdir(os.path.join(root_unet,"UNET_SEQ")) if f[:(12+len(distrib))]==f"params_UNet_{distrib}"]
        for f in files:
            params = ".".join(f[12:].split('.')[:-1]).split("_")
            try: # Cross-validation
                distrib, arch, pred, depth, nl, nc, non_separable, lr, batch_size, epochs, rep, k = params[0], params[1], params[2], int(params[3][1:]), int(params[4][2:]), int(params[5][2:]), int(params[6][2:]), float(params[7][2:]), int(params[8][1:]), int(params[9][1:]), int(params[10][3:]), int(params[11][1:])
            except IndexError: # Test
                distrib, arch, pred, depth, nl, nc, non_separable, lr, batch_size, epochs, rep = params[0], params[1], params[2], int(params[3][1:]), int(params[4][2:]), int(params[5][2:]), int(params[6][2:]), float(params[7][2:]), int(params[8][1:]), int(params[9][1:]), int(params[10][3:])

            parameters = f"params_UNet_{distrib}_{arch}_{pred}_d{depth}_nl{nl}_nc{nc}_ns{non_separable}_lr{lr}_b{batch_size}_e{epochs}"
            if parameters in unet_dict.keys():
                unet_dict[parameters].append((pred,distrib,f))
            else:
                unet_dict[parameters] = [(pred,distrib,f)]
        
        for key,v in unet_dict.items():
            print(key)
            if v[0][1]=="gtcnd":
                n_outputs = 3 # (L, mu, sigma)
            elif v[0][1]=="csgd":
                n_outputs = 3 # (k, theta, sigma)
            
            if v[0][0] == "trainval" and len(v) == args.nreps*args.nfolds:
                params_list = []
                for rep in range(1,args.nreps+1):
                    params_array_tmp = np.empty(shape=(n_trainval,d1,d2,n_outputs),dtype=np.float32)
                    for k in range(args.nfolds):
                        params_array_tmp[trainval_dow==k] = np.load(os.path.join(os.path.join(root_unet,"UNET_SEQ"),key+f"_rep{rep}_k{k}.npy"))
                    params_list.append(params_array_tmp)
                np.save(
                    os.path.join(root_unet,key+f"_nreps{args.nreps}.npy"),
                    np.array(params_list)
                )
            
            elif v[0][0] == "test" and len(v) == args.nreps:
                np.save(
                    os.path.join(root_unet,key+f"_nreps{args.nreps}.npy"),
                    np.array([np.load(os.path.join(os.path.join(root_unet,"UNET_SEQ"),key+f"_rep{rep}.npy")) for rep in range(1,args.nreps+1)])
                )
            else:
                print(f"ISSUE FOR {key}")
                print(v[0][0], len(v))
                print(v)