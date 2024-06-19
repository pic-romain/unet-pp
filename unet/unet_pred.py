# --------------------------------- ARGUMENTS -------------------------------- #

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--depth", type=int, choices=range(1,6), default=2, help="Depth of the UNet")
parser.add_argument("--no_log", action="store_true", help="Not using log-transformed precipitation in the predictors")
parser.add_argument("--no_constant", action="store_true", help="Not using constant fields for prediction")
parser.add_argument("-ns", "--non_separable", action="store_true", help="Use non-separable convolution in the U-Net")
parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4, help="Learning rate of the Gradient Descent")
parser.add_argument("-b", "--batch_size",  type=int,  default=16, help="Batch size")
parser.add_argument("-e", "--epochs", type=int, default=1, help="Number of training epochs")

parser.add_argument("--pred", type=str, choices=["cv","test"], default="cv")
parser.add_argument("--distrib", type=str, choices=["gtcnd","csgd"], default="gtcnd")
parser.add_argument("--rep", type=int, default=1, help="Repetition number")
parser.add_argument("--nfolds", type=int, default=7, help="Number of folds for cross-validation")
parser.add_argument("--k", type=int, default=0, help="Fold number for cross-validation")
parser.add_argument("--save_freq", type=int, default=5, help="Frequency of saving the parameters in epochs")
parser.add_argument("--verbose", type=int, choices=[0,1,2], default=2)
args = parser.parse_args()
print(args)

# ---------------------------------- IMPORTS --------------------------------- #

import os,sys
sys.path.insert(1,'../utils')

from learning import train_val_split
from unet import UNet2D_Groenquist, rescale_data, NN_gtcnd_model, CRPS_gtcnd, NN_csgd_model, CRPS_csgd

import tensorflow as tf
from tensorflow import keras
import numpy as np
tf.keras.backend.set_floatx('float32')

# ---------------------------------------------------------------------------- #

k = args.k

root = "../data"
root_out = "../output/unet_models"

if args.distrib == "gtcnd":
    n_outputs = 3 # (L, mu, sigma)
    NN_model = NN_gtcnd_model
    CRPS_loss = CRPS_gtcnd
elif args.distrib == "csgd":
    n_outputs = 3 # (k, theta, delta)
    NN_model = NN_csgd_model
    CRPS_loss = CRPS_csgd

# ---------------------------------------------------------------------------- #
#                               CROSS-VALIDATION                               #
# ---------------------------------------------------------------------------- #

if args.pred=="cv":
    print(f"CROSS-VALIDATION K={k+1}")

    # ---------------------------- Importing the data ---------------------------- #

    X_trainval = np.load(os.path.join(root,"X_trainval.npy"))
    Y_trainval = np.load(os.path.join(root,"Y_trainval.npy"))
    trainval_dow = np.load(os.path.join(root,"trainval_dow.npy"))
    n_trainval,d1,d2 = X_trainval.shape[:-1]
    
    if not args.no_constant:
        # Import constant fields
        constant_fields = np.load("../data/X_constant.npy")
        X_trainval = np.concatenate(
            [
                X_trainval,
                np.tile(constant_fields[np.newaxis,...],reps=[n_trainval,1,1,1])
            ],
            axis=-1
        )
    
    # ------------------------- Creating the U-Net model ------------------------- #

    opt = keras.optimizers.Adam(learning_rate=args.learning_rate)
    UNET_ref = UNet2D_Groenquist((d1,d2),
                                num_params=n_outputs,in_channels=X_trainval.shape[-1],
                                filters_list=[2**(i+6) for i in range(args.depth)],
                                separable=not(args.non_separable))
    model_ref = NN_model(NN=UNET_ref)
    model_ref.compile(optimizer=opt, loss=CRPS_loss, run_eagerly=False)
    
    # ---------------------------- Rescaling the data ---------------------------- #

    X_train, X_val, Y_train, Y_val, train_index, val_index = train_val_split(X_trainval,Y_trainval,trainval_dow,k)
    X_train, X_val = rescale_data(X_train,X_val) # Rescaling
    n_val = len(val_index)

    for q,j in enumerate(range(0,args.epochs,args.save_freq if args.save_freq>0 else args.epochs)):
        print(f"EPOCH {j}-{j+args.save_freq}/{args.epochs}")
        model_ref.fit(X_train, Y_train, epochs=args.save_freq if args.save_freq>0 else args.epochs, batch_size=args.batch_size,verbose=args.verbose)
        params_val = model_ref.NN.predict(X_val, batch_size=args.batch_size,verbose=args.verbose)
        
        # Save params
        n_epochs = (j + args.save_freq) if args.save_freq>0 else args.epochs
        np.save(os.path.join(root_out,f"parameters/UNET_SEQ/params_UNet_{args.distrib}_G_trainval_d{args.depth}_nl{int(args.no_log)}_nc{int(args.no_constant)}_ns{int(args.non_separable)}_lr{args.learning_rate}_b{args.batch_size}_e{n_epochs}_rep{args.rep}_k{k}.npy"),params_val)

# ---------------------------------------------------------------------------- #
#                                     TEST                                     #
# ---------------------------------------------------------------------------- #

elif args.pred=="test":
    print("TEST")
    # ---------------------------- Importing the data ---------------------------- #

    X_trainval = np.load(os.path.join(root,"X_trainval.npy"))
    X_test = np.load(os.path.join(root,"X_test.npy"))
    Y_trainval = np.load(os.path.join(root,"Y_trainval.npy"))
    Y_test = np.load(os.path.join(root,"Y_test.npy"))
    n_trainval,d1,d2 = X_trainval.shape[:-1]
    n_test = X_test.shape[0]
    
    if not args.no_constant:
        # Import constant fields
        constant_fields = np.load("../data/X_constant.npy")
        X_trainval = np.concatenate([X_trainval,np.tile(constant_fields[np.newaxis,...],reps=[X_trainval.shape[0],1,1,1])],axis=-1)
        X_test = np.concatenate([X_test,np.tile(constant_fields[np.newaxis,...],reps=[X_test.shape[0],1,1,1])],axis=-1)
    
    # ------------------------- Creating the U-Net model ------------------------- #

    opt = keras.optimizers.Adam(learning_rate=args.learning_rate)
    UNET_ref = UNet2D_Groenquist((d1,d2),
                                num_params=n_outputs,in_channels=X_trainval.shape[-1],
                                filters_list=[2**(i+6) for i in range(args.depth)],
                                separable=not(args.non_separable))
    model_ref = NN_model(NN=UNET_ref)
    model_ref.compile(optimizer=opt, loss=CRPS_loss, run_eagerly=False)

    # ---------------------------- Rescaling the data ---------------------------- #

    X_trainval, X_test = rescale_data(X_trainval,X_test) # Rescaling
    for q,j in enumerate(range(0,args.epochs,args.save_freq if args.save_freq>0 else args.epochs)):
        print(f"EPOCH {j}-{j+args.save_freq}/{args.epochs}")

        model_ref.fit(
            X_trainval, Y_trainval,
            epochs=args.save_freq if args.save_freq>0 else args.epochs,
            batch_size=args.batch_size,
            verbose=args.verbose
        )
        params_test = model_ref.NN.predict(
            X_test,
            batch_size=args.batch_size,
            verbose=args.verbose
        )
        
        # Save params
        n_epochs = (j + args.save_freq) if args.save_freq>0 else args.epochs
        np.save(os.path.join(root_out,f"parameters/UNET_SEQ/params_UNet_{args.distrib}_G_test_d{args.depth}_nl{int(args.no_log)}_nc{int(args.no_constant)}_ns{int(args.non_separable)}_lr{args.learning_rate}_b{args.batch_size}_e{n_epochs}_rep{args.rep}.npy"),params_test)