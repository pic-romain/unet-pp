# ---------------------------------------------------------------------------- #
#                                    IMPORTS                                   #
# ---------------------------------------------------------------------------- #

import numpy as np

# ---------------------------------------------------------------------------- #
#                                   FUNCTIONS                                  #
# ---------------------------------------------------------------------------- #

def train_val_split(X_trainval,Y_trainval,trainval_dow,k):
    train_index = np.where(trainval_dow!=k)[0]
    val_index = np.where(trainval_dow==k)[0] 

    X_train, X_val = X_trainval[train_index], X_trainval[val_index]
    Y_train, Y_val = Y_trainval[train_index], Y_trainval[val_index]
    return X_train, X_val, Y_train, Y_val, train_index, val_index