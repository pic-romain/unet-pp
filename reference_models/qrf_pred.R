# --------------------------------- ARGUMENTS -------------------------------- #

library(optparse)
parser <- OptionParser(formatter = IndentedHelpFormatter)
parser <- add_option(parser, "--ntree", type = "integer", default = 2000, help = "Number of trees")
parser <- add_option(parser, "--mtry", type = "integer", default = 4, help = "Number of variables randomly sampled as candidates at each split")
parser <- add_option(parser, "--nodesizemin", type = "integer", default = 20, help = "Minimum size of terminal nodes")

parser <- add_option(parser, "--pred", type = "character", default = "test", help = "Prediction method")
parser <- add_option(parser, "--nfolds", type = "integer", default = 7, help = "Number of folds")
parser <- add_option(parser, "--nquantiles", type = "integer", default = (17 + 1) * 6 - 1, help = "Number of quantiles")
parser <- add_option(parser, "--ncpu", type = "integer", default = 4, help = "Number of CPUs to use")
args <- parse_args(parser)
print(args)

# ---------------------------------- IMPORTS --------------------------------- #

library(parallel)
library(foreach)
library(reticulate)
library(ranger)

# ---------------------------------------------------------------------------- #
#                                   FUNCTIONS                                  #
# ---------------------------------------------------------------------------- #

.qrf.fit <- function(x, y, mtry, node.size, quantiles, n_trees, x.test) {
    df_train <- data.frame(cbind(y, x))
    df_test <- data.frame(x.test)
    colnames(df_train) <- c("y", paste0("x", 1:dim(x)[2]))
    colnames(df_test) <- paste0("x", 1:dim(x.test)[2])
    # Training
    qrf_fit <- ranger::ranger(y ~ ., data = df_train, num.trees = n_trees, mtry = mtry, min.node.size = node.size, quantreg = TRUE)

    # OOB Prediction of quantiles
    quantile_prediction <- predict(qrf_fit, data = df_test, type = "quantiles", quantiles = quantiles)$predictions

    return(quantile_prediction)
}

# ---------------------------------------------------------------------------- #

np <- reticulate::import("numpy")
root <- "../data/"
root_out <- "../output/reference_models/models/"

# Initialize parallel computing
cl <- parallel::makeCluster(args$ncpu)
doParallel::registerDoParallel(cl)

# ---------------------------------------------------------------------------- #
#                               CROSS-VALIDATION                               #
# ---------------------------------------------------------------------------- #
if (args$pred == "cv") {
    print("K-FOLD CV")
    X_trainval <- np$load(paste0(root, "X_trainval.npy"))
    Y_trainval <- np$load(paste0(root, "Y_trainval.npy"))
    fold <- np$load(paste0(root, "trainval_dow.npy"))

    coord <- expand.grid(0:(args$nfolds - 1), 1:dim(X_trainval)[2], 1:dim(X_trainval)[3])
    out <- foreach(pos = 1:dim(coord)[1]) %dopar% {
        k <- coord[pos, 1]
        i <- coord[pos, 2]
        j <- coord[pos, 3]

        .qrf.fit(x = X_trainval[fold != k, i, j,], y = Y_trainval[fold != k, i, j], mtry = args$mtry, node.size = args$nodesizemin, quantiles = seq(1 / (args$nquantiles + 1), args$nquantiles / (args$nquantiles + 1), 1 / (args$nquantiles + 1)), n_trees = args$ntree, x.test = X_trainval[fold == k, i, j,])
    }
    parallel::stopCluster(cl)

    print("SAVING THE RESULTS")
    qrf_pred <- array(numeric(), c(dim(X_trainval)[1:3], args$nquantiles))
    for (pos in 1:dim(coord)[1]) {
        k <- coord[pos, 1]
        i <- coord[pos, 2]
        j <- coord[pos, 3]

        qrf_pred[fold == k, i, j,] <- out[[pos]]
    }
    np$save(paste0(root_out, "qrf_pred_trainval_q", args$nquantiles, "_ntree", args$ntree, "_mtry", args$mtry, "_nodesize", args$nodesizemin, ".npy"), qrf_pred)


# ---------------------------------------------------------------------------- #
#                                     TEST                                     #
# ---------------------------------------------------------------------------- #
} else if (args$pred == "test") {
    print("TEST")
    X_trainval <- np$load(paste0(root, "X_trainval.npy"))
    Y_trainval <- np$load(paste0(root, "Y_trainval.npy"))
    X_test <- np$load(paste0(root, "X_test.npy"))

    coord <- expand.grid(1:dim(X_trainval)[2], 1:dim(X_trainval)[3])
    out <- foreach(pos = 1:dim(coord)[1]) %dopar% {
        i <- coord[pos, 1]
        j <- coord[pos, 2]

        .qrf.fit(x = X_trainval[, i, j,], y = Y_trainval[, i, j], mtry = args$mtry, node.size = args$nodesizemin, quantiles = seq(1 / (args$nquantiles + 1), args$nquantiles / (args$nquantiles + 1), 1 / (args$nquantiles + 1)), n_trees = args$ntree, x.test = X_test[, i, j,])
    }
    parallel::stopCluster(cl)

    print("SAVING THE RESULTS")
    qrf_pred <- array(numeric(), c(dim(X_test)[1:3], args$nquantiles))
    for (pos in 1:dim(coord)[1]) {
        i <- coord[pos, 1]
        j <- coord[pos, 2]

        qrf_pred[, i, j,] <- out[[pos]]
    }
    np$save(paste0(root_out, "qrf_pred_test_q", args$nquantiles, "_ntree", args$ntree, "_mtry", args$mtry, "_nodesize", args$nodesizemin, ".npy"), qrf_pred)

} else {
    parallel::stopCluster(cl)
    stop("Invalid prediction method")
}
