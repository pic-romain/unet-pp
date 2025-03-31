# --------------------------------- ARGUMENTS -------------------------------- #

library(optparse)
parser <- OptionParser(formatter = IndentedHelpFormatter)
parser <- add_option(parser, "--ntree", type = "integer", default = 2000, help = "Number of trees")
parser <- add_option(parser, "--mtry", type = "integer", default = 4, help = "Number of variables randomly sampled as candidates at each split")
parser <- add_option(parser, "--nodesizemin", type = "integer", default = 20, help = "Minimum size of terminal nodes")

parser <- add_option(parser, "--pred", type = "character", default = "test", help = "Prediction method")
parser <- add_option(parser, "--nfolds", type = "integer", default = 7, help = "Number of folds")
parser <- add_option(parser, "--nquantiles", type = "integer", default = (17 + 1) * 6 - 1, help = "Number of quantiles")
parser <- add_option(parser, "--ncpu", type = "integer", default = 16, help = "Number of cpus to use")
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

.csgd.pw.fit <- function(data, init) {
    fit.PWM <- .csgd.fitPWM(x = data, k0 = init[1], theta0 = init[2], delta0 = init[3])
    fits <- list(PWM = fit.PWM)
    return(list(fit = fits))
}

.PWM <- function(k = NA, theta = NA, delta = NA, empiric = FALSE, Unif = NULL, NbSamples = 10 ^ 4, N = 200) {
    if (!empiric) {
        cdf0 <- .pcsgd(x = 0, k = k, theta = theta, delta = delta)
        cdf1 <- .pcsgd(x = 0, k = k + 1, theta = theta, delta = delta)
        cdf2 <- .pcsgd(x = 0, k = k + 2, theta = theta, delta = delta)
        cdf3 <- .pcsgd(x = 0, k = k + 3, theta = theta, delta = delta)
        mu0 <- (1 - cdf0) * (k * theta * (1 - cdf1) - delta * (1 - cdf0))
        mu1 <- (1 - cdf0) * (k * (k + 1) * theta ^ 2 * (1 - cdf2) - 2 * delta * k * theta * (1 - cdf1) + delta ^ 2 * (1 - cdf0))
        mu2 <- (1 - cdf0) * (k * (k + 1) * (k + 2) * theta ^ 3 * (1 - cdf3) - 3 * delta * k * (k + 1) * theta ^ 2 * (1 - cdf2) + 3 * delta ^ 2 * k * theta * (1 - cdf1) - delta ^ 3 * (1 - cdf0))
        return(c(mu0, mu1, mu2))
    } else {
        if (is.null(Unif)) {
            Unif <- runif(NbSamples)
        }
        X <- .qcsgd(p = Unif, k = k, theta = theta, delta = delta)
        res <- c(mean(X), mean(X ^ 2), mean(X ^ 3))
        return(res)
    }
}

.csgd.fitPWM <- function(x, k0 = NA, theta0 = NA, delta0 = NA, empiric = FALSE, Unif = NULL, NbSamples = 10 ^ 4) {
    mu0hat <- mean(x)
    mu1hat <- mean(x ^ 2)
    mu2hat <- mean(x ^ 3)

    fct <- function(param, x) {
        pwm.theor <- .PWM(k = param[1], theta = param[2], delta = param[3], empiric = empiric, Unif = Unif, NbSamples = NbSamples)
        pwm.empir <- c(mu0hat, mu1hat, mu2hat)
        return(matrix(pwm.theor - pwm.empir, ncol = 3))
    }
    param0 <- c(k0, theta0, delta0)
    res <- gmm::gmm(fct, x, param0, optfct = "nlminb", lower = c(0.0001, 0.0001, 0.0001), upper = c(100, 100, 100), vcov = "iid")
    thetahat <- res$coefficients
    names(thetahat) <- c("k", "theta", "delta")
    return(thetahat)
}

.qcsgd <- function(p, k, theta, delta) {
    return(ifelse(p >= pgamma(delta, shape = k, scale = theta), - delta + qgamma(p = p, shape = k, scale = theta), 0))
}
.pcsgd <- function(x, k, theta, delta) {
    cdf <- pgamma(x + delta, shape = k, scale = theta)
    return(ifelse(x >= 0, cdf, 0))
}

.qrf.csgd.fit <- function(x, y, mtry, node.size, quantiles, n_trees, x.test) {
    # Data to data.frame
    df_train <- data.frame(cbind(y, x))
    df_test <- data.frame(x.test)
    colnames(df_train) <- c("y", paste0("x", 1:dim(x)[2]))
    colnames(df_test) <- paste0("x", 1:dim(x.test)[2])

    # QRF fit
    qrf_fit <- ranger::ranger(y ~ ., data = df_train, num.trees = n_trees, mtry = mtry, min.node.size = node.size, quantreg = TRUE)
    quantile_prediction <- predict(qrf_fit, data = df_test, type = "quantiles", quantiles = 1:500/501)$predictions
    qrf_pred <- as.matrix(quantile_prediction)
    n_quantiles_int <- ncol(qrf_pred)
    used_csgd <- rep(TRUE, nrow(qrf_pred))
    qrf_csgd_pred <- matrix(, nrow = nrow(qrf_pred), ncol = length(quantiles))
    for (r in 1:nrow(qrf_pred)) {
        csgd <- TRUE
        obs <- qrf_pred[r,]
        obsn0 <- obs[obs >= 0.05] # predictions above 0.05mm
        obsn1 <- obs[obs > 10] # predictions above 10mm
        if ((length(obsn0) >= trunc(n_quantiles_int * 0.2)) && (length(obsn1) >= trunc(n_quantiles_int * 0.02))) {
            # Fit CSGD on the whole data
            params_csgd <- .csgd.pw.fit(data = obs, init = c(1, 1, 0.1))$fit$PWM
            # Predict quantiles from CSGD for quantiles higher than 10mm
            qrf_pred[r, (n_quantiles_int - length(obsn1) + 1):n_quantiles_int] <- .qcsgd(seq(1 - length(obsn1) / (n_quantiles_int + 1), n_quantiles_int / (n_quantiles_int + 1), length.out = length(obsn1)), k = params_csgd[1], theta = params_csgd[2], delta = params_csgd[3])

            # Only change quantiles that are predicted higher by the CSGD
            diff <- obs - qrf_pred[r,]
            sus <- which(diff > 0)
            qrf_pred[r, sus] <- obs[sus]

            qrf_csgd_pred[r,] <- as.numeric(quantile(qrf_pred[r,], probs = quantiles, na.rm = TRUE, type = 6))
        } else {
            csgd <- FALSE
            qrf_csgd_pred[r,] <- as.numeric(quantile(qrf_pred[r,], probs = quantiles, na.rm = TRUE, type = 6))
        }
        used_csgd[r] <- csgd
    }
    return(cbind(qrf_csgd_pred, used_csgd))
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

        .qrf.csgd.fit(X_trainval[fold != k, i, j,], y = Y_trainval[fold != k, i, j], mtry = args$mtry, node.size = args$nodesizemin, quantiles = seq(1 / (args$nquantiles + 1), args$nquantiles / (args$nquantiles + 1), 1 / (args$nquantiles + 1)), n_trees = args$ntree, x.test = X_trainval[fold == k, i, j,])
    }
    parallel::stopCluster(cl)

    print("SAVING THE RESULTS")
    qrfcsgd_pred <- array(numeric(), c(dim(X_trainval)[1:3], args$nquantiles))
    csgd_activation <- array(numeric(), dim(X_trainval)[1:3])
    for (pos in 1:dim(coord)[1]) {
        k <- coord[pos, 1]
        i <- coord[pos, 2]
        j <- coord[pos, 3]

        qrfcsgd_pred[fold == k, i, j,] <- out[[pos]][, 1:args$nquantiles]
        csgd_activation[fold == k, i, j] <- out[[pos]][, args$nquantiles + 1]
    }
    np$save(paste0(root_out, "qrf+csgd_pred_trainval_q", args$nquantiles, "_ntree", args$ntree, "_mtry", args$mtry, "_nodesize", args$nodesizemin, ".npy"), qrfcsgd_pred)
    np$save(paste0(root_out, "csgd_activation_trainval_q", args$nquantiles, "_ntree", args$ntree, "_mtry", args$mtry, "_nodesize", args$nodesizemin, ".npy"), csgd_activation)


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

        .qrf.csgd.fit(x = X_trainval[, i, j,], y = Y_trainval[, i, j], mtry = args$mtry, node.size = args$nodesizemin, quantiles = seq(1 / (args$nquantiles + 1), args$nquantiles / (args$nquantiles + 1), 1 / (args$nquantiles + 1)), n_trees = args$ntree, x.test = X_test[, i, j,])
    }
    parallel::stopCluster(cl)

    print("SAVING THE RESULTS")
    qrfcsgd_pred <- array(numeric(), c(dim(X_test)[1:3], args$nquantiles))
    csgd_activation <- array(numeric(), dim(X_test)[1:3])
    for (pos in 1:dim(coord)[1]) {
        i <- coord[pos, 1]
        j <- coord[pos, 2]

        qrfcsgd_pred[, i, j,] <- out[[pos]][, 1:args$nquantiles]
        csgd_activation[, i, j] <- out[[pos]][, args$nquantiles + 1]
    }
    np$save(paste0(root_out, "qrf+csgd_pred_test_q", args$nquantiles, "_ntree", args$ntree, "_mtry", args$mtry, "_nodesize", args$nodesizemin, ".npy"), qrfcsgd_pred)
    np$save(paste0(root_out, "csgd_activation_test_q", args$nquantiles, "_ntree", args$ntree, "_mtry", args$mtry, "_nodesize", args$nodesizemin, ".npy"), csgd_activation)

} else {
    parallel::stopCluster(cl)
    stop("Invalid prediction method")
}