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

# General fitting function for GTCND
.gtcnd.pw.fit <- function(data, init) {
    fit.PWM <- .gtcnd.fitPWM(x = data, mu0 = init[1], sigma0 = init[2])
    fits <- list(PWM = fit.PWM)
    return(list(fit = fits))
}

# Moments method for GTCND
.PWM <- function(orders, mu = NA, sigma = NA, empiric = FALSE, Unif = NULL, NbSamples = 10 ^ 4, N = 200) {
    if (!empiric) {
        pdf <- dnorm(mu / sigma)
        cdf <- pnorm(-mu / sigma)
        mu0 <- mu + pdf * sigma / (1 - cdf)
        mu1 <- sigma ^ 2 * (1 - mu / sigma * pdf / (1 - cdf) - (pdf / (1 - cdf)) ^ 2)
        return(c(mu0, mu1))
    } else {
        if (is.null(Unif)) {
            Unif <- runif(NbSamples)
        }

        X <- .qtruncnorm(p = Unif, mu = mu, sigma = sigma)
        res <- c(mean(X), var(X))

        return(res)
    }
}

# Fit GTCND parameters
.gtcnd.fitPWM <- function(x, mu0 = NA, sigma0 = NA, empiric = FALSE, Unif = NULL, NbSamples = 10 ^ 4) {
    mu0hat <- mean(x)
    mu1hat <- var(x)

    fct <- function(theta, x) {
        pwm.theor <- .PWM(mu = theta[1], sigma = theta[2], empiric = empiric, Unif = Unif, NbSamples = NbSamples)
        pwm.empir <- c(mu0hat, mu1hat)
        return(matrix(pwm.theor - pwm.empir, ncol = 2))
    }
    theta0 <- c(mu0, sigma0)
    res <- gmm::gmm(fct, x, theta0, optfct = "nlminb", lower = c(-100, 0.0001), upper = c(100, 20), vcov = "iid")
    thetahat <- res$coefficients
    names(thetahat) <- c("mu", "sigma")
    return(thetahat)
}

# Quantile function for GTCND
.qtruncnorm <- function(p, mu, sigma) {
    cdf <- pnorm(-mu / sigma)
    return(mu + sigma * qnorm(p + cdf * (1 - p)))
}

# QRF+GTCND fitting function
.qrf.gtcnd.fit <- function(x, y, mtry, node.size, quantiles, n_trees, x.test) {
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
    used_gtcnd <- rep(TRUE, nrow(qrf_pred))
    qrf_gtcnd_pred <- matrix(, nrow = nrow(qrf_pred), ncol = length(quantiles))
    for (r in 1:nrow(qrf_pred)) {
        gtcnd <- TRUE
        obs <- qrf_pred[r,]
        obsn0 <- obs[obs >= 0.05]
        obsn1 <- obs[obs > 10]
        if ((length(obsn0) >= trunc(n_quantiles_int * 0.2)) && (length(obsn1) >= trunc(n_quantiles_int * 0.02))) {
            # Fit gtcnd
            params_gtcnd <- .gtcnd.pw.fit(data = obsn0, init = c(0, 1))$fit$PWM
            # Predict quantiles from gtcnd for quantiles higher than 10mm
            qrf_pred[r, (n_quantiles_int - length(obsn1) + 1):n_quantiles_int] <- .qtruncnorm(seq(1 - length(obsn1) / (n_quantiles_int + 1), n_quantiles_int / (n_quantiles_int + 1), length.out = length(obsn1)), mu = params_gtcnd[1], sigma = params_gtcnd[2]) # nolinter

            # Only change quantiles that are predicted higher by the gtcnd
            diff <- obs - qrf_pred[r,]
            sus <- which(diff > 0)
            qrf_pred[r, sus] <- obs[sus]

            qrf_gtcnd_pred[r,] <- as.numeric(quantile(qrf_pred[r,], probs = quantiles, na.rm = TRUE, type = 6))
        } else {
            gtcnd <- FALSE
            qrf_gtcnd_pred[r,] <- as.numeric(quantile(qrf_pred[r,], probs = quantiles, na.rm = TRUE, type = 6))
        }
        used_gtcnd[r] <- gtcnd
    }
    return(cbind(qrf_gtcnd_pred, used_gtcnd))
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

        .qrf.gtcnd.fit(X_trainval[fold != k, i, j,], y = Y_trainval[fold != k, i, j], mtry = args$mtry, node.size = args$nodesizemin, quantiles = seq(1 / (args$nquantiles + 1), args$nquantiles / (args$nquantiles + 1), 1 / (args$nquantiles + 1)), n_trees = args$ntree, x.test = X_trainval[fold == k, i, j,]) # nolinter
    }
    parallel::stopCluster(cl)

    print("SAVING THE RESULTS")
    qrfgtcnd_pred <- array(numeric(), c(dim(X_trainval)[1:3], args$nquantiles))
    gtcnd_activation <- array(numeric(), dim(X_trainval)[1:3])
    for (pos in 1:dim(coord)[1]) {
        k <- coord[pos, 1]
        i <- coord[pos, 2]
        j <- coord[pos, 3]

        qrfgtcnd_pred[fold == k, i, j,] <- out[[pos]][, 1:args$nquantiles]
        gtcnd_activation[fold == k, i, j] <- out[[pos]][, args$nquantiles + 1]
    }
    np$save(paste0(root_out, "qrf+gtcnd_pred_trainval_q", args$nquantiles, "_ntree", args$ntree, "_mtry", args$mtry, "_nodesize", args$nodesizemin, ".npy"), qrfgtcnd_pred)
    np$save(paste0(root_out, "gtcnd_activation_trainval_q", args$nquantiles, "_ntree", args$ntree, "_mtry", args$mtry, "_nodesize", args$nodesizemin, ".npy"), gtcnd_activation)

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

        .qrf.gtcnd.fit(x = X_trainval[, i, j,], y = Y_trainval[, i, j], mtry = args$mtry, node.size = args$nodesizemin, quantiles = seq(1 / (args$nquantiles + 1), args$nquantiles / (args$nquantiles + 1), 1 / (args$nquantiles + 1)), n_trees = args$ntree, x.test = X_test[, i, j,])
    }
    parallel::stopCluster(cl)
    
    print("SAVING THE RESULTS")
    qrfgtcnd_pred <- array(numeric(), c(dim(X_test)[1:3], args$nquantiles))
    gtcnd_activation <- array(numeric(), dim(X_test)[1:3])
    for (pos in 1:dim(coord)[1]) {
        i <- coord[pos, 1]
        j <- coord[pos, 2]

        qrfgtcnd_pred[, i, j,] <- out[[pos]][, 1:args$nquantiles]
        gtcnd_activation[, i, j] <- out[[pos]][, args$nquantiles + 1]
    }
    np$save(paste0(root_out, "qrf+gtcnd_pred_test_q", args$nquantiles, "_ntree", args$ntree, "_mtry", args$mtry, "_nodesize", args$nodesizemin, ".npy"), qrfgtcnd_pred)
    np$save(paste0(root_out, "gtcnd_activation_test_q", args$nquantiles, "_ntree", args$ntree, "_mtry", args$mtry, "_nodesize", args$nodesizemin, ".npy"), gtcnd_activation)

} else {
    parallel::stopCluster(cl)
    stop("Invalid prediction method")
}