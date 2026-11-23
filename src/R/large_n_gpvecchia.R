#!/usr/bin/env Rscript

# Section 3.6, extra check (CPU): HV and LR with GPvecchia's compiled code on
# the data sets of src/python/large_n_spatial.py. calculate_posterior_VL
# returns the posterior mean and the posterior precision factor V; GPvecchia
# does not compute the covariance factor V^{-T}, so these times exclude
# Stage 3 and favor HV and LR. The ordering and conditioning sets
# (vecchia_specify) are timed separately.

suppressPackageStartupMessages({
  library(GPvecchia)
  library(Matrix)
})

script_dir <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))))
source(file.path(script_dir, "utils.R"))

args <- parse_cli_args(list(side = 100, models = "gaussian,logistic,poisson,gamma",
                            n_datasets = 10, conditioning_sizes = "5,10,15,20,25,35,50,80,120",
                            data_dir = "large_n/data", output = "",
                            range = 0.15, smoothness = 0.5, noise_variance = 0.2, gamma_shape = 2,
                            lr_sizes = "", model = "gaussian"))
side <- as.integer(args$side)
n <- side^2
models <- strsplit(args$models, ",")[[1]]
sizes <- as.integer(strsplit(args$conditioning_sizes, ",")[[1]])
if (args$output == "") args$output <- sprintf("large_n/gpvecchia_side%d.csv", side)
# range in the GPvecchia::MaternFun parameterization (distances scaled by
# sqrt(2 * smoothness)); smoothness 1.5 needs range = sqrt(3) * python range
covparms <- c(1, as.numeric(args$range), as.numeric(args$smoothness))
likparms <- list(sigma = sqrt(args$noise_variance), alpha = args$gamma_shape)

grid <- seq(0, 1, length.out = side)
locs <- as.matrix(expand.grid(x1 = grid, x2 = grid))

approximations <- list()
for (m in sizes) {
  start <- proc.time()[["elapsed"]]
  hv <- vecchia_specify(locs, m = m, ordering = "maxmin", conditioning = "mra")
  hv_time <- proc.time()[["elapsed"]] - start
  N <- ncol(hv$U.prep$revNNarray) - 1L
  start <- proc.time()[["elapsed"]]
  lr <- vecchia_specify(locs, m = N, ordering = "maxmin", conditioning = "firstm")
  lr_time <- proc.time()[["elapsed"]] - start
  approximations[[length(approximations) + 1]] <- list(name = "HV", m = m, N = N, approx = hv, pattern_time = hv_time)
  if (args$lr_sizes == "")
    approximations[[length(approximations) + 1]] <- list(name = "LR", m = m, N = N, approx = lr, pattern_time = lr_time)
}
# LR sizes N independent of HV (matched to NN-triple on nonzeros per row)
for (N in if (args$lr_sizes == "") integer(0) else as.integer(strsplit(args$lr_sizes, ",")[[1]])) {
  start <- proc.time()[["elapsed"]]
  lr <- vecchia_specify(locs, m = N, ordering = "maxmin", conditioning = "firstm")
  approximations[[length(approximations) + 1]] <- list(name = "LR", m = N, N = N, approx = lr,
                                                       pattern_time = proc.time()[["elapsed"]] - start)
}

# Warm up
invisible(calculate_posterior_VL(rnorm(n), approximations[[1]]$approx, "gaussian",
                                 covparms = covparms, likparms = likparms))

results <- list()
for (d in seq_len(args$n_datasets)) {
  for (model in models) {
    con <- file(file.path(args$data_dir, sprintf("side%d", side),
                          sprintf("%s_d%d.bin", model, d)), "rb")
    v <- readBin(con, "double", n = 2 * n)
    close(con)
    x <- v[1:n]
    z <- v[(n + 1):(2 * n)]
    for (a in approximations) {
      start <- proc.time()[["elapsed"]]
      fit <- calculate_posterior_VL(z, a$approx, gpvecchia_likelihood(model),
                                    covparms = covparms, likparms = likparms,
                                    return_all = TRUE)
      elapsed <- proc.time()[["elapsed"]] - start
      ls <- -dposterior(x, fit)
      results[[length(results) + 1]] <- data.frame(
        side = side, n = n, model = model, dataset = d, method = a$name,
        param = a$m, N = a$N, time_total = elapsed, pattern_time = a$pattern_time,
        ls = ls, rmspe = sqrt(mean((as.numeric(fit$mean) - x)^2)),
        converged = fit$cnvgd)
      cat(sprintf("side %d %s d%d %s m=%d (N=%d): %.2f s, LS %.1f\n", side, model, d,
                  a$name, a$m, a$N, elapsed, ls))
    }
    write.csv(do.call(rbind, results), args$output, row.names = FALSE)
  }
}
