#!/usr/bin/env Rscript

# Section 3.2 maps: posterior means of the exact Laplace approximation and of
# hierarchical Vecchia (HV) for one simulated spatial data set, written to a
# CSV that src/python/plot_spatial_maps.py combines with NN-triple.

suppressPackageStartupMessages({
  library(GPvecchia)
  library(Matrix)
})

script_dir <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))))
source(file.path(script_dir, "utils.R"))

args <- parse_cli_args(list(n = 1600, dataset = 1, model = "gaussian",
                            data_dir = "", range = 0.15, variance = 1.0,
                            smoothness = 0.5, noise_variance = 0.2,
                            gamma_shape = 2, conditioning_sizes = "10",
                            output = ""))
args$n <- as.integer(args$n)
args$dataset <- as.integer(args$dataset)
sizes <- as.integer(strsplit(args$conditioning_sizes, ",")[[1]])
if (args$data_dir == "") args$data_dir <- sprintf("simulation_%s", args$model)
if (args$output == "") {
  args$output <- file.path(args$data_dir,
                           sprintf("fields_seed%d_n%d.csv", args$dataset, args$n))
}

data <- read.csv(file.path(
  args$data_dir,
  sprintf("%s_seed%d_domain1_dim2_n%d_smooth%.1f_range%.2f.csv",
          args$model, args$dataset, args$n, args$smoothness, args$range)))
locs <- as.matrix(data[c("x1", "x2")])
covparms <- c(args$variance, args$range, args$smoothness)
likelihood_model <- gpvecchia_likelihood(args$model)
likparms <- list(sigma = sqrt(args$noise_variance), alpha = args$gamma_shape)

posterior_mean <- function(approx) {
  fit <- GPvecchia::calculate_posterior_VL(
    z = data$z, vecchia.approx = approx, likelihood_model = likelihood_model,
    covparms = covparms, likparms = likparms, return_all = TRUE)
  as.numeric(fit$mean)
}

out <- data
out$exact <- posterior_mean(GPvecchia::vecchia_specify(
  locs, m = args$n - 1, ordering = "maxmin", conditioning = "firstm"))
hv_sizes <- data.frame(requested = sizes, hv_m = NA_integer_)
for (k in seq_along(sizes)) {
  m <- sizes[k]
  hv <- GPvecchia::vecchia_specify(locs, m = m, ordering = "maxmin",
                                   conditioning = "mra")
  hv_sizes$hv_m[k] <- ncol(hv$U.prep$revNNarray) - 1L
  cat(sprintf("%s: HV requested m = %d, achieved m = %d\n", args$model, m,
              hv_sizes$hv_m[k]))
  out[[sprintf("hv_%d", m)]] <- posterior_mean(hv)
}
write.csv(out, args$output, row.names = FALSE)
# achieved HV sizes (maximum conditioning-set size), for the panel titles
write.csv(hv_sizes, sub("\\.csv$", "_hv_m.csv", args$output), row.names = FALSE)
cat(sprintf("Wrote %s\n", args$output))
