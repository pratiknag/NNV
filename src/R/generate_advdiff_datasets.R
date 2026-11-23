#!/usr/bin/env Rscript

# Section 3.3: latent fields evolving by linear advection-diffusion, observed
# at a random fraction of the grid at each time point.

suppressPackageStartupMessages({
  library(GPvecchia)
  library(fields)
  library(Matrix)
})

script_dir <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))))
source(file.path(script_dir, "utils.R"))

parse_args <- function() {
  defaults <- parse_cli_args(list(n = 1600, n_datasets = 80, n_times = 20,
                   model = "gaussian", output_dir = "",
                   range = 0.15, variance = 1.0, smoothness = 0.5,
                   noise_variance = 0.25, gamma_shape = 2, frac_obs = 0.1,
                   diffusion = 4e-5, advection = 1e-2, seed = 1988))
  if (defaults$output_dir == "") {
    defaults$output_dir <- sprintf("filtering_%s", defaults$model)
  }
  defaults
}

args <- parse_args()
if (args$n != as.integer(sqrt(args$n))^2) {
  stop("n must be a perfect square for the regular spatial grid.")
}
model_id <- match(args$model, data_models)
dir.create(args$output_dir, recursive = TRUE, showWarnings = FALSE)
set.seed(args$seed)
side <- sqrt(args$n)
grid <- seq(0, 1, length.out = side)
locs <- as.matrix(expand.grid(x1 = grid, x2 = grid))
covparms <- c(args$variance, args$range, args$smoothness)
# Sigma_0 = Q_1 = ... = Q_T
covariance <- GPvecchia::MaternFun(fields::rdist(locs), covparms)
chol_covariance <- chol(covariance)
evolution <- advection_diffusion_matrix(args$n, args$advection, args$diffusion)
n_obs <- round(args$frac_obs * args$n)

# The evolution matrix is shared with the Python filter (0-based indices).
evolution_triplets <- Matrix::summary(evolution)
write.csv(data.frame(i = evolution_triplets$i - 1, j = evolution_triplets$j - 1,
                     x = evolution_triplets$x),
          file.path(args$output_dir, sprintf("evolution_n%d.csv", args$n)),
          row.names = FALSE)

for (dataset_id in seq_len(args$n_datasets)) {
  frames <- vector("list", args$n_times)
  latent <- rep(0, args$n)
  for (t in seq_len(args$n_times)) {
    innovation <- as.numeric(t(chol_covariance) %*% rnorm(args$n))
    latent <- if (t == 1) innovation else as.numeric(evolution %*% latent) + innovation
    obs_inds <- sample(args$n, n_obs)
    # The Gaussian noise and the observed locations are always drawn so that
    # the main RNG stream, and hence every latent field, is identical across
    # data models.
    noise <- rnorm(n_obs, sd = sqrt(args$noise_variance))
    if (args$model == "gaussian") {
      values <- latent[obs_inds] + noise
    } else {
      # Non-Gaussian observations come from a separate per-(dataset, time)
      # seed; the main stream is restored afterwards.
      main_state <- .Random.seed
      set.seed(args$seed + 100000 * model_id + 1000 * dataset_id + t)
      values <- simulate_observations(args$model, latent[obs_inds], noise,
                                      args$gamma_shape)
      assign(".Random.seed", main_state, envir = .GlobalEnv)
    }
    observations <- rep(NA_real_, args$n)
    observations[obs_inds] <- values
    frames[[t]] <- data.frame(x1 = locs[, 1], x2 = locs[, 2], t = t,
                              y = latent, z = observations)
  }
  output_path <- file.path(
    args$output_dir,
    sprintf("%s_seed%d_n%d_T%d_range%.2f.csv",
            args$model, dataset_id, args$n, args$n_times, args$range)
  )
  write.csv(do.call(rbind, frames), output_path, row.names = FALSE)
  cat(sprintf("Generated dataset %d/%d: %s\n",
              dataset_id, args$n_datasets, output_path))
}
