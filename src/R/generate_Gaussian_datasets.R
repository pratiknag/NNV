#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(GPvecchia)
  library(fields)
})
script_dir <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))))
source(file.path(script_dir, "utils.R"))

parse_args <- function() {
  defaults <- parse_cli_args(list(n = 1024, n_datasets = 100,
                   model = "gaussian", output_dir = "",
                   range = 0.15, variance = 1.0, smoothness = 0.5,
                   noise_variance = 0.2, gamma_shape = 2, seed = 1988))
  if (defaults$output_dir == "") {
    defaults$output_dir <- sprintf("simulation_%s", defaults$model)
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
covariance <- GPvecchia::MaternFun(fields::rdist(locs), covparms)
chol_covariance <- chol(covariance)

for (dataset_id in seq_len(args$n_datasets)) {
  latent <- as.numeric(t(chol_covariance) %*% rnorm(args$n))
  # The Gaussian noise is always drawn so that the main RNG stream, and hence
  # every latent field, is identical across data models.
  noise <- rnorm(args$n, sd = sqrt(args$noise_variance))
  if (args$model == "gaussian") {
    observations <- latent + noise
  } else {
    # Non-Gaussian observations come from a separate per-dataset seed; the
    # main stream is restored afterwards.
    main_state <- .Random.seed
    set.seed(args$seed + 100000 * model_id + dataset_id)
    observations <- simulate_observations(args$model, latent, noise,
                                          args$gamma_shape)
    assign(".Random.seed", main_state, envir = .GlobalEnv)
  }
  output_path <- file.path(
    args$output_dir,
    sprintf("%s_seed%d_domain1_dim2_n%d_smooth%.1f_range%.2f.csv",
            args$model, dataset_id, args$n, args$smoothness, args$range)
  )
  write.csv(data.frame(x1 = locs[, 1], x2 = locs[, 2],
                       y = latent, z = observations),
            output_path, row.names = FALSE)
  cat(sprintf("Generated dataset %d/%d: %s\n",
              dataset_id, args$n_datasets, output_path))
}
