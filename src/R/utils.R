# Shared helpers for the spatial and the filtering simulations.

data_models <- c("gaussian", "logistic", "poisson", "gamma")

parse_cli_args <- function(defaults) {
  # Parse --name=value arguments; each value takes the type of its default.
  for (arg in commandArgs(trailingOnly = TRUE)) {
    pair <- strsplit(sub("^--", "", arg), "=", fixed = TRUE)[[1]]
    if (length(pair) != 2 || !pair[1] %in% names(defaults)) {
      stop("Arguments must use --name=value and match the documented options.")
    }
    defaults[[pair[1]]] <- if (is.character(defaults[[pair[1]]])) pair[2] else as.numeric(pair[2])
  }
  if (!defaults$model %in% data_models) {
    stop(sprintf("model must be one of: %s", paste(data_models, collapse = ", ")))
  }
  defaults
}

# gamma uses GPvecchia's log-rate parameterization, matching the data generator.
gpvecchia_likelihood <- function(model) {
  switch(model,
    gaussian = "gaussian", logistic = "logistic",
    poisson = "poisson", gamma = "gamma_alt"
  )
}

simulate_observations <- function(model, latent, noise, gamma_shape) {
  # The Gaussian noise is always drawn by the caller so that the main RNG
  # stream is identical across data models; the other models draw from
  # whatever stream the caller has set.
  switch(model,
    gaussian = latent + noise,
    logistic = rbinom(length(latent), 1, plogis(latent)),
    poisson = rpois(length(latent), exp(latent)),
    gamma = rgamma(length(latent), shape = gamma_shape, rate = exp(latent))
  )
}

dposterior <- function(y, pred) {
  # Gaussian posterior log-density using the posterior precision.
  y <- as.numeric(y)
  mu <- as.numeric(pred$mean)
  W <- pred$W
  V <- pred$V

  if (length(y) != length(mu) || nrow(W) != length(y)) {
    stop("y, pred$mean, and pred$W have incompatible dimensions.")
  }

  residual <- y - mu
  W_residual <- as.numeric(W %*% residual)
  quad_term <- -0.5 * sum(residual * W_residual)

  diagonal_V <- as.numeric(diag(V))
  if (any(!is.finite(diagonal_V)) || any(diagonal_V <= 0)) {
    stop("pred$V must have a positive finite diagonal.")
  }
  det_term <- sum(log(diagonal_V))
  pi_term <- -0.5 * length(y) * log(2 * pi)

  quad_term + det_term + pi_term
}

advection_diffusion_matrix <- function(n, advection, diffusion) {
  # Centered finite-difference evolution matrix on a sqrt(n) x sqrt(n) grid,
  # as in evolAdvDiff of Jurek and Katzfuss (2022).
  nx <- as.integer(round(sqrt(n)))
  dx <- 1 / nx
  D <- diffusion / dx^2
  c1 <- 1 + 4 * D - 2 * advection / dx
  c2 <- -D + advection / dx
  c3 <- -D
  below <- rep(c2, n - 1)
  above <- rep(c3, n - 1)
  # No coupling across the edges of the grid rows
  below[nx * seq_len(nx - 1)] <- 0
  above[nx * seq_len(nx - 1)] <- 0
  Matrix::bandSparse(n, k = c(-nx, -1, 0, 1, nx),
                     diagonals = list(rep(c2, n - nx), below, rep(c1, n),
                                      above, rep(c3, n - nx)))
}
