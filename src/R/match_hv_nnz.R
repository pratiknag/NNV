#!/usr/bin/env Rscript

# Section 3.6: requested GPvecchia conditioning sizes m for HV whose average
# number of nonzeros per row of the posterior covariance factor L (= mean
# |C_i| + 1, since L has the HV pattern) is closest to given targets (those of
# NN-triple). Prints a CSV with columns target, m, N, nnz_per_row.

suppressPackageStartupMessages(library(GPvecchia))

script_dir <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))))
source(file.path(script_dir, "utils.R"))

args <- parse_cli_args(list(side = 100, targets = "20,40", m_min = 10, m_max = 160,
                            output = "", model = "gaussian"))
side <- as.integer(args$side)
targets <- as.numeric(strsplit(args$targets, ",")[[1]])
grid <- seq(0, 1, length.out = side)
locs <- as.matrix(expand.grid(x1 = grid, x2 = grid))

scan <- do.call(rbind, lapply(seq(args$m_min, args$m_max), function(m) {
  hv <- suppressMessages(vecchia_specify(locs, m = m, ordering = "maxmin",
                                         conditioning = "mra", verbose = FALSE))
  rev_nn <- hv$U.prep$revNNarray
  data.frame(m = m, N = ncol(rev_nn) - 1L, nnz_per_row = mean(rowSums(!is.na(rev_nn))))
}))
best <- do.call(rbind, lapply(targets, function(t) {
  i <- which.min(abs(scan$nnz_per_row - t))
  data.frame(target = t, scan[i, ])
}))
print(best)
write.csv(best, args$output, row.names = FALSE)
