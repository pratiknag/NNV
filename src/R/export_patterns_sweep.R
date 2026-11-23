#!/usr/bin/env Rscript

# Section 3.6: orderings and conditioning sets of HV (conditioning = "mra")
# and LR (conditioning = "firstm" with the same N) from GPvecchia on a
# side x side grid of [0, 1]^2, for a list of requested conditioning sizes.
# For each, writes <dir>/<method>_m<requested>.{ord,nn,meta,time}:
#   .ord   1-based ordering (int32)
#   .nn    revNNarray (n x width, column-major int32, NA as 0)
#   .meta  n, width
#   .time  elapsed seconds of vecchia_specify

suppressPackageStartupMessages(library(GPvecchia))

script_dir <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))))
source(file.path(script_dir, "utils.R"))

args <- parse_cli_args(list(side = 100, conditioning_sizes = "10,20", output_dir = "",
                            lr_sizes = "", model = "gaussian"))
side <- as.integer(args$side)
sizes <- as.integer(strsplit(args$conditioning_sizes, ",")[[1]])
# Optional LR sizes N, independent of HV (written as lr_N<N>); if not given,
# LR uses the N of each HV size (written as lr_m<m>)
lr_sizes <- if (args$lr_sizes == "") integer(0) else as.integer(strsplit(args$lr_sizes, ",")[[1]])
dir.create(args$output_dir, recursive = TRUE, showWarnings = FALSE)

grid <- seq(0, 1, length.out = side)
locs <- as.matrix(expand.grid(x1 = grid, x2 = grid))

write_pattern <- function(approx, elapsed, prefix) {
  rev_nn <- approx$U.prep$revNNarray
  stopifnot(all(rev_nn[, ncol(rev_nn)] == seq_len(nrow(rev_nn))))
  rev_nn[is.na(rev_nn)] <- 0L
  writeBin(as.integer(approx$ord), paste0(prefix, ".ord"), size = 4)
  writeBin(as.integer(rev_nn), paste0(prefix, ".nn"), size = 4)
  writeLines(as.character(c(nrow(rev_nn), ncol(rev_nn))), paste0(prefix, ".meta"))
  writeLines(as.character(elapsed), paste0(prefix, ".time"))
  ncol(rev_nn) - 1L
}

for (N in lr_sizes) {
  start <- proc.time()[["elapsed"]]
  lr <- GPvecchia::vecchia_specify(locs, m = N, ordering = "maxmin",
                                   conditioning = "firstm")
  write_pattern(lr, proc.time()[["elapsed"]] - start,
                file.path(args$output_dir, sprintf("lr_N%d", N)))
  cat(sprintf("side %d, LR N = %d\n", side, N))
}

for (m in sizes) {
  start <- proc.time()[["elapsed"]]
  hv <- GPvecchia::vecchia_specify(locs, m = m, ordering = "maxmin",
                                   conditioning = "mra")
  hv_time <- proc.time()[["elapsed"]] - start
  N <- write_pattern(hv, hv_time, file.path(args$output_dir, sprintf("hv_m%d", m)))
  start <- proc.time()[["elapsed"]]
  lr <- GPvecchia::vecchia_specify(locs, m = N, ordering = "maxmin",
                                   conditioning = "firstm")
  lr_time <- proc.time()[["elapsed"]] - start
  write_pattern(lr, lr_time, file.path(args$output_dir, sprintf("lr_m%d", m)))
  cat(sprintf("side %d, requested m = %d: N = %d (HV %.1f s, LR %.1f s)\n",
              side, m, N, hv_time, lr_time))
}
