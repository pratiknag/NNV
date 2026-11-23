# Schematic of the two sparsity patterns under the maximin ordering (Section 2.2).
# Writes figures/patterns.pdf. Run from the repository root: Rscript src/R/patterns.R

n_side <- 16             # 16 x 16 grid on [0,1]^2, randomly perturbed
jitter <- 0.3            # perturbation of each coordinate, as a fraction of the grid spacing
rho <- 2                 # radius parameter
seed <- 9                # design shown in the paper (its location i has no other location on the circle)
target_i <- c(0.60, 0.40)  # a fine-scale location i near this point

blue <- "#2a78d6"        # precision factor (nearest-neighbor pattern)
orange <- "#eb6834"      # covariance factor (support pattern)
ink <- "#0b0b0b"
gray_early <- "#6f6e69"  # locations ordered before location i
gray_late <- "#c9c7c0"   # locations ordered after location i

# perturbed grid, so that the locations are irregular and the ordering has no ties
set.seed(seed)
h <- 1 / (n_side - 1)
g <- seq(0, 1, length.out = n_side)
locs <- as.matrix(expand.grid(g, g))
locs <- locs + matrix(runif(length(locs), -jitter * h, jitter * h), ncol = 2)
locs <- pmin(pmax(locs, 0), 1)
n <- nrow(locs)

# exact maximin ordering, starting with the location closest to the center
dist_to <- function(p) sqrt((locs[, 1] - p[1])^2 + (locs[, 2] - p[2])^2)
ord <- integer(n)
ell <- numeric(n)
ord[1] <- which.min(dist_to(c(0.5, 0.5)))
ell[1] <- Inf
mind <- dist_to(locs[ord[1], ])
mind[ord[1]] <- -1
for (k in 2:n) {
  nxt <- which.max(mind)
  ord[k] <- nxt
  ell[k] <- mind[nxt]
  mind <- pmin(mind, dist_to(locs[nxt, ]))
  mind[ord[1:k]] <- -1
}
s <- locs[ord, ]                     # locations in maximin order
D <- as.matrix(dist(s))

# patterns: S_rho (upper; column i = K_i) and the support pattern S^L_rho (lower; row i = J_i)
upper <- outer(seq_len(n), seq_len(n), "<=")          # (j, i) with j <= i
S_nn <- upper & (D < rho * matrix(ell, n, n, byrow = TRUE))
S_cov <- t(upper) & (D < rho * matrix(ell, n, n, byrow = TRUE))

# location i: fine scale (second half of the ordering), six to nine earlier locations in its
# disk and also one or two later ones (which are not in column i), and no location (almost)
# on its circle
cand <- which(seq_len(n) > n / 2)
n_earlier <- sapply(cand, function(k) sum(D[k, seq_len(k - 1)] < rho * ell[k]))
n_later <- sapply(cand, function(k) sum(D[k, -(1:k)] < rho * ell[k]))
cand <- cand[n_earlier >= 6 & n_earlier <= 9 & n_later >= 1 & n_later <= 2]
cand <- cand[sapply(cand, function(k) min(abs(D[k, -k] - rho * ell[k])) > 0.05 * h)]
stopifnot(length(cand) > 0)
i <- cand[which.min((s[cand, 1] - target_i[1])^2 + (s[cand, 2] - target_i[2])^2)]
K_i <- setdiff(which(S_nn[, i]), i)   # column i of S_rho: earlier locations within rho * ell_i
J_i <- setdiff(which(S_cov[i, ]), i)  # row i of S^L_rho: earlier j with ||s_i - s_j|| < rho * ell_j

# triangle size in panel (b) increases with the scale ell_j (log scale; ell_1 = Inf is capped)
l_fin <- ell[is.finite(ell)]
size_of <- function(l) {
  l <- pmin(l, max(l_fin))
  0.65 + 1.1 * (log(l) - log(min(l_fin))) / (log(max(l_fin)) - log(min(l_fin)))
}

cat(sprintf("seed = %d, n = %d, rho = %g, grid spacing h = %.3f\n", seed, n, rho, h))
cat(sprintf("i = %d, ell_i = %.3f, |K_i| = %d, |J_i| = %d, earlier locations: %d\n",
            i, ell[i], length(K_i) + 1, length(J_i) + 1, i - 1))
cat(sprintf("row i of S^L: distances %.2f to %.2f, ell_j from %.3f to %s\n", min(D[i, J_i]), max(D[i, J_i]),
            min(ell[J_i]), format(max(ell[J_i]))))
cat(sprintf("nnz S_rho = %d, nnz S^L_rho = %d, max row of S^L_rho = %d\n",
            sum(S_nn), sum(S_cov), max(rowSums(S_cov))))

circle <- function(center, r, ...) {
  a <- seq(0, 2 * pi, length.out = 200)
  lines(center[1] + r * cos(a), center[2] + r * sin(a), ...)
}
base_panel <- function(title) {  # all locations, shaded by whether they precede location i
  plot(NA, xlim = c(-0.03, 1.03), ylim = c(-0.03, 1.03), xaxs = "i", yaxs = "i", axes = FALSE,
       xlab = "", ylab = "")
  box(col = "#b5b4ae", lwd = 0.6)
  points(s[-(1:i), , drop = FALSE], pch = 16, cex = 0.7, col = gray_late)
  points(s[1:(i - 1), , drop = FALSE], pch = 16, cex = 0.7, col = gray_early)
  mtext(title, side = 3, line = 0.3, cex = 0.85, col = ink)
}
star <- function() points(s[i, 1], s[i, 2], pch = 8, col = ink, cex = 0.95, lwd = 1.3)

dir.create("figures", showWarnings = FALSE)
cairo_pdf("figures/patterns.pdf", width = 6.5, height = 2.66, pointsize = 10, family = "Arial")  # embeds fonts
# square plot regions (pty = "s") of equal size: panel (c) gets a slightly wider cell for its
# larger side margins (0.8 more lines), and all panels share the top and bottom margins
layout(matrix(c(1, 2, 3, 4, 4, 4), nrow = 2, byrow = TRUE), widths = c(1, 1, 1.062),
       heights = c(1, 0.09))
par(mar = c(1.5, 0.6, 1.6, 0.6), oma = c(0, 0, 0, 0), pty = "s")

# (a) column i of the precision factor: earlier locations within rho * ell_i of s_i
base_panel(expression(paste("(a) Column ", italic(i), " of precision factor")))
circle(s[i, ], rho * ell[i], col = blue, lwd = 1.2)
points(s[K_i, , drop = FALSE], pch = 21, bg = blue, col = "white", cex = 1.1, lwd = 0.6)
star()

# (b) row i of the covariance factor: earlier locations s_j with ||s_i - s_j|| < rho * ell_j,
#     largest triangles drawn first; the dashed circle is the circle of panel (a)
base_panel(expression(paste("(b) Row ", italic(i), " of covariance factor")))
circle(s[i, ], rho * ell[i], col = blue, lwd = 0.9, lty = 2)
jj <- J_i[order(-size_of(ell[J_i]))]
points(s[jj, , drop = FALSE], pch = 24, bg = orange, col = "white", cex = size_of(ell[jj]), lwd = 0.6)
star()

# (c) both patterns as one matrix: upper triangle S_rho, lower triangle S^L_rho
M <- matrix(0, n, n)
M[S_nn] <- 1
M[S_cov] <- 2
diag(M) <- 3
par(mar = c(1.5, 1.6, 1.6, 0.4))
image(seq_len(n), seq_len(n), t(M[n:1, ]), col = c("white", blue, orange, ink),
      breaks = c(-0.5, 0.5, 1.5, 2.5, 3.5), axes = FALSE, xlab = "", ylab = "", useRaster = TRUE)
rect(0.5, 0.5, n + 0.5, n + 0.5, border = "#b5b4ae", lwd = 0.6)  # frame the matrix itself
for (side in 1:2) {  # ticks at column i (bottom) and row i (left; row 1 is at the top), on the frame
  axis(side, at = if (side == 1) i else n - i + 1, labels = expression(italic(i)), pos = 0.5,
       lwd = 0, lwd.ticks = 0.8, tcl = -0.25, mgp = c(0, 0.45, 0), cex.axis = 1, col.ticks = ink,
       col.axis = ink, las = 1)
}
mtext("(c) Both patterns, all locations", side = 3, line = 0.3, cex = 0.85, col = ink)

# shared legend, placed by hand so that the entries sit close together
par(mar = c(0, 0, 0, 0), pty = "m")
plot(NA, xlim = c(0, 1), ylim = c(0, 1), axes = FALSE, xlab = "", ylab = "")
entries <- list(
  list(pch = 8, bg = NA, col = ink, cex = 0.9, lab = expression(paste("location ", italic(i)))),
  list(pch = 16, bg = NA, col = gray_early, cex = 0.75, lab = "earlier locations"),
  list(pch = 16, bg = NA, col = gray_late, cex = 0.75, lab = "later locations"),
  list(pch = 21, bg = blue, col = "white", cex = 1.1,
       lab = expression(paste(bold(S)[rho], " (precision factor)"))),
  list(pch = 24, bg = orange, col = "white", cex = 1.0,
       lab = expression(paste(bold(S)[rho]^L, " (covariance factor)")))
)
gap <- 0.03
pad <- 0.013
widths <- sapply(entries, function(e) pad + strwidth(e$lab, cex = 0.85))
total <- sum(widths) + gap * (length(entries) - 1)
cat(sprintf("legend width: %.2f of the figure width\n", total))
x <- (1 - total) / 2
for (k in seq_along(entries)) {
  e <- entries[[k]]
  points(x, 0.5, pch = e$pch, bg = e$bg, col = e$col, cex = e$cex, lwd = if (e$pch == 8) 1.2 else 0.6)
  text(x + pad, 0.5, e$lab, adj = c(0, 0.5), cex = 0.85, col = ink)
  x <- x + widths[k] + gap
}
invisible(dev.off())
