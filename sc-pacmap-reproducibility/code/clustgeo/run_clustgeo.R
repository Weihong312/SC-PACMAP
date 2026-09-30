# Project-authored I/O bridge only; clustering is the unmodified CRAN package.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2L) stop("Expected library and work directory (or --check)")
lib <- normalizePath(args[1L], mustWork = TRUE)
.libPaths(c(lib, .Library), include.site = FALSE)
if (!requireNamespace("ClustGeo", quietly = TRUE)) {
  stop("ClustGeo is unavailable. Run: python code/clustgeo_benchmark.py --setup")
}
if (as.character(packageVersion("ClustGeo")) != "2.1") {
  stop("This benchmark requires the original CRAN ClustGeo 2.1 package")
}
metadata <- c(paste("R", as.character(getRversion()), sep = "="),
              paste("ClustGeo", as.character(packageVersion("ClustGeo")), sep = "="),
              paste("library", find.package("ClustGeo"), sep = "="),
              "backend=ClustGeo::hclustgeo",
              "scale=TRUE", "weights=equal", "distance=euclidean")
if (args[2L] == "--check") {
  cat(paste(metadata, collapse = "\n"), "\n", sep = "")
  quit(status = 0)
}
work <- args[2L]
k <- as.integer(args[3L])
alphas <- as.numeric(args[-(1:3)])
y <- as.matrix(read.csv(file.path(work, "embedding.csv"), header = FALSE))
coords <- as.matrix(read.csv(file.path(work, "coordinates.csv"), header = FALSE))
stopifnot(ncol(y) == 2L, nrow(y) == nrow(coords),
          ncol(coords) %in% c(2L, 3L), all(is.finite(y)), all(is.finite(coords)),
          k >= 2L, k <= nrow(y), length(alphas) > 0L,
          all(is.finite(alphas)), all(alphas >= 0 & alphas <= 1))
D0 <- stats::dist(y, method = "euclidean")
D1 <- stats::dist(coords, method = "euclidean")
if (max(D0) <= 0 || max(D1) <= 0) stop("ClustGeo needs positive maximum distances")
partitions <- vapply(alphas, function(alpha) {
  tree <- ClustGeo::hclustgeo(D0, D1, alpha = alpha, scale = TRUE, wt = NULL)
  as.integer(stats::cutree(tree, k = k)) - 1L
}, integer(nrow(y)))
write.table(partitions, file.path(work, "partitions.csv"), sep = ",",
            row.names = FALSE, col.names = FALSE)
writeLines(metadata, file.path(work, "backend.txt"))
