# Install/restore the original CRAN package in a project-local R library.
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2L)
lib <- args[1L]
lockfile <- args[2L]
dir.create(lib, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(normalizePath(lib), .Library), include.site = FALSE)
options(repos = c(CRAN = "https://cloud.r-project.org"), timeout = 300)

if (!requireNamespace("renv", quietly = TRUE)) {
  install.packages("renv", lib = lib)
}
if (file.exists(lockfile)) {
  renv::restore(project = dirname(lockfile), library = lib,
                lockfile = lockfile, prompt = FALSE)
} else {
  # ClustGeo 2.1 is the official CRAN implementation, including dependencies.
  if (!requireNamespace("ClustGeo", quietly = TRUE) ||
      as.character(packageVersion("ClustGeo")) != "2.1") {
    renv::install("ClustGeo@2.1", project = dirname(lockfile), library = lib)
  }
  # Include recommended/transitive packages in the isolated library too, so
  # the lockfile does not rely on packages found in a user's R installation.
  repeat {
    installed <- installed.packages(lib.loc = lib)
    dependencies <- unique(unlist(tools::package_dependencies(
      installed[, "Package"], db = installed,
      which = c("Depends", "Imports", "LinkingTo"))))
    base <- rownames(installed.packages(lib.loc = .Library, priority = "base"))
    missing <- setdiff(dependencies, c(installed[, "Package"], base, "R"))
    if (!length(missing)) break
    # R-distributed recommended packages must match this R runtime. Copy the
    # official installed versions instead of requesting newer incompatible ones.
    recommended <- rownames(installed.packages(lib.loc = .Library, priority = "recommended"))
    bundled <- intersect(missing, recommended)
    for (package in bundled) {
      if (!file.copy(file.path(.Library, package), lib, recursive = TRUE)) {
        stop("Unable to copy R-recommended package: ", package)
      }
    }
    remaining <- setdiff(missing, bundled)
    if (length(remaining)) {
      renv::install(remaining, project = dirname(lockfile), library = lib)
    }
  }
  renv::snapshot(project = dirname(lockfile), library = lib,
                 lockfile = lockfile, type = "all", prompt = FALSE)
}
stopifnot(as.character(packageVersion("ClustGeo", lib.loc = lib)) == "2.1")
cat("Original ClustGeo", as.character(packageVersion("ClustGeo")),
    "ready in", normalizePath(lib), "\n")
