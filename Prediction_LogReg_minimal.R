# Load libraries and data -------------------------------------------------


# General packages (riskRegression version should be >= 2021.10.10)
pkgs <- c("rms", "splines", "riskRegression")
vapply(pkgs, function(pkg) {
  if (!require(pkg, character.only = TRUE)) install.packages(pkg)
  require(pkg, character.only = TRUE, quietly = TRUE)
}, FUN.VALUE = logical(length = 1L))

# Load datasets
rdata <- readRDS("Data/rdata.rds")
vdata <- readRDS("Data/vdata.rds")

# Set seed (for bootstrapping)
set.seed(2022)

# Fit logistic model ---------------------------------------

