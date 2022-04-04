#' Calculate confidence interval for the c-statistic
#'
#' @param x object rms:rcorr.cens. See details (help(rms::rcorr.cens))
#' @param k number of digits for the output (default 3)
#' @param alpha type I error level (default 0.05)
#'
#' @return
#'
#' @author Daniele Giardiello
#'
#' @examples

c_stat_ci <- function(x, k = 3, alpha = .05) {
  se <- x["S.D."]/2
  Lower95 <- x["C Index"] - qnorm(1 - alpha / 2) * se
  Upper95 <- x["C Index"] + qnorm(1 - alpha / 2) * se
  round(cbind(
    "C-statistic" = x["C Index"],
    Lower95, 
    Upper95), k)
}