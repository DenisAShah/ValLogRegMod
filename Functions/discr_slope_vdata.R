# Function to calculate the discrimination slope
# when  individual data of development set is unavailable

#'
#' @param pred estimated predicted probability
#' @param y outcome variable (0 = control, 1 = case)
#' @param k number of digits (default 3)
#'
#' @return
#'
#' @author Daniele Giardiello
#'
#' @examples
#' 
discr_slope <- function(pred, y, k = 3) {
  mean_group <- tapply(pred, y, mean)
  dslope <- abs(mean_group[1] - mean_group[2])
  return(dslope)
} 