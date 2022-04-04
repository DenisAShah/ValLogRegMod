#' Calculate discrimination slope when individual data of development set is available
#'
#' @param fit logistic regression fit using rms::lrm()
#' @param y outcome variable (0 = control, 1 = case)
#' @param new_data data to calculate the discrimination slope 
#'
#' @return
#'
#' @author Daniele Giardiello
#'
#' @examples

discr_slope <- function(fit, y, new_data, k = 3) {
  pred <- predict(fit, 
                  newdata = new_data, 
                  type = "fitted.ind")
  mean_group <- tapply(pred, y, mean)
  dslope <- round(abs(mean_group[1] - mean_group[2]), k)
  names(dslope) <- "Discrimination slope"
  return(dslope)
}