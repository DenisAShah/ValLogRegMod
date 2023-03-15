#' Calculate optimism-corrected bootstrap internal validation for discrimation slope, Brier and Scaled Brier Score
#' @param db data to calculate the optimism-corrected bootstrap
#' @param B number of bootstrap sample (default 10)
#' @param outcome outcome variable (e.g., 0 = control, 1 = case) 
#' @param formula_model formula for the model (rms:lrm model)
#' @param formula_score formula to identify outcome in riskRegression::Score() function (e.g., y ~ 1)
#'
#' @return
#'
#' @author Daniele Giardiello
#'
#' @examples

# General packages (riskRegression version should be >= 2021.10.10)
pkgs <- c("rms", "riskRegression", "tidyverse")
vapply(pkgs, function(pkg) {
  if (!require(pkg, character.only = TRUE)) install.packages(pkg)
  require(pkg, character.only = TRUE, quietly = TRUE)
}, FUN.VALUE = logical(length = 1L))

bootstrap_cv_lrm <- function(db, 
                             B = 10,
                             outcome,
                             formula_model,
                             formula_score) {
  
  frm_model <- as.formula(formula_model)
  frm_score <- as.formula(formula_score)
  db$id <- 1:nrow(db)
  
  
  # Duplicate data
  db_ext <- db |> slice(rep(row_number(), B))
  db_ext$.rep <- with(db_ext, ave(seq_along(id), id, FUN = seq_along)) # add an index identifying the replications
  
  db_tbl <- db_ext |>
    group_by(.rep) |>
    nest() |>
    rename(
      orig_data = data,
      id_boot = .rep
    )
  
  # Create bootstrap samples
  sample_boot <- function(db, B) {
    db_boot <- matrix(NA, nrow = nrow(db) * B, ncol = ncol(db))
    sample_row <- list()
    for (j in 1:B) {
      sample_row[[j]] <- sample(nrow(db), size = nrow(db), replace = TRUE)
    }
    sample_row <- unlist(sample_row)
    db_boot <- db[sample_row, ]
    db_boot$id_boot <- sort(rep(1:B, nrow(db)))
    db_boot <- db_boot |>
      group_by(id_boot) |>
      nest() |>
      rename(boot_data = data)
    return(db_boot)
  }
  
  # Join original data and the bootstrap data in a nested tibble
  a <- sample_boot(db, B)
  b <- a |> left_join(db_tbl)
  
  # Create optimism-corrected performance measures
  b <- b |> dplyr::mutate(
    lrm_boot = purrr::map(
      boot_data,
      ~ rms::lrm(frm_model, data = ., x = T, y = T)
    ),
    
    lrm_apparent = purrr::map(
      orig_data,
      ~ rms::lrm(frm_model, data = ., x = T, y = T)
    ),
    
    # Discrimination slope
    dslope_app = purrr::map2_dbl(
      orig_data, lrm_apparent,
      function(.x, .y, new_data = .x, k = 3){
        pred <- predict(.y, 
                        newdata = as.data.frame(.x), 
                        type = "fitted.ind")
        
        mean_group <- tapply(pred, .x[outcome], mean)
        
        dslope <- round(abs(mean_group[1] - mean_group[2]), k)
        
        names(dslope) <- "Discrimination slope"
        
        return(dslope)
      }),
      
      
      dslope_orig = map2_dbl(
        orig_data, lrm_boot,
        
        function(.x, .y, new_data = .x, k = 3){
          
          pred <- predict(.y, 
                          newdata = as.data.frame(.x), 
                          type = "fitted.ind")
          
          mean_group <- tapply(pred, .x[outcome], mean)
          
          dslope <- round(abs(mean_group[1] - mean_group[2]), k)
          
          names(dslope) <- "Discrimination slope"
          
          return(dslope)
        }
        
      ),
      
      dslope_boot =
        purrr::map2_dbl(
          boot_data, lrm_boot,
          
          function(.x, .y, new_data = .x, k = 3){
            
            pred <- predict(.y, 
                            newdata = as.data.frame(.x), 
                            type = "fitted.ind")
            
            mean_group <- tapply(pred, .x[outcome], mean)
            
            dslope <- round(abs(mean_group[1] - mean_group[2]), k)
            
            names(dslope) <- "Discrimination slope"
            return(dslope)
          }),
          
      dslope_diff = purrr::map2_dbl(
          dslope_boot, dslope_orig,
          function(a, b) {
              a - b
            }
          ),
    
    # Brier score
    
    Score_app = purrr::map2(
      orig_data, lrm_apparent,
      ~ riskrRegression::Score(list("Logistic" = .y),
                               formula = frm_score,
                               data = .x, 
                               metrics = "brier",
                               summary = "ipa"
      )$Brier$score
    ),
    
    Brier_app = purrr::map_dbl(Score_app, ~ .x$Brier[[2]]),
    IPA_app =purrr:: map_dbl(Score_app, ~ .x$IPA[[2]]),
    
    Score_orig = purrr::map2(
      orig_data, lrm_boot,
      ~ riskRegression::Score(list("Logistic" = .y),
                              formula = frm_score,
                              data = .x, 
                              metrics = "brier",
                              summary = "ipa"
      )$Brier$score
    ),
    
    Brier_orig = purrr::map_dbl(Score_orig, ~ .x$Brier[[2]]),
    IPA_orig = purrr::map_dbl(Score_orig, ~ .x$IPA[[2]]),
    
    Score_boot = purrr::map2(
      boot_data, lrm_boot,
      ~ riskRegression::Score(list("Logistic" = .y),
                              formula = frm_score,
                              data = .x,
                              metrics = "brier",
                              summary = "ipa"
      )$Brier$score
    ),
    
    Brier_boot = purrr::map_dbl(Score_boot, ~ .x$Brier[[2]]),
    IPA_boot = purrr::map_dbl(Score_boot, ~ .x$IPA[[2]]),
    Brier_diff = purrr::map2_dbl(
      Brier_boot, Brier_orig,
      function(a, b) {
        a - b
      }
    ),
    
    IPA_diff = purrr::map2_dbl(
      IPA_boot, IPA_orig,
      function(a, b) {
        a - b
      }
    )
  )
  
  # Generate output
  Brier_corrected <- b$Brier_app[1] - mean(b$Brier_diff)
  IPA_corrected <- b$IPA_app[1] - mean(b$IPA_diff)
  dslope_corrected <- b$dslope_app[1] - mean(b$dslope_diff)
  
  res <- c("Discrimination slope corrected" = dslope_corrected,
           "Brier corrected" = Brier_corrected, 
           "IPA corrected" = IPA_corrected 
  )
  return(res)
}

