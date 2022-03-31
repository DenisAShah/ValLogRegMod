# Load libraries and data -------------------------------------------------


# General packages (riskRegression version should be >= 2021.10.10)
pkgs <- c("rms", "riskRegression")
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

fit_lrm <- glm(tum_res ~ 
                  ter_pos + preafp + prehcg + 
                  sqpost + reduc10,
               data = rdata,
               family = binomial,
               x = T,
               y = T
                )

# Discrimination ---------------------

## C-index
# Apparent validation
val_rdata <- rcorr.cens(predict(fit_lrm), 
                        S = rdata$tum_res)

# External validation
val_vdata <- rcorr.cens(
  predict(fit_lrm, newdata = vdata), 
  S = vdata$tum_res)

res_discr <- c(
  "Apparent" = val_rdata[["C Index"]],
  "External validation" = val_vdata[["C Index"]]
)

## Calibration ---------------------------

# Calibration-in-the-large
lp <- predict(fit_lrm, newdata = vdata)
vdata$y <- as.numeric(vdata$tum_res) - 1 # convert outcome to numeric
cal_intercept <- glm(y  ~ offset(lp), 
                     family = binomial,
                     data = vdata)
intercept_CI <- confint(cal_intercept) # confidence intervals

# Calibration slope
cal_slope <- glm(y  ~ lp,
                 family = binomial,
                 data = vdata)
slope_CI <- confint(cal_slope) # Confidence interval

res_cal <- matrix(
  c(
    cal_intercept$coefficients,
    intercept_CI[1],
    intercept_CI[2],
    
    cal_slope$coefficients[2],
    slope_CI[1],
    slope_CI[2]
  ),
  ncol = 3,
  nrow = 2,
  byrow = T,
  dimnames = list(c("Intercept","Slope"),
                  c("Estimate", "2.5 %", "97.5 %"))
)

res_cal

## Calibration plot
vdata$pred <- predict(fit_lrm,
                      newdata = vdata,
                      type = "response")

# ....


# Overall performances ---------------------

# Brier score and scaled Brier 

# Development data
score_rdata <- Score(
  list("Development set" = fit_lrm),
  formula = tum_res ~ 1,
  data = rdata,
  conf.int = TRUE,
  metrics = c("auc", "brier"),
  summary = c("ipa"),
  plots = "calibration"
)

score_rdata$Brier$score

# Validation data
score_vdata <- Score(
  list("Validation set" = fit_lrm),
  formula = tum_res ~ 1,
  data = vdata,
  conf.int = TRUE,
  metrics = c("auc", "brier"),
  summary = c("ipa"),
  plots = "calibration"
)

score_vdata$Brier$score

## Clinical utility --------------------

# 1. Set grid of thresholds
thresholds <- seq(0, 1.0, by = 0.01)

# 2. Calculate observed risk for all patients exceeding threshold (i.e. treat-all)
survfit_all <- summary(
  survfit(Surv(ryear, rfs) ~ 1, data = gbsg5), 
  times = horizon
)
f_all <- 1 - survfit_all$surv

# 3. Calculate Net Benefit across all thresholds
list_nb <- lapply(thresholds, function(ps) {
  
  # Treat all
  NB_all <- f_all - (1 - f_all) * (ps / (1 - ps))
  
  # Based on threshold
  p_exceed <- mean(vdata$pred > ps)
  survfit_among_exceed <- try(
    summary(
      survfit(Surv(ryear, rfs) ~ 1, data = gbsg5[gbsg5$pred > ps, ]), 
      times = horizon
    ), silent = TRUE
  )
  
  # If a) no more observations above threshold, or b) among subset exceeding..
  # ..no indiv has event time >= horizon, then NB = 0
  if (class(survfit_among_exceed) == "try-error") {
    NB <- 0
  } else {
    f_given_exceed <- 1 - survfit_among_exceed$surv
    TP <- f_given_exceed * p_exceed
    FP <- (1 - f_given_exceed) * p_exceed
    NB <- TP - FP * (ps / (1 - ps))
  }
  
  # Return together
  df_res <- data.frame("threshold" = ps, "NB" = NB, "treat_all" = NB_all)
  return(df_res)
})

# Combine into data frame
df_nb <- do.call(rbind.data.frame, list_nb)

# read off at 23% threshold
df_nb[df_nb$threshold == 0.23,]

# Make basic decision curve plot
par(
  xaxs = "i", 
  yaxs = "i", 
  las = 1, 
  mar = c(6.1, 5.8, 4.1, 2.1), 
  mgp = c(4.25, 1, 0)
)
plot(
  df_nb$threshold, 
  df_nb$NB,
  type = "l", 
  lwd = 2,
  ylim = c(-0.1, 0.6),
  xlim = c(0, 1), 
  xlab = "",
  ylab = "Net Benefit",
  bty = "n", 
)
lines(df_nb$threshold, df_nb$treat_all, type = "l", col = "darkgray", lwd = 2)
abline(h = 0, lty = 2, lwd = 2)
legend(
  "topright", 
  c("Treat all", "Treat none", "Prediction model"),
  lwd = c(2, 2, 2), 
  lty = c(1, 2, 1), 
  col = c("darkgray", "black", "black"), 
  bty = "n"
)
mtext("Threshold probability", 1, line = 2)
title("Validation data")
