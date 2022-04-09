# Load libraries and data -------------------------------------------------


# General packages (riskRegression version should be >= 2021.10.10)
pkgs <- c("rms", "riskRegression")
vapply(pkgs, function(pkg) {
  if (!require(pkg, character.only = TRUE)) install.packages(pkg)
  require(pkg, character.only = TRUE, quietly = TRUE)
}, FUN.VALUE = logical(length = 1L))

# Load validation dataset
vdata <- readRDS("Data/vdata.rds")

# Set seed (for bootstrapping)
set.seed(2022)

# Design matrix of predictors
des_matr <- as.data.frame(model.matrix(~ ter_pos + preafp +
                                         prehcg + sqpost + 
                                         reduc10,
                                       data = vdata))

# Coefficients of the developed model
coeff <- c(-0.302, 0.995, 0.859, 0.554, 0.074, -0.264)
# Linear predictor and the estimated predicted probability
# in the validation data
vdata$lp <- as.vector(as.matrix(des_matr) %*% cbind(coeff))
vdata$pred <- exp(vdata$lp) / (1 + exp(vdata$lp))


# Discrimination ---------------------

## C-index

# External validation
val_vdata <- rcorr.cens(
  vdata$lp, 
  S = vdata$tum_res)

res_discr <- 
  cbind("C-statistic" = val_vdata[["C Index"]],
        "2.5 %" = val_vdata[["C Index"]] - qnorm(.975)*(val_vdata[["S.D."]]/2),
        "97.5 %" = val_vdata[["C Index"]] + qnorm(.975)*(val_vdata[["S.D."]]/2)
        )

res_discr

## Calibration ---------------------------

# Calibration-in-the-large
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
# First, prepare histogram of estimated risks for x-axis
spike_bounds <- c(0, 0.20)
bin_breaks <- seq(0, 1, length.out = 100 + 1)
freqs <- table(cut(vdata$pred, breaks = bin_breaks))
bins <- bin_breaks[-1]
freqs_valid <- freqs[freqs > 0]
freqs_rescaled <- spike_bounds[1] + (spike_bounds[2] - spike_bounds[1]) * 
  (freqs_valid - min(freqs_valid)) / (max(freqs_valid) - min(freqs_valid))

# Calibration based on a secondary logistic regression
fit_cal <- glm(y ~ pred,
               family = binomial,
               data = vdata) # rms::rcs() for more flexible modelling possible

cal_obs <- predict(fit_cal, 
                   type = "response",
                   se.fit =  TRUE)
alpha <- .05
dt_cal <- cbind.data.frame("obs" = cal_obs$fit,
                           
                           "lower" = 
                             cal_obs$fit - 
                             qnorm(1 - alpha / 2)*cal_obs$se.fit,
                           
                           "upper" = cal_obs$fit + 
                             qnorm(1 - alpha / 2)*cal_obs$se.fit,
                           
                           "pred" = vdata$pred)
dt_cal <- dt_cal[order(dt_cal$pred),]

par(xaxs = "i", yaxs = "i", las = 1)
plot(lowess(vdata$pred, vdata$y, iter = 0),
     type = "l",
     xlim = c(0, 1),
     ylim = c(-.1, 1),
     xlab = "Predicted probability",
     ylab = "Actual probability",
     bty = "n",
     lwd = 2,
     main = "Calibration plot")
lines(dt_cal$pred, dt_cal$obs, lwd = 2, lty = 2)
lines(dt_cal$pred, dt_cal$lower, lwd = 2, lty = 3)
lines(dt_cal$pred, dt_cal$upper, lwd = 2, lty = 3)
abline(a = 0, b = 1, col = "gray")
segments(
  x0 = bins[freqs > 0], 
  y0 = spike_bounds[1], 
  x1 = bins[freqs > 0], 
  y1 = freqs_rescaled
)
legend(x = .02, y = 1.2,
       c("Ideal", "Lowess", "Logistic", "95% confidence interval"),
       lwd = c(1, 2, 2, 2),
       lty = c(1, 1, 2, 3),
       col = c("gray", "black", "black", "black"),
       bty = "n",
       seg.len = .5,
       cex = .50,
       x.intersp = .5,
       y.intersp = .5)

# Calibration measures ICI, E50, E90 based on secondary logistic regression
res_calmeas <-
  cbind(
    "ICI" = mean(abs(dt_cal$obs - dt_cal$pred)),
    "E50" = median(abs(dt_cal$obs - dt_cal$pred)),
    "E90" = unname(quantile(abs(dt_cal$obs - dt_cal$pred), probs = .90))
  )

res_calmeas

# NOTE: Calibration measures as ICI, E50, and E90 might be also estimated
# using lowess estimation 

# Overall performances ---------------------

# Brier score and scaled Brier 

# Validation data
score_vdata <- Score(
  list(vdata$pred),
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

f_all <- mean(vdata$y)

# 3. Calculate Net Benefit across all thresholds
list_nb <- lapply(thresholds, function(ps) {
  
  # Treat all
  NB_all <- f_all - (1 - f_all) * (ps / (1 - ps))
  
  # Based on threshold
  tdata <- vdata[vdata$pred > ps,] 
  TP <- sum(tdata$y)
  FP <- sum(tdata$y == 0)
  NB <- (TP / nrow(vdata))- (FP / nrow(vdata)) * (ps / (1 - ps))
  
  # Return together
  df_res <- data.frame("threshold" = ps, "NB" = NB, "treat_all" = NB_all)
  return(df_res)
})

# Combine into data frame
df_nb <- do.call(rbind.data.frame, list_nb)

# Remove NB < 0
df_nb <- df_nb[df_nb$NB >= 0, ]

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
  ylim = c(-0.1, 0.8),
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


