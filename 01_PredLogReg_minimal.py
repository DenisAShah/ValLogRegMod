# Install libraries
# using python please use, for example: 
# pip install pandas 

# In R / R studio
# pkgs <- c("reticulate")
# vapply(pkgs, function(pkg) {
#   if (!require(pkg, character.only = TRUE)) install.packages(pkg)
#   require(pkg, character.only = TRUE, quietly = TRUE)
# }, FUN.VALUE = logical(length = 1L))

# py_install("pandas","numpy", "scipy", "statsmodels", 
#            "matplotlib", "sklearn")


# Load libraries and data
import pandas as pd
import numpy as np
import scipy as sp
import statsmodels.api as smf
import matplotlib.pyplot as plt
import sklearn as sk

# Get work directory
# os.getcwd()
url_rdata = "https://raw.githubusercontent.com/danielegiardiello/ValLogRegMod/main/Data/rdata.csv"
url_vdata = "https://raw.githubusercontent.com/danielegiardiello/ValLogRegMod/main/Data/vdata.csv"
# NOTE: go to 
# "https://github.com/danielegiardiello/ValLogRegMod/blob/main/Data/vdata.csv"
# then click" Raw" button to the upper right corner of the file preview.
# Copy and paste the url link to have the raw gitHub version of the data
rdata = pd.read_csv(url_rdata)
vdata = pd.read_csv(url_vdata)
# Inspect data:
# print(rdata.head(5)) # print the first five rows
# print(vdata.head(5)) # print the first five rows
# rdata.info() # inspect data as in R str()
# vdata.info() # inspect data as in R str()

## Data manipulation ----
# Development data 
# Converting categorical variables to dummies
rdata = pd.get_dummies(data = rdata, 
                       columns = ["ter_pos", "preafp", "prehcg"])
# Dropping columns not needed
rdata.drop(["ter_pos_No", "preafp_No", "prehcg_No"], 
           axis = 1, inplace = True)
print(rdata.head(1)) # check

# Validation data 
vdata = pd.get_dummies(data = vdata, 
                       columns=["ter_pos", "preafp", "prehcg"])
# Dropping columns not needed
vdata.drop(["ter_pos_No", "preafp_No", "prehcg_No"],
            axis = 1,
            inplace = True)
print(vdata.head(1)) # check

## Fitting the logistic regression model ------------------
# Logistic regression using statsmodels library
y = rdata["tum_res"]
X = rdata[["ter_pos_Yes", "preafp_Yes", "prehcg_Yes", "sqpost", "reduc10"]]
X = X.assign(intercept = 1.0)

lrm = smf.GLM(y, X, family = smf.families.Binomial())
result_lrm = lrm.fit()
result_lrm.summary()


# Save estimated predicted probabilites in the development data
pred = result_lrm.predict(X)

# Save coefficients of the developed model
coeff = result_lrm.params

# Save predictors of the validation model
cov = vdata         
cov = cov.assign(intercept = 1.0)
cov = cov[["ter_pos_Yes", "preafp_Yes","prehcg_Yes", "sqpost", "reduc10", "intercept"]]

# Calculating the linear predictor (X*beta)
lp = np.matmul(cov, coeff)

# Calculated the estimated predicted probabilities in the validation data
pred_val = np.exp(lp) / (1 + np.exp(lp))

# Or just use:
# result_lrm.predict(cov)


# Discrimination -------------------
# C-statistic
import lifelines
from lifelines.utils import concordance_index

# Create dataframe val_out containing all info useful
# to assess prediction performance
# y_val = outcome of the validation data
# lp = linear predictor calculated in the validation data
# pred_val = estimated predicted probability in the validation data
val_out =  pd.DataFrame({'y_val': vdata["tum_res"], 
                         'lp_val' : lp,
                         'pred_val' : pred_val})                      
val_out = val_out.assign(intercept = 1.0) # Add intercept

# Creating bootstrap data of validation set to estimate bootstrap
# confidence intervals of performance measures as
# c-stat, discrimination slope and brier score
# NOTE: I need to understand how to set up a random seed to reproduce
# the same boostrapped data
B = 2000
bval_out = {}
for j in range(B): 
  bval_out[j] = sk.utils.resample(val_out, 
      replace = True, 
      n_samples = len(val_out))


# Estimating c-statistic
cstat = concordance_index(val_out.y_val, val_out.lp_val)


# Discrimination slope
val_out_group = val_out.groupby("y_val").mean()
dslope = abs(val_out_group.pred_val[1] - val_out_group.pred_val[0])

# Bootstrap percentile
cstat_boot = [0] * B
val_bgroup = {}
dslope_boot = [0] * B
for j in range(B):
  cstat_boot[j] = concordance_index(bval_out[j].y_val, bval_out[j].lp_val)
  val_bgroup[j] = bval_out[j].groupby("y_val").mean().pred_val
  dslope_boot[j] = abs(val_bgroup[j][1] - val_bgroup[j][0])

# Save results
res_discr = np.reshape(
  (cstat,
   np.percentile(cstat_boot, q = 2.5),
   np.percentile(cstat_boot, q = 97.5), 
   
  dslope,
   np.percentile(dslope_boot, q = 2.5),
   np.percentile(dslope_boot, q = 97.5)),
   
   (2, 3)
)

res_discr = pd.DataFrame(res_discr, 
                         columns = ["Estimate", "2.5 %", "97.5 %"],
                         index = ["C-statistic", "Discrimination slope"])
res_discr

# Calibration --------------

# Calibration intercept (calibration-in-the-large)
# df_cal_int = pd.concat(y_val, lp_val)
cal_int = smf.GLM(val_out.y_val, 
                  val_out.intercept, 
                  family = smf.families.Binomial(),
                  offset = val_out.lp_val)
res_cal_int = cal_int.fit()
res_cal_int.summary()


# Calibration slope
cal_slope = smf.GLM(val_out.y_val, 
                    val_out[["intercept", "lp_val"]], 
                    family = smf.families.Binomial())
res_cal_slope = cal_slope.fit()
res_cal_slope.summary()
res_cal_slope.params[1]

# Calibration plot 
# Method used: The actual probability is estimated using a 'secondary' logistic regression model
# using the predicted probabilities as a covariate.
# Non-parametric curve (smooth using lowess) is also used as an alternative method.

pred_val_cal = pd.DataFrame({'pred_val' : pred_val})
pred_val_cal['intercept'] = 1.0
moderate_cal = smf.GLM(val_out.y_val, 
                       val_out[["intercept", "pred_val"]], 
                       family = smf.families.Binomial())
res_moderate_cal = moderate_cal.fit()
res_moderate_cal.summary()

# Estimated the standard error of the predicted probabilities
# to add confidence bands to the calibration plot estimated using
# a 'secondary' logistic regression model.
# We need: 
# a. matrix of variance and covariance of the 'secondary' logistic model
res_moderate_cal.cov_params()

# b. estimate the linear predictor as x*beta
lp_cal = np.matmul(val_out[["intercept", "pred_val"]],
                  res_moderate_cal.params)

# Estimating the density 
dlogis = sp.stats.logistic.pdf(lp_cal) # logistic density function = exp(-xb) / (1 + exp(-xb))**2)

# Estimating the standard error of predicted probabilities
se_fit = [0] * len(vdata)
for j in range(len(vdata)):
  se_fit[j] = np.dot(dlogis[j], val_out[["intercept", "pred_val"]].loc[j])
  se_fit[j] = np.dot(se_fit[j], res_moderate_cal.cov_params())
  se_fit[j] = np.dot(se_fit[j], val_out[["intercept", "pred_val"]].loc[j].T)
  se_fit[j] = np.dot(se_fit[j], dlogis[j])
se_fit = np.sqrt(se_fit)
# NOTE: I would like to improve and use only matrix operators rather than
# generalizing a single individual case using for loop

# Lowess
lowess = smf.nonparametric.lowess
fit_lowess = lowess(val_out.y_val, 
                    val_out.pred_val, 
                    frac = 2/3,
                    it = 0) # same f and iter parameters as R

# Create df for calibration plot based on secondary log reg
alpha = 0.05
df_cal = pd.DataFrame({
    'obs' :  res_moderate_cal.predict(val_out[["intercept", "pred_val"]]),
    'pred' : val_out.pred_val,
    'se_fit' : se_fit,
    'lower_95' : res_moderate_cal.predict(val_out[["intercept", "pred_val"]]) - sp.stats.norm.ppf(1 - alpha / 2) * se_fit,
    'upper_95' : res_moderate_cal.predict(val_out[["intercept", "pred_val"]]) + sp.stats.norm.ppf(1 - alpha / 2) * se_fit
})

# Sorting
df_cal = df_cal.sort_values(by = ['pred'])

# Calibration plots
# Calibration plots
p1 = plt.plot(df_cal.pred, df_cal.obs, "--", 
         label = "Logistic", color = "black")
p2 = plt.plot(fit_lowess[:, 0], fit_lowess[:, 1], "-",
         color = "blue", label = "Non parametric")  
plt.legend(loc = "upper left")
p3 = plt.plot(df_cal.pred, df_cal.lower_95, "--", 
         label = "Logistic", color = "black")
p4 = plt.plot(df_cal.pred, df_cal.upper_95, "--", 
         label = "Logistic", color = "black")

plt.xlabel("Predicted probability")
plt.ylabel("Actual probability")
plt.title("Calibration plot")
plt.show()
plt.clf()
plt.cla()
plt.close('all')

# Calibration metrics based on a secondary logistic regression model
cal_metrics = pd.DataFrame(
  {'ICI' : np.mean(abs(df_cal.obs - df_cal.pred)),
   'E50' : np.median(abs(df_cal.obs - df_cal.pred)),
   'E90' : np.quantile(abs(df_cal.obs - df_cal.pred), 
                       0.9, 
                       interpolation = 'midpoint')}, 
  index = [0]
)
cal_metrics

# Overall performance measures --------------
# Brier Score
from sklearn.metrics import brier_score_loss
bs_lrm = brier_score_loss(val_out.y_val, val_out.pred_val)


# Scaled brier score
# Develop null model and estimate the Brier Score for the null model
lrm_null = smf.GLM(val_out.y_val, val_out.intercept, 
                   family = smf.families.Binomial())                  
result_lrm_null = lrm_null.fit()
result_lrm_null.summary()
result_lrm_null.params[0]

val_out_null = pd.DataFrame(
  { 
    'y_val' : vdata["tum_res"],
    'lp_null' : [result_lrm_null.params[0]] * len(vdata),
    'pred_null' : np.exp([result_lrm_null.params[0]] * len(vdata)) / (1 + np.exp([result_lrm_null.params[0]] * len(vdata)))
  }
)
bs_lrm_null = brier_score_loss(val_out_null.y_val, 
                               val_out_null.pred_null)
                               
# Bootstrap percentile confidence intervals
B = 2000
bval_out_null = {}
boot_brier = [0] * B
boot_brier_null = [0] * B

for j in range(B): 
  bval_out_null[j] = sk.utils.resample(val_out_null, 
      replace = True, 
      n_samples = len(val_out_null))
      
  boot_brier[j] = brier_score_loss(bval_out[j].y_val, 
                                   bval_out[j].pred_val),
  
  boot_brier_null[j] = brier_score_loss(bval_out_null[j].y_val, 
                                   bval_out_null[j].pred_null)
  
scaled_brier_boot = 1 - (np.array(boot_brier)/np.array(boot_brier_null))

# Overall performance results
overall_metrics = np.reshape(
  (bs_lrm,
   np.percentile(boot_brier, q = 2.5),
   np.percentile(boot_brier, q = 97.5), 
   
  1 - bs_lrm / bs_lrm_null,
   np.percentile(scaled_brier_boot, q = 2.5),
   np.percentile(scaled_brier_boot, q = 97.5)),
   
   (2, 3)
)

overall_metrics = pd.DataFrame(overall_metrics, 
                               columns = ["Estimate", "2.5 %", "97.5 %"], 
                               index = ["Brier Score", "Scaled Brier"])
overall_metrics

# Clinical utility ------
thresholds = np.arange(0, 1, step = 0.01)
leng = np.arange(0, len(thresholds), step = 1)
f_all = np.mean(val_out.y_val)
NB_all = [0] * len(thresholds)
NB = [0] * len(thresholds)

for j in leng:
  NB_all[j] = f_all - (1 - f_all) * (thresholds[j] / (1 - thresholds[j]))
  tdata = val_out[val_out["pred_val"] > thresholds[j]]
  TP = tdata.y_val.sum() # or np.sum(tdata["y_val"])
  FP = (tdata["y_val"] == 0).sum()
  NB[j]= (TP / len(val_out)) - (FP / len(val_out)) * (thresholds[j] / (1 - thresholds[j]))
  
  
# Create dataframe
df_dca = pd.DataFrame({
  'threshold' : thresholds,
  'NB_all' : NB_all,
  'NB' : NB
  }
)

# Plot decision curves
plt.plot(df_dca.threshold, df_dca.NB, "-", color = "black", label = "Prediction model")
plt.plot(df_dca.threshold, df_dca.NB_all, color = "gray", label = "Treat all")
plt.xlim([0, 1])
plt.ylim([-0.05, 0.8])
plt.xlabel("Threshold")
plt.ylabel("Net Benefit")
plt.title("Decision curve - validation data")
plt.axhline(y = 0, linestyle = 'dashdot', color = 'black', label = "Treat none")
plt.legend(loc = "upper right")
plt.show()
plt.show()
plt.clf()
plt.cla()
plt.close('all')

# Next steps: 
# improve plotting and standard error of secondary logistic model 
# Secondary logistic model using non-linear terms e.g. splines.

