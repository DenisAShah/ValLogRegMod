# Load libraries and data
import pandas as pd
import numpy as np
import math
import statsmodels.api as smf

# Get work directory
# os.getcwd()
url = 'https://github.com/danielegiardiello/ValLogRegMod/blob/main/Data/rdata.csv'
file_rdata = "C:/Users/dgiardiello/Documents/GitHub/ValLogRegMod/Data/rdata.csv"
file_vdata = "C:/Users/dgiardiello/Documents/GitHub/ValLogRegMod/Data/vdata.csv"
rdata = pd.read_csv(file_rdata)
vdata = pd.read_csv(file_vdata)
print(rdata.head(5)) # print the first five rows
print(vdata.head(5)) # print the first five rows
rdata.info() # inspect data as in R str()
vdata.info() # inspect data as in R str()

## Data manipulation ---
# Development data 
# Converting categorical variables to dummies
rdata = pd.get_dummies(data = rdata, columns=["ter_pos", "preafp", "prehcg"])
# Dropping columns not needed
rdata.drop(["ter_pos_No", "preafp_No", "prehcg_No"], axis = 1, inplace = True)
print(rdata.head(1))

# Validation data 
vdata = pd.get_dummies(data = vdata, columns=["ter_pos", "preafp", "prehcg"])
# Dropping columns not needed
vdata.drop(["ter_pos_No", "preafp_No", "prehcg_No"], axis = 1, inplace = True)
print(vdata.head(1))

## Fitting the logistic regression model ------------------
# Logistic regression using statsmodels library
y = rdata["tum_res"]
X = rdata[["ter_pos_Yes", "preafp_Yes", "prehcg_Yes", "sqpost", "reduc10"]]
X['intercept'] = 1.0

lrm = smf.GLM(y, X, family = smf.families.Binomial())
result_lrm = lrm.fit()
result_lrm.summary()

# Save estimated predicted probabilites in the development data
pred = result_lrm.predict(X)

# Save coefficients of the developed model
coeff = result_lrm.params

# Save predictors of the validation model
cov = vdata         
cov["intercept"] = 1
cov = cov[["ter_pos_Yes", "preafp_Yes","prehcg_Yes", "sqpost", "reduc10", "intercept"]]

# Calculating the linear predictor (X*beta)
lp = np.multiply(coeff, cov)
lp = lp.sum(axis = 1)

# Calculated the estimated predicted probabilities in the validation data
pred_val = np.exp(lp) / (1 + np.exp(lp))


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
val_out['intercept'] = 1.0 # Add intercept

# Estimating c-statistic
concordance_index(val_out.y_val, val_out.lp_val)

# Discrimination slope
val_out_group = val_out.groupby("y_val").mean()
dslope = abs(val_out_group.pred_val[1] - val_out_group.pred_val[0])
dslope

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

# Calibration plot based on a secondary logistic regression model
pred_val_cal = pd.DataFrame({'pred_val' : pred_val})
pred_val_cal['intercept'] = 1.0
moderate_cal = smf.GLM(val_out.y_val, 
                       val_out[["intercept", "pred_val"]], 
                       family = smf.families.Binomial())
res_moderate_cal = moderate_cal.fit()
res_moderate_cal.summary()

df_cal = pd.DataFrame({
    'obs' :  res_moderate_cal.predict(val_out[["intercept", "pred_val"]]),
    'pred' : val_out.pred_val
})

# Sorting
df = df_cal.sort_values(by = ['pred'])

# Calibration plots
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(df_cal.pred, df_cal.obs, "--", label = "Logistic")
ax.legend(loc = "best")
plt.show()
# NOTE: work in progress


# Calibration metrics based on a secondary logistic regression model
cal_metrics = pd.DataFrame(
  {'ICI' : np.mean(abs(df_cal.obs - df_cal.pred)),
   'E50' : np.median(abs(df_cal.obs - df_cal.pred)),
   'E90' : np.quantile(abs(df_cal.obs - df_cal.pred), 
                       0.9)}, index = [0]
)
cal_metrics

# Overall performance measures --------------
# Brier Score
from sklearn.metrics import brier_score_loss
bs_lrm = brier_score_loss(y_val, pred_val)
bs_lrm


# Scaled Brier Score

# Clinical utility ------
# NOTE: work in progress

