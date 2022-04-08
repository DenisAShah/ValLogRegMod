# Basic model
fit_lrm <- glm(tum_res ~ 
                 ter_pos + preafp + prehcg + 
                 sqpost + reduc10,
               data = rdata,
               family = binomial,
               x = T,
               y = T
)

str(rdata)


# Design matrix of predictors
des_matr <- as.data.frame(model.matrix(~ ter_pos + preafp + prehcg + sqpost + reduc10, 
                                       data = vdata))

round(fit_lrm$coefficients, 3)
# Coefficients
coeff <- c(-0.302, 0.995, 0.859, 0.554, 0.074, -0.264)
# Prognostic index (PI)
vdata$LP <- as.vector(as.matrix(des_matr) %*% cbind(coeff))
vdata$pred_calc <- exp(vdata$LP) / (1 + exp(vdata$LP))
