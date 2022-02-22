
library(rio)
library(tidyverse)
library(rms)

testis <- import("C:/Users/dgiardiello/Documents/GitHub/ValLogRegMod/Data/t821.sav")
names(testis) <- tolower(names(testis))
testis <- testis |> 
  mutate(tum_res = case_when(histr3 == 1 ~ 0,
                             histr3 %in% c(2, 3) ~ 1)) |>
  
  mutate(tum_res = as.factor(tum_res),
         
         ter_pos = case_when(ter == 0 ~ 1,
                             ter == 1 ~ 0),
         
         ter_pos = factor(ter_pos, 
                          levels = c(0, 1),
                          labels = c("No", "Yes")),
         
         preafp = case_when(preafp == 0 ~ 1,
                            preafp == 1 ~ 0),
         
         preafp = factor(preafp,
                         levels = c(0, 1),
                         labels = c("No", "Yes")),
         
         prehcg = case_when(prehcg == 0 ~ 1,
                            prehcg == 1 ~ 0),
         
         prehcg = factor(prehcg,
                         levels = c(0, 1),
                         labels = c("No", "Yes"))
  ) 
  


rdata <- testis |>
  
  filter(study == 1) |>
  
  select(patkey, tum_res, ter_pos, preafp,
         prehcg, sqpost, reduc10, 
         lnldhst)

vdata <- testis |>
  
  filter(study == 3) |>
  
  select(patkey, tum_res, ter_pos, preafp,
         prehcg, sqpost, reduc10, 
         lnldhst)

dd <- datadist(rdata, adjto.cat = "first")
options(datadist = "dd")
mod1 <- lrm(tum_res ~ ter_pos + preafp + prehcg + sqpost + reduc10,
            data = rdata, x = T, y = T)

mod1_glm <- glm(tum_res ~ ter_pos + preafp + prehcg + sqpost + reduc10,
            data = rdata, family = binomial)

mod2_glm <- glm(tum_res ~ ter_pos + preafp + prehcg + sqpost + reduc10 + lnldhst,
                data = rdata, family = binomial)

summary(mod1)
options(datadist = NULL)

saveRDS(vdata,
        "C:/Users/dgiardiello/Documents/GitHub/ValLogRegMod/Data/vdata.rds")

attr(rdata$patkey, "ATT") <- NULL
attr(rdata$tum_res, "ATT") <- NULL
attr(rdata$ter, "ATT") <- NULL
attr(rdata$preafp, "ATT") <- NULL
attr(rdata$prehcg, "ATT") <- NULL
attr(rdata$sqpost, "ATT") <- NULL
attr(rdata$reduc10, "ATT") <- NULL
attr(rdata$lnldhst, "ATT") <- NULL

attr(rdata$patkey, "format.spss") <- NULL
attr(rdata$tum_res, "format.spss") <- NULL
attr(rdata$ter, "format.spss") <- NULL
attr(rdata$preafp, "format.spss") <- NULL
attr(rdata$prehcg, "format.spss") <- NULL
attr(rdata$sqpost, "format.spss") <- NULL
attr(rdata$reduc10, "format.spss") <- NULL
attr(rdata$lnldhst, "format.spss") <- NULL

