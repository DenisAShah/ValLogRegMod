# Assessing the performance of prediction models with binary outcomes: a framework for some traditional and novel measures 

R Code repository for the manuscript ['Assessing the performance of prediction models: a framework for some traditional and novel measures'](https://journals.lww.com/epidem/Fulltext/2010/01000/Assessing_the_Performance_of_Prediction_Models__A.22.aspx) by Steyerberg et al. (2010). We provided how to develop and validate a risk prediction model for binary outcomes using logistic regression including more recent updates about performance assessment (e.g. using calibration hierarchy definitions by Van Calster et al. (2016) [here](https://www.sciencedirect.com/science/article/pii/S0895435615005818)) .



The repository contains the following code:  

+ Minimal and essential [code](https://github.com/danielegiardiello/ValLogRegMod/blob/main/01_PredLogReg_minimal.R) to develop and validate a risk prediction model with binary outcomes when both development and validation data are available. People with basic or low statistical knowledge and basic R programming knowledge are encouraged to use these files. **To reproduce the main results of the manuscript, this script is sufficient**.  A "quick and experimental"  Python code is [here](https://github.com/danielegiardiello/ValLogRegMod/blob/main/01_PredLogReg_minimal.py) to reproduce the main results of the manuscript. I encourage users to install and use Python in RStudio as suggested, for example, [here](https://support.rstudio.com/hc/en-us/articles/360023654474-Installing-and-Configuring-Python-with-RStudio).

+ Minimal and essential [code](https://github.com/danielegiardiello/ValLogRegMod/blob/main/02_PredLogReg_minimal.R) to validate a risk prediction model in a external data when model equation of a developed risk prediction model is available. A more extensive and elaborated output is [here](https://github.com/danielegiardiello/ValLogRegMod/blob/main/02_PredLogReg.md) and the corresponding .Rmd source code is [here](https://github.com/danielegiardiello/ValLogRegMod/blob/main/02_PredLogReg.Rmd).

+ Extensive output and [code](https://github.com/danielegiardiello/ValLogRegMod/blob/main/03_PredLogReg.md) to develop and validate a risk prediction model with a binary outcome. The .Rmd source code is [here](https://github.com/danielegiardiello/ValLogRegMod/blob/main/03_PredLogReg.Rmd). People with advanced knowledge in statistics are encouraged to use these files.

External [functions](https://github.com/danielegiardiello/ValLogRegMod/tree/main/Functions) and [figures](https://github.com/danielegiardiello/ValLogRegMod/tree/main/imgs) are available in the corresponding subfolders.  


## Usage

You can either download a zip file containing the directory, or you can clone it by using

```bash
git clone https://github.com/danielegiardiello/ValRegMod.git
```

In either case, you can then use the `ValRegMod.Rproj` file to open
and Rstudio session in the directory you have just downloaded. You may then knit
both rmarkdown files, or run them line-by-line.


## Contributions

| Name                                                         | Affiliation                           | Role                  |
| ------------------------------------------------------------ | ------------------------------------- | ----------------------|
| [Daniele Giardiello](https://github.com/danielegiardiello/)  | The Netherlands Cancer Institute (NL) <br /> Leiden University Medical Center (NL) <br /> EURAC research (IT) | Author/maintainer     |



