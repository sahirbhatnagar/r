# McGill Summer School in Health Data Analytics

## Data analysis using penalized regression methods

In high-dimensional (HD) data, where the number of covariates (p) greatly exceeds the number of observations (n), estimation can benefit from the bet-on-sparsity principle, i.e., only a small number of predictors are relevant in the response. This assumption can lead to more interpretable models, improved predictive accuracy, and algorithms that are computationally efficient. In medical data, where the sample sizes are particularly small due to high data collection costs, we must often assume a sparse model because there isn’t enough information to estimate p parameters. For these reasons, penalized regression methods have generated substantial interest over the past decade since they can set model coefficients exactly to zero. We will provide an overview of the lasso and group-lasso; two of the most popular penalized regressions techniques available. We will provide details on both the theoretical and computational aspects of these methods and demonstrate a real-data example with R code.


RStudio: [![Binder](http://mybinder.org/badge_logo.svg)](http://mybinder.org/v2/gh/sahirbhatnagar/mcgillHDA/master?urlpath=rstudio)

[Slides](https://github.com/sahirbhatnagar/mcgillHDA/raw/master/Bhatnagar_penalized_regression_McGill_Health_data_analytics_2019.pdf)



