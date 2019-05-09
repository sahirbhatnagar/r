## ---- lasso-example ----
# to obtain glmnet and install it directly from CRAN.

# install.packages("glmnet", repos = "http://cran.us.r-project.org")

# load the glmnet package and gglasso package for bardet data:

library(glmnet)
library(gglasso)

# The default model used in the package is the Guassian linear
# model or "least squares", which we will demonstrate in this
# section. We load a set of data created beforehand for
# illustration. 
help("bardet")
data("bardet")
x <- bardet$x
y <- bardet$y

# The command loads an input matrix x and a response
# vector y from this saved R data archive.
#
# We fit the model using the most basic call to glmnet.

fit = glmnet(x, y)


# "fit" is an object of class glmnet that contains all the
# relevant information of the fitted model for further use.
# We do not encourage users to extract the components directly.
# Instead, various methods are provided for the object such
# as plot, print, coef and predict that enable us to execute
# those tasks more elegantly.

# We can visualize the coefficients by executing the plot function:

plot(fit)



# Each curve corresponds to a variable. It shows the path of
# its coefficient against the l1-norm of the whole
# coefficient vector at as lambda varies. The axis above
# indicates the number of nonzero coefficients at the
# current lambda, which is the effective degrees of freedom
# (df) for the lasso. Users may also wish to annotate
# the curves; this can be done by setting label = TRUE
# in the plot command.

# A summary of the glmnet path at each step is displayed
# if we just enter the object name or use
# the print function:

print(fit)

# It shows from left to right the number of nonzero
# coefficients (Df), the values of -log(likelihood)
# (%dev) and the value of lambda (Lambda).
# Although by default glmnet calls for 100 values of
# lambda the program stops early if %dev% does not
# change sufficently from one lambda to the next
# (typically near the end of the path.)

# We can obtain the actual coefficients at one or more lambda's
# within the range of the sequence:

coef0 = coef(fit,s=0.1)


# The function glmnet returns a sequence of models
# for the users to choose from. In many cases, users
# may prefer the software to select one of them.
# Cross-validation is perhaps the simplest and most
# widely used method for that task.
#
# cv.glmnet is the main function to do cross-validation
# here, along with various supporting methods such as
# plotting and prediction. We still act on the sample
# data loaded before.

cvfit = cv.glmnet(x, y)

# cv.glmnet returns a cv.glmnet object, which is "cvfit"
# here, a list with all the ingredients of the
# cross-validation fit. As for glmnet, we do not
# encourage users to extract the components directly
# except for viewing the selected values of lambda.
# The package provides well-designed functions
# for potential tasks.

# We can plot the object.

plot(cvfit)

# It includes the cross-validation curve (red dotted line),
# and upper and lower standard deviation curves along the
# lambda sequence (error bars). Two selected lambda's are
# indicated by the vertical dotted lines (see below).

# We can view the selected lambda's and the corresponding
# coefficients. For example,

cvfit$lambda.min

# lambda.min is the value of lambda that gives minimum
# mean cross-validated error. The other lambda saved is
# lambda.1se, which gives the most regularized model
# such that error is within one standard error of
# the minimum. To use that, we only need to replace
# lambda.min with  lambda.1se above.

coef1 = coef(cvfit, s = "lambda.min")


# Note that the coefficients are represented in the
# sparse matrix format. The reason is that the
# solutions along the regularization path are
# often sparse, and hence it is more efficient
# in time and space to use a sparse format.
# If you prefer non-sparse format,
# pipe the output through  as.matrix().


# Predictions can be made based on the fitted
# cv.glmnet object. Let's see a toy example.

predict(cvfit, newx = x[1:5,], s = "lambda.min")


# newx is for the new input matrix and s,
# as before, is the value(s) of lambda at which
# predictions are made.






## ---- adaptive-lasso-example ----



################################################
# penalty.factor example
################################################


# [penalty.factor] argument allows users to apply separate penalty
# factors to each coefficient. Its default is 1 for each parameter,
# but other values can be specified. In particular, 
# any variable with penalty.factor equal to zero is not penalized at all! Let
# v_j denote the penalty factor for j-th variable.

# Note the penalty factors are internally rescaled to sum to nvars.

# This is very useful when people have prior knowledge or
# preference over the variables. In many cases, some
# variables may be so important that one wants to keep
# them all the time, which can be achieved by setting
# corresponding penalty factors to 0:

p.fac = rep(1, 200)
p.fac[c(31, 174, 151)] = 0
pfit = glmnet(x, y, penalty.factor = p.fac)
plot(pfit, xvar="lambda", label = TRUE)

# We see from the labels that the three variables with 0
# penalty factors always stay in the model, while the
# others follow typical regularization paths and
# shrunken to 0 eventually.


################################################
# Adaptive lasso example
################################################

# Some other useful arguments: [exclude] allows one to block
# certain variables from being the model at all. Of course,
# one could simply subset these out of x, but sometimes
# exclude is more useful, since it returns a full vector
# of coefficients, just with the excluded ones set to zero.
# There is also an intercept argument which defaults to
# TRUE; if FALSE the intercept is forced to be zero.


## The adaptive lasso needs a first stage that is consistent. 
## Zou (2006) recommends OLS or ridge
## first stage lasso
thelasso.cv<-cv.glmnet(x,y,family = "gaussian",alpha=1) 
## Second stage weights from the coefficients of the first stage
## coef() is a sparseMatrix
bhat<-as.matrix(coef(thelasso.cv,s="lambda.1se"))[-1,1] 
if(all(bhat==0)){
  ## if bhat is all zero then assign very close to zero weight to all.
  ## Amounts to penalizing all of the second stage to zero.
  bhat<-rep(.Machine$double.eps*2,length(bhat))
}
## the adaptive lasso weight
adpen<-(1/pmax(abs(bhat),.Machine$double.eps)) 
## Second stage lasso (the adaptive lasso)
m_adlasso <- glmnet(x,y,family = "gaussian",alpha=1,exclude=which(bhat==0),
penalty.factor=adpen)
plot(m_adlasso)






## ---- elastic-net ----

################################################
# Elastic net example
################################################

# glmnet provides various options for users to customize
# the fit. We introduce some commonly used options here
# and they can be specified in the glmnet function.

# [family="gaussian"] is the default family option in
# the function glmnet. "gaussian"

# [alpha] is for the elastic-net mixing parameter alpha,
# with range alpha in [0,1]. alpha=1 is the lasso
# (default) and alpha=0 is the ridge.

# [nlambda] is the number of lambda values in the sequence.
# Default is 100.


# As an example, we set alpha=0.2 (more like a ridge regression),
# and give double weights to the latter half of the observations.
# To avoid too long a display here, we set nlambda to 20.
# In practice, however, the number of values of lambda is
# recommended to be 100 (default) or more. In most cases,
# it does not come with extra cost because of the warm-starts
# used in the algorithm, and for nonlinear models leads to
# better convergence properties.

fit = glmnet(x, y, alpha = 0.2, family="gaussian")

plot(fit, xvar = "lambda", label = TRUE)

## ---- elastic-net-example2 ----

foldid=sample(1:10,size=length(y),replace=TRUE)
cv1=cv.glmnet(x,y,foldid=foldid,alpha=1)
cv.5=cv.glmnet(x,y,foldid=foldid,alpha=.5)
cv0=cv.glmnet(x,y,foldid=foldid,alpha=0)

par(mfrow=c(2,2))
plot(cv1);plot(cv.5);plot(cv0)
plot(log(cv1$lambda),cv1$cvm,pch=19,col="red",xlab="log(Lambda)",ylab=cv1$name)
points(log(cv.5$lambda),cv.5$cvm,pch=19,col="grey")
points(log(cv0$lambda),cv0$cvm,pch=19,col="blue")
legend("topleft",legend=c("alpha= 1","alpha= .5","alpha 0"),pch=19,col=c("red","grey","blue"))

## ---- group-lasso ----

################################################
# Group lasso example
################################################

install.packages("gglasso", repos = "http://cran.us.r-project.org")
# load gglasso library
library(gglasso)

# load bardet data set
data(bardet)

# define 20 groups 
group1 <- rep(1:20,each=5)

# fit group lasso penalized least squares
m1 <- gglasso(x=bardet$x,y=bardet$y,group=group1,loss="ls")

plot(m1)

# 5-fold cross validation using group lasso 
# penalized ls regression
cv <- cv.gglasso(x=bardet$x, y=bardet$y, group=group1, loss="ls",
pred.loss="L2", lambda.factor=0.05, nfolds=5)

plot(cv)

# load colon data set
data(colon)

# define group index
group2 <- rep(1:20,each=5)

# fit group lasso penalized logistic regression
m2 <- gglasso(x=colon$x,y=colon$y,group=group2,loss="logit")

plot(m2)

# 5-fold cross validation using group lasso 
# penalized logisitic regression
cv2 <- cv.gglasso(x=colon$x, y=colon$y, group=group2, loss="logit",
pred.loss="misclass", lambda.factor=0.05, nfolds=5)

plot(cv2)

# the coefficients at lambda = lambda.1se
pre = coef(cv$gglasso.fit, s = cv$lambda.1se)





## ---- simulation-demo-1 ----

# Compare lasso, adaptive lasso, scad and 
# mcp in the logistic regression and least squares
#
# load R libraries
# glmnet: for LASSO, adaptive LASSO, elastic net 
#			penalized least squares and logistic regression (all 
#   		supports poisson, multinomial and cox model)
# ncvreg: for SCAD and MCP penalized least squares and logistic regression
library(glmnet)
library(ncvreg)
library(MASS)

##############################
#	PART I Least squares

n=100
p=200

# true beta
truebeta <- c(4,4,4,-6*sqrt(2),4/3,rep(0,p-5))

# error variance 
sigma2 <- 0.3

# The covariance between Xj and Xk is cov(X_j, X_k) = rho^|i-j|
# covariance matrix
covmat <- function(rho, p) {
  rho^(abs(outer(seq(p), seq(p), "-")))
}
# rho = 0.1, generate covariance matrix for X
sigma <- covmat(0.1,p)

# X ~ N(0, sigma)
# epsilon ~ N(0, sigma2)
# The true model:  	y = x*truebeta + epsilon
x <- mvrnorm(n,rep(0,p),sigma)
epsilon <- rnorm(n,0,sd=sqrt(sigma2))
y <- x %*% truebeta + epsilon
  
# fit lasso and use five-fold CV to select lambda
cvfit <- cv.glmnet(x = x, y = y, alpha = 1, family = "gaussian")
plot(cvfit)

# lambda selected by CV
cvfit$lambda.min

# plot the solution paths
plot(cvfit$glmnet.fit,label=TRUE,xvar="lambda")
# plot lambda.min
abline(v=log(cvfit$lambda.min),lty=1)
# plot lambda.1se
abline(v=log(cvfit$lambda.1se),lty=2)



model_compare <- matrix(NA, nrow=5, ncol=p,
                dimnames=list(c("true model",
				"lasso","adaptive lasso","mcp","scad"),
				paste("V",seq(p),sep="")))

# save the true model
model_compare[1, ] <- truebeta


## coefficients estimated by lasso
cvfit <- cv.glmnet(x = x, y = y, alpha = 1, 
family = "gaussian")
tmp <- cvfit$glmnet.fit$beta
lasso_beta <- as.matrix(tmp[,cvfit$lambda==cvfit$lambda.min])
model_compare[2, ] <- lasso_beta


# Compute weight and fit an adaptive lasso
weight = 1/(lasso_beta)^2
# Some est. coef. is zero, the corresponding weight is Inf
# to prevent numerical error, convert Inf to a large number (e.g. 1e6)
weight[weight==Inf] = 1e6

cvfit <- cv.glmnet(x = x, y = y, alpha = 1, 
		family = "gaussian", 
		nfolds = 5, penalty.factor=weight)

## coefficients estimated by adaptive lasso
tmp <- cvfit$glmnet.fit$beta
adaptive_lasso_beta <- as.matrix(tmp[,cvfit$lambda==cvfit$lambda.min])
model_compare[3, ] <- adaptive_lasso_beta

## coefficients estimated by mcp
cvfit <- cv.ncvreg(X = x, y = y, penalty = "MCP", family = "gaussian")
mcp_beta <- cvfit$fit$beta[, cvfit$min]
model_compare[4, ] <- mcp_beta[-1]

## coefficients estimated by scad
cvfit <- cv.ncvreg(X = x, y = y, penalty = "SCAD", family = "gaussian")
scad_beta <- cvfit$fit$beta[, cvfit$min]
model_compare[5, ] <- scad_beta[-1]

# make a comparison of the estimated coef. from four methods
# we see that lasso over-selected, adaptive lasso, scad and mcp fix the problem.

model_compare





## ---- simulation-demo-2 ----

##############################
#	PART II logistic regression 


# generate data
n <- 200 
p <- 8

# truebeta
truebeta <- c(6,3.5,0,5,rep(0,p-4)) 
truebeta

# generate x and y from the true model
# The true model:  	P(y=1|x) = exp(x*truebeta)/(exp(x*truebeta)+1)
x <- matrix(rnorm(n*p), n, p)
feta <- x %*% truebeta 
fprob <- ifelse(feta < 0, exp(feta)/(1+exp(feta)), 1/(1 + exp(-feta)))
y <- rbinom(n, 1, fprob)


model_compare <- matrix(NA, nrow=5, ncol=p,
                        dimnames=list(c("true model","lasso",
						"adaptive lasso","mcp","scad"),
						paste("V",seq(p),sep="")))


# save the true model
model_compare[1, ] <- truebeta


# lasso case
# cv.glmfit fit lasso model and use cross validation for lambda selection

# family = "binomial", logistic regression
# family = "gaussian", least squares

# alpha controls the degree of L1 penalty term, 
#		alpha = 1, lasso, 
#		alpha = 0, ridge, 
#		alpha = (0,1), elastic net

# nfolds = 5, five-fold cross validation (CV)
cvfit <- cv.glmnet(x = x, y = y, alpha = 1, family = "binomial", nfolds = 5)

# make a plot of the CV result 
# the left vertical line (lambda.min) correspondes to the lambda 
# that gives smallest deviance.
# the right vertical line (lambda.1se) correspondes to the lambda 
# from the one standard deviation rule
plot(cvfit)

# plot the solution paths
plot(cvfit$glmnet.fit,label=TRUE,xvar="lambda")
# plot lambda.min
abline(v=log(cvfit$lambda.min),lty=1)
# plot lambda.1se
abline(v=log(cvfit$lambda.1se),lty=2)


# save the Lasso coefficient from solution path, each column 
# represents an estimate for a lambda value
tmp <- cvfit$glmnet.fit$beta
tmp
# the beta that correspondes to lambda.min selected by CV
lasso_beta <- as.matrix(tmp[,cvfit$lambda==cvfit$lambda.min])
model_compare[2, ] <- lasso_beta


# Compute weight and fit an adaptive lasso
weight = 1/(lasso_beta)^2
# Some est. coef. is zero, the corresponding weight is Inf
# to prevent numerical error, convert Inf to a large number (e.g. 1e6)
weight[weight==Inf] = 1e6

cvfit <- cv.glmnet(x = x, y = y, alpha = 1, family = "binomial", 
						nfolds = 5, penalty.factor=weight)

## coefficients estimated by adaptive lasso
tmp <- cvfit$glmnet.fit$beta
adaptive_lasso_beta <- as.matrix(tmp[,cvfit$lambda==cvfit$lambda.min])
model_compare[3, ] <- adaptive_lasso_beta


## coefficients estimated by mcp
cvfit <- cv.ncvreg(X = x, y = y, penalty = "MCP", family = "binomial")
mcp_beta <- cvfit$fit$beta[, cvfit$min]
model_compare[4, ] <- mcp_beta[-1]

## coefficients estimated by scad
cvfit <- cv.ncvreg(X = x, y = y, penalty = "SCAD", family = "binomial")
scad_beta <- cvfit$fit$beta[, cvfit$min]
model_compare[5, ] <- scad_beta[-1]

# make a comparison of the estimated coef. from four methods
model_compare
