library("rjags")
library("ggplot2")

########################################################################################
## Prepare the data
dat <- read.csv("fertilityAndEconomic_shrink.csv", sep=",", dec=".", encoding = "UTF-8")
dat_cols <-  c("DEMO_IND", "Country", "Time", "Value")
dat <- dat[dat_cols]

# Make data cross table
dat <- reshape(dat, idvar=c("Country", "Time"), timevar="DEMO_IND", direction="wide")

max(na.omit(dat$Value.SP_DYN_TFRT_IN))

# Rename columns of values
names <- colnames(dat)
colnames(dat) <- sapply(names, function(x) ifelse(substr(x, 1, 5) == "Value",
                                                  substr(x, 7, 100),
                                                  x))
max(na.omit(dat$SP_DYN_TFRT_IN))

# Remove unnesessery columns
usedCols <- c("Country", "Time", "200101", "NY_GDP_MKTP_CD", "NY_GDP_PCAP_CD",
              "SH_DYN_AIDS_ZS", "SI_POV_DDAY", "SP_DYN_IMRT_IN", "SP_DYN_LE00_IN",
              "SP_POP_GROW", "SP_DYN_TFRT_IN")
dat <- dat[usedCols]

# Remove incomplete data
dat <- na.omit(dat)

# Convert Time to factor to avoid unneсessary comparsion between years numbers.
dat$Time <- as.factor(dat$Time)

# Create train and test data
set.seed(101) # Set Seed so that same sample can be reproduced in future also


########################################################################################
## Analise the data
years <- levels(dat$Time)
countries <- levels(dat$Country)

# Uniform arrangement of the observations in years and countries
pairs(dat[c(1,2)])

# General view on dependencies between fertility and predictors
pairs(dat[-c(1)], panel = panel.smooth)

pairs(dat[dat$Country=="Panama", -c(1)], panel = panel.smooth)
pairs(dat[dat$Time=="2000", -c(1)], panel = panel.smooth)


plot(dat$SP_POP_GROW, dat$SP_DYN_TFRT_IN)
plot(log(dat$SP_POP_GROW), dat$SP_DYN_TFRT_IN)
plot(dat$SP_POP_GROW, log(dat$SP_DYN_TFRT_IN))
plot(log(dat$SP_POP_GROW), log(dat$SP_DYN_TFRT_IN))

# Histogram for fertility rate
ggplot(dat, aes(x=dat$SP_DYN_TFRT_IN)) +
  geom_histogram(bins=80)



########################################################################################
# Model 1. Fit the bayesian model
mod1_string = "model {
  for (i in 1:length(y)) {
    y[i] ~ dnorm(a[i], prec)
    a[i] = b0[k[i]] + b1[k[i]]*population[i] + b2*NY_GDP_MKTP_CD[i] + b3*NY_GDP_PCAP_CD[i] + b4*SH_DYN_AIDS_ZS[i] + b5*SI_POV_DDAY[i] + b6*SP_DYN_IMRT_IN[i] + b7*SP_DYN_LE00_IN[i] + b8*SP_POP_GROW[i]
    k[i] ~ dcat(omega)
  }
  
  for (j in 1:5) {
    b0[j] ~ dnorm(mu0[j], precTau0)
    b1[j] ~ dnorm(mu1[j], precTau1)

    mu0[j] ~ dnorm(0.0, 1.0/10000.0)
    mu1[j] ~ dnorm(0.0, 1.0/10000.0)
  }
  
  b2 ~ dnorm(0.0, 1.0/10000.0)
  b3 ~ dnorm(0.0, 1.0/10000.0)
  b4 ~ dnorm(0.0, 1.0/10000.0)
  b5 ~ dnorm(0.0, 1.0/10000.0)
  b6 ~ dnorm(0.0, 1.0/10000.0)
  b7 ~ dnorm(0.0, 1.0/10000.0)
  b8 ~ dnorm(0.0, 1.0/10000.0)
  
  prec ~ dgamma(1.0/2.0, 1.0*1.0/2.0)
  sig = sqrt(1.0/prec)
  
  precTau0 ~ dgamma(1.0/2.0, 1.0*1.0/2.0)
  tau0 = sqrt(1.0/precTau0)

  precTau1 ~ dgamma(1.0/2.0, 1.0*1.0/2.0)
  tau1 = sqrt(1.0/precTau1)
  
  omega ~ ddirich(c(1.0, 1.0, 1.0, 1.0, 1.0))
} "

# Rename fertility rate to y;
# 200101 to population.
# Exclude Country, Time columns.
data_jags = as.list(cbind(y=log(dat$SP_DYN_TFRT_IN), population=dat$`200101`,
                          dat[-c(1, 2, 3, 11)]))

params1 = c("b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8",
           "sig", "tau0", "tau1", "omega", "k")

mod1 = jags.model(textConnection(mod1_string), data=data_jags, n.chains=3)
update(mod1, 1e4)

mod1_sim = coda.samples(model=mod1,
                       variable.names=params1,
                       n.iter=5e3)
mod1_csim = as.mcmc(do.call(rbind, mod1_sim))

## convergence diagnostics
summary(mod1_sim)
par(mar=c(1,1,1,1))
plot(mod1_sim, ask = T)

autocorr.plot(mod1_sim)
effectiveSize(mod1_sim)
raftery.diag(mod1_csim)

# model converge, but k is about 2 for all observations.
# k parameter dose not have a sense
means1 <-  colMeans(mod1_csim)
k_values <- means1[seq(12,550)]
hist(k_values)

# Classical linear regression estimation of the fertility rate
fit <- lm(log(SP_DYN_TFRT_IN) ~ . , data = dat[-c(1, 2)])
cCoeffs <- summary(fit)$coefficients[,1]

# compare classical linear regression model
# and our mixture model
modelsDiff <- sum((cCoeffs - means1[c(1,6,11,12,13,14,15,16,17)])^2)


########################################################################################
# Model 2. Create the simpliest model
mod2_string = "model {
  for (i in 1:length(y)) {
    y[i] ~ dnorm(a[i], prec)
    a[i] = b0 + b1*population[i] + b2*NY_GDP_MKTP_CD[i] + b3*NY_GDP_PCAP_CD[i] + b4*SH_DYN_AIDS_ZS[i] + b5*SI_POV_DDAY[i] + b6*SP_DYN_IMRT_IN[i] + b7*SP_DYN_LE00_IN[i] + b8*SP_POP_GROW[i]
  }
  
  b0 ~ dnorm(0.0, 1.0/10000.0)
  b1 ~ dnorm(0.0, 1.0/10000.0)
  b2 ~ dnorm(0.0, 1.0/10000.0)
  b3 ~ dnorm(0.0, 1.0/10000.0)
  b4 ~ dnorm(0.0, 1.0/10000.0)
  b5 ~ dnorm(0.0, 1.0/10000.0)
  b6 ~ dnorm(0.0, 1.0/10000.0)
  b7 ~ dnorm(0.0, 1.0/10000.0)
  b8 ~ dnorm(0.0, 1.0/10000.0)
  
  prec ~ dgamma(1.0/2.0, 1.0*1.0/2.0)
  sig = sqrt(1.0/prec)
} "

params2 = c("b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8", "sig")

mod2 = jags.model(textConnection(mod2_string), data=data_jags, n.chains=3)
update(mod2, 1e4)

mod2_sim = coda.samples(model=mod2,
                       variable.names=params2,
                       n.iter=5e4)
mod2_csim = as.mcmc(do.call(rbind, mod2_sim))

## convergence diagnostics
summary(mod2_sim)
par(mar=c(1,1,1,1))
plot(mod2_sim, ask = T)

coda::traceplot(mod2_csim)

autocorr.plot(mod2_sim)
effectiveSize(mod2_sim)
raftery.diag(mod2_csim)
gelman.diag(mod2_sim)

# compare classical linear regression model
# and our mixture model
means2 <-  colMeans(mod2_csim)
models2Diff <- sum((cCoeffs - means2[c(1,2,3,4,5,6,7,8,9)])^2)


#############################################################################################
# Check modeling assumptions of the Model 2

# Function for prediction of one data point x
predictFR <- function(coeffs, dataX){
  logY <-  apply(dataX, 1, function(xRow) sum(coeffs * xRow))
  exp(logY)
}

# compute predictions for given observations
coeffsMod2 <- means2[1:9]
testX <- cbind(b0 = rep(1, nrow(dat)), dat[-c(1,2,11)])
testY <- dat[,11]
yhat <- predictFR(coeffsMod2, testX)

# Residuals
resid <- testY - yhat
plot(testY, resid)
qqnorm(resid)

# Sum of square residuals
sum(resid^2)

# Intervals of highest posterior density for each parameter
HPDinterval(mod2_csim)


# Compare model 1 and 2
dic.samples(mod1, n.iter=1e3)
dic.samples(mod2, n.iter=1e3)


# Compute probability that the coefficient given
# has sign needed
probOfSign <- function(mod_csim, coeffIdx, sig){
  if(sig > 0){
    return(mean(mod_csim[, coeffIdx] > 0))
  }
  return(mean(mod_csim[, coeffIdx] < 0))
}

probOfSign(mod2_csim, 2, -1) # P(b1 < 0)
probOfSign(mod2_csim, 3, -1) # P(b2 < 0)
probOfSign(mod2_csim, 4, -1) # P(b3 < 0)
probOfSign(mod2_csim, 5, 1)  # P(b4 > 0)
probOfSign(mod2_csim, 6, 1)  # P(b5 > 0)
probOfSign(mod2_csim, 7, 1)  # P(b6 > 0)
probOfSign(mod2_csim, 8, -1) # P(b7 < 0)
probOfSign(mod2_csim, 9, -1) # P(b8 < 0)

meanProbOfFirstAssumption <- mean(c(0.759, 0.8513867, 0.04618667, 0.30872,
       0.9998867, 1, 0.7594933, 0))

########################################################################################
# Model 3. Fit the bayesian model with all coefficients relay on classes of countries

mod3_string = "model {
  for (i in 1:length(y)) {
    y[i] ~ dnorm(a[i], prec)
    a[i] = b[1, k[i]] + b[2, k[i]]*population[i] + b[3, k[i]]*NY_GDP_MKTP_CD[i] + b[4, k[i]]*NY_GDP_PCAP_CD[i] + b[5, k[i]]*SH_DYN_AIDS_ZS[i] + b[6, k[i]]*SI_POV_DDAY[i] + b[7, k[i]]*SP_DYN_IMRT_IN[i] + b[8, k[i]]*SP_DYN_LE00_IN[i] + b[9, k[i]]*SP_POP_GROW[i]
    k[i] ~ dcat(omega)
  }
  
  for (j in 1:5) {
    for(u in 1:9) {
      b[u, j] ~ dnorm(mu[u, j], precTau[u])
      mu[u, j] ~ dnorm(0.0, 1.0/10000.0)
    }
  }

  for(u in 1:9) {  
    precTau[u] ~ dgamma(1.0/2.0, 1.0*1.0/2.0)
    tau[u] = sqrt(1.0/precTau[u])
  }

  prec ~ dgamma(1.0/2.0, 1.0*1.0/2.0)
  sig = sqrt(1.0/prec)

  omega ~ ddirich(c(1.0, 1.0, 1.0, 1.0, 1.0))
} "


params3 = c("b", "sig", "tau", "omega", "k")

mod3 = jags.model(textConnection(mod3_string), data=data_jags, n.chains=3)
update(mod3, 1e4)

mod3_sim = coda.samples(model=mod3,
                        variable.names=params3,
                        n.iter=5e4)
mod3_csim = as.mcmc(do.call(rbind, mod3_sim))

## convergence diagnostics
summary(mod3_sim)
par(mar=c(1,1,1,1))
plot(mod3_sim, ask = T)

autocorr.plot(mod3_sim)
effectiveSize(mod3_sim)
raftery.diag(mod3_csim)

# model converge, but k is about 2 for all observations.
# k parameter dose not have a sense
means3 <-  colMeans(mod3_csim)
k_values <- means3[seq(46,568)]
hist(k_values)

# get coefficients of class k
getKCoeffs <- function(means, k){
  means[seq(k*9-8, k*9)]
}

# compare classical linear regression model
# and our mixture model
coeffsMod3 <- getKCoeffs(means3, 2)
models3Diff <- sum((cCoeffs - coeffsMod3)^2)

# Check residuals
yhat3 <- predictFR(coeffsMod3, testX)

# Residuals
resid3 <- testY - yhat3
plot(testY, resid3)
qqnorm(resid3)

# Sum of square residuals
sum(resid3^2)
