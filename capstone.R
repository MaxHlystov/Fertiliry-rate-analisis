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
# Now Selecting 85% of data as sample from total 'n' rows of the data  
dataNRows <- nrow(dat)
trainNRows <- floor(.85*dataNRows)
sample <- sample.int(n = dataNRows, size = trainNRows, replace = F)
trainDat <- dat[sample, ]
testDat  <- dat[-sample, ]


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


plot(log(dat$`200101`), log(dat$SP_DYN_TFRT_IN))

# Histogram for fertility rate
ggplot(dat, aes(x=dat$SP_DYN_TFRT_IN)) +
  geom_histogram(bins=80)

# Classical linear regression estimation of the fertility rate
fit <- lm(SP_DYN_TFRT_IN ~ . , data = dat)
summary(fit)
plot(fit)


########################################################################################
# Fit the bayesian model
mod_string = "model {
  for (i in 1:length(y)) {
    y[i] ~ dnorm(la[i], prec)
    log(la[i]) = b0[k[i]] + b1[k[i]]*population[i] + b2*NY_GDP_MKTP_CD[i] + b3*NY_GDP_PCAP_CD[i] + b4*SH_DYN_AIDS_ZS[i] + b5*SI_POV_DDAY[i] + b6*SP_DYN_IMRT_IN[i] + b7*SP_DYN_LE00_IN[i] + b8*SP_POP_GROW[i]
    k[i] ~ dcat(omega)
  }

  for (j in 1:5) {
    b0[j] ~ dnorm(mu0[j], precTau)
    b1[j] ~ dnorm(mu1[j], precTau)
    
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

  precTau ~ dgamma(1.0/2.0, 1.0*1.0/2.0)
  tau = sqrt(1.0/precTau)

  omega ~ ddirich(c(0.1, 0.1, 0.1, 0.1, 0.1))
} "

set.seed(11)

# Rename fertility rate to y;
# 200101 to population.
# Exclude Country, Time columns.
data_jags = as.list(cbind(y=dat$SP_DYN_TFRT_IN, population=dat$`200101`,
                          dat[-c(1, 2, 3, 11)]))

params = c("b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8",
           "mu0", "mu1", "sig", "tau", "omega")

mod = jags.model(textConnection(mod_string), data=data_jags, n.chains=3)
update(mod, 1e3)

mod_sim = coda.samples(model=mod,
                       variable.names=params,
                       n.iter=5e3)
mod_csim = as.mcmc(do.call(rbind, mod_sim))

## convergence diagnostics
summary(mod_sim)
par(mar=c(1,1,1,1))
plot(mod_sim)


coda::traceplot(mod_csim)

autocorr.plot(mod_sim)
effectiveSize(mod_sim)
raftery.diag(mod_csim)
gelman.diag(mod_sim)
