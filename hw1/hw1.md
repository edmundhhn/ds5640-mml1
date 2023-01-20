Homework 1
================
Edmund Hui
2023-01-19

``` r
library('class')
library('dplyr')
```

    ## 
    ## Attaching package: 'dplyr'

    ## The following objects are masked from 'package:stats':
    ## 
    ##     filter, lag

    ## The following objects are masked from 'package:base':
    ## 
    ##     intersect, setdiff, setequal, union

``` r
## load binary classification example data from author website 
## 'ElemStatLearn' package no longer available
load(url('https://web.stanford.edu/~hastie/ElemStatLearn/datasets/ESL.mixture.rda'))
dat <- ESL.mixture
```

``` r
plot_mix_data <- function(dat, datboot=NULL) {
  if(!is.null(datboot)) {
    dat$x <- datboot$x
    dat$y <- datboot$y
  }
  plot(dat$x[,1], dat$x[,2],
       col=ifelse(dat$y==0, 'blue', 'orange'),
       pch=20,
       xlab=expression(x[1]),
       ylab=expression(x[2]))
  ## draw Bayes (True) classification boundary
  prob <- matrix(dat$prob, length(dat$px1), length(dat$px2))
  cont <- contourLines(dat$px1, dat$px2, prob, levels=0.5)
  rslt <- sapply(cont, lines, col='purple')
}

plot_mix_data(dat)
```

![](hw1_files/figure-gfm/unnamed-chunk-3-1.png)<!-- -->

``` r
fit_lc <- function(y, x) {
  
  # Original
  #x <- cbind(1, x)
  #beta <- drop(solve(t(x)%*%x)%*%t(x)%*%y)
  
  
  d <- data.frame(y=y, x=x)
  colnames(d) <- c("y", "x1", "x2")
  
  # Using lm
  f <- lm(formula = y ~ x1 + x2, data=d)
  f
}

## make predictions from linear classifier
predict_lc <- function(x, f) {
  #cbind(1, x) %*% beta
  predict(f, data.frame(x))
}

## fit model to mixture data and make predictions
lc_beta <- fit_lc(dat$y, dat$x)
lc_pred <- predict_lc(dat$xnew, lc_beta)

## reshape predictions as a matrix
lc_pred <- matrix(lc_pred, length(dat$px1), length(dat$px2))
contour(lc_pred,
        xlab=expression(x[1]),
        ylab=expression(x[2]))
```

![](hw1_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

``` r
## find the contours in 2D space such that lc_pred == 0.5
lc_cont <- contourLines(dat$px1, dat$px2, lc_pred, levels=0.5)

## plot data and decision surface
plot_mix_data(dat)
sapply(lc_cont, lines)
```

![](hw1_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

    ## [[1]]
    ## NULL

``` r
## do bootstrap to get a sense of variance in decision surface
resample <- function(dat) {
  idx <- sample(1:length(dat$y), replace = T)
  dat$y <- dat$y[idx]
  dat$x <- dat$x[idx,]
  return(dat)
}
```

``` r
## plot linear classifier for three bootstraps
par(mfrow=c(1,3))
for(b in 1:3) {
  datb <- resample(dat)
  ## fit model to mixture data and make predictions
  lc_beta <- fit_lc(datb$y, datb$x)
  print(lc_beta)
  lc_pred <- predict_lc(datb$xnew, lc_beta)
  
  ## reshape predictions as a matrix
  lc_pred <- matrix(lc_pred, length(datb$px1), length(datb$px2))
  
  ## find the contours in 2D space such that lc_pred == 0.5
  lc_cont <- contourLines(datb$px1, datb$px2, lc_pred, levels=0.5)
  
  ## plot data and decision surface
  plot_mix_data(dat, datb)
  sapply(lc_cont, lines)
}
```

    ## 
    ## Call:
    ## lm(formula = y ~ x1 + x2, data = d)
    ## 
    ## Coefficients:
    ## (Intercept)           x1           x2  
    ##    0.284232     0.002014     0.277341

    ## 
    ## Call:
    ## lm(formula = y ~ x1 + x2, data = d)
    ## 
    ## Coefficients:
    ## (Intercept)           x1           x2  
    ##     0.28689      0.02617      0.28134

    ## 
    ## Call:
    ## lm(formula = y ~ x1 + x2, data = d)
    ## 
    ## Coefficients:
    ## (Intercept)           x1           x2  
    ##     0.35367     -0.05417      0.20720

![](hw1_files/figure-gfm/unnamed-chunk-7-1.png)<!-- -->

Using lm with square terms. When including square terms the variance
increases and bias decreases, The model is thus more likely to be
overfitting

``` r
## fit linear classifier
fit_lc <- function(y, x) {
  
  # Original
  #x <- cbind(1, x)
  #beta <- drop(solve(t(x)%*%x)%*%t(x)%*%y)
  
  
  d <- data.frame(y=y, x=x)
  colnames(d) <- c("y", "x1", "x2")
  
  # Using lm
  #f <- lm(formula = y ~ x1 + x2, data=d)
  
  
  f <- lm(formula = y ~ I(x1^2) + I(x2^2) + I(x1*x2) + x1 + x2, data=d)
  f
}

# ## make predictions from linear classifier
# predict_lc <- function(x, f) {
#   #cbind(1, x) %*% beta
#   predict(f, data.frame(x))
# }

## fit model to mixture data and make predictions
lc_beta <- fit_lc(dat$y, dat$x)
lc_pred <- predict_lc(dat$xnew, lc_beta)

## reshape predictions as a matrix
lc_pred <- matrix(lc_pred, length(dat$px1), length(dat$px2))
contour(lc_pred,
        xlab=expression(x[1]),
        ylab=expression(x[2]))
```

![](hw1_files/figure-gfm/unnamed-chunk-8-1.png)<!-- -->

``` r
## find the contours in 2D space such that lc_pred == 0.5
lc_cont <- contourLines(dat$px1, dat$px2, lc_pred, levels=0.5)

## plot data and decision surface
plot_mix_data(dat)
sapply(lc_cont, lines)
```

![](hw1_files/figure-gfm/unnamed-chunk-9-1.png)<!-- -->

    ## [[1]]
    ## NULL

``` r
# ## fit knn classifier
# ## use 5-NN to estimate probability of class assignment
# knn_fit <- knn(train=dat$x, test=dat$xnew, cl=dat$y, k=5, prob=TRUE)
# knn_pred <- attr(knn_fit, 'prob')
# knn_pred <- ifelse(knn_fit == 1, knn_pred, 1-knn_pred)
# 
# ## reshape predictions as a matrix
# knn_pred <- matrix(knn_pred, length(dat$px1), length(dat$px2))
# contour(knn_pred,
#         xlab=expression(x[1]),
#         ylab=expression(x[2]),
#         levels=c(0.25, 0.5, 0.75))
# 
# 
# ## find the contours in 2D space such that knn_pred == 0.5
# knn_cont <- contourLines(dat$px1, dat$px2, knn_pred, levels=0.5)
# 
# ## plot data and decision surface
# plot_mix_data(dat)
# sapply(knn_cont, lines)
```

``` r
## plot linear classifier for three bootstraps
par(mfrow=c(1,3))
for(b in 1:3) {
  datb <- resample(dat)
  ## fit model to mixture data and make predictions
  lc_beta <- fit_lc(datb$y, datb$x)
  print(lc_beta)
  lc_pred <- predict_lc(datb$xnew, lc_beta)
  
  ## reshape predictions as a matrix
  lc_pred <- matrix(lc_pred, length(datb$px1), length(datb$px2))
  
  ## find the contours in 2D space such that lc_pred == 0.5
  lc_cont <- contourLines(datb$px1, datb$px2, lc_pred, levels=0.5)
  
  ## plot data and decision surface
  plot_mix_data(dat, datb)
  sapply(lc_cont, lines)
}
```

    ## 
    ## Call:
    ## lm(formula = y ~ I(x1^2) + I(x2^2) + I(x1 * x2) + x1 + x2, data = d)
    ## 
    ## Coefficients:
    ## (Intercept)      I(x1^2)      I(x2^2)   I(x1 * x2)           x1           x2  
    ##   0.2854408   -0.0006663   -0.0937030   -0.0838526    0.0333306    0.4152098

    ## 
    ## Call:
    ## lm(formula = y ~ I(x1^2) + I(x2^2) + I(x1 * x2) + x1 + x2, data = d)
    ## 
    ## Coefficients:
    ## (Intercept)      I(x1^2)      I(x2^2)   I(x1 * x2)           x1           x2  
    ##    0.280563     0.002728    -0.075230    -0.061567     0.064967     0.408778

    ## 
    ## Call:
    ## lm(formula = y ~ I(x1^2) + I(x2^2) + I(x1 * x2) + x1 + x2, data = d)
    ## 
    ## Coefficients:
    ## (Intercept)      I(x1^2)      I(x2^2)   I(x1 * x2)           x1           x2  
    ##     0.29849     -0.03241     -0.11184     -0.13989      0.14159      0.48752

![](hw1_files/figure-gfm/unnamed-chunk-11-1.png)<!-- -->

``` r
## plot 5-NN classifier for three bootstraps
# par(mfrow=c(1,3))
# for(b in 1:3) {
#   datb <- resample(dat)
#   
#   knn_fit <- knn(train=datb$x, test=datb$xnew, cl=datb$y, k=5, prob=TRUE)
#   knn_pred <- attr(knn_fit, 'prob')
#   knn_pred <- ifelse(knn_fit == 1, knn_pred, 1-knn_pred)
#   
#   ## reshape predictions as a matrix
#   knn_pred <- matrix(knn_pred, length(datb$px1), length(datb$px2))
#   
#   ## find the contours in 2D space such that knn_pred == 0.5
#   knn_cont <- contourLines(datb$px1, datb$px2, knn_pred, levels=0.5)
#   
#   ## plot data and decision surface
#   plot_mix_data(dat, datb)
#   sapply(knn_cont, lines)
# }
# 
# ## plot 20-NN classifier for three bootstraps
# par(mfrow=c(1,3))
# for(b in 1:3) {
#   datb <- resample(dat)
#   
#   knn_fit <- knn(train=datb$x, test=datb$xnew, cl=datb$y, k=20, prob=TRUE)
#   knn_pred <- attr(knn_fit, 'prob')
#   knn_pred <- ifelse(knn_fit == 1, knn_pred, 1-knn_pred)
#   
#   ## reshape predictions as a matrix
#   knn_pred <- matrix(knn_pred, length(datb$px1), length(datb$px2))
#   
#   ## find the contours in 2D space such that knn_pred == 0.5
#   knn_cont <- contourLines(datb$px1, datb$px2, knn_pred, levels=0.5)
#   
#   ## plot data and decision surface
#   plot_mix_data(dat, datb)
#   sapply(knn_cont, lines)
# }
```
