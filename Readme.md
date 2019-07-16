# Efficient statistical computing with C++ Armadillo
*Maciej Bak*  
*Swiss Institute of Bioinformatics*

This is a very small repository that presents the libraries needed for efficient statistical computing in R interfacing to C++. Functions presented are useful to build statistical models based on likelihood and optimize their parameters.  
A gentle introduction to Rcpp may be found at:  
[http://adv-r.had.co.nz/Rcpp.html][link1]  
A more comprehensive guide is available at:  
[https://teuder.github.io/rcpp4everyone_en][link2]  

// Armadillo documentation is available at:
// http://arma.sourceforge.net/docs.html

## Dependencies
G++, ARMADILLO 9.5
The software has been tested with R version 3.5.1. Required packages:
* Rcpp (>1.0.0)
* numDeriv (>2016.8-1)
* tictoc (>1.0)

Requirements might be installed directly from within the R interpreter:
```
$ R
> install.packages(c("Rcpp", "numDeriv", "tictoc"))
> q()
```
// Compile with:
// $ g++ -std=c++11 main.cpp -o exe -O2 -larmadillo

## Repository
This repository contains six files:

| File | Description |
| ------ | ------ |
| README.md | (this file) |
| tests.R | R script with example calls of C++ functions |
| tests.cpp | C++ functions with optimisation-related algorithms |
| zeroin.c | C routine for root finding of an univariate function |
| modified_optim.c | modified version of the C file with a method to minimize a given multivariate function |
| LICENSE | The GNU General Public License v3.0 |

These libraries and functions should be enough to implement complex Bayesian models, fit the parameters efficiently and optimize the runtime on big datasets. Functions' logic and sygnatures are described within the files above.

## License
GNU General Public License 


[link1]: <http://adv-r.had.co.nz/Rcpp.html>
[link2]: <https://teuder.github.io/rcpp4everyone_en>
