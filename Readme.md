# Efficient statistical computing with C++ Armadillo
*Maciej Bak*  
*Swiss Institute of Bioinformatics*

Toy exampes on the usage of C++ Armadillo package for linear algebra and simple statistics as well as other custom functions useful for statistical modelling. These resources should be enough to implement complex Bayesian models, fit the parameters efficiently (likelihood optimization) and minimize the runtime on big datasets. Functions' logic and sygnatures are described within the main source file of this repository.

## Dependancies, Installation & Tests
The only external library requred to test this repository is teh C++ library [Armadillo][link1].  
Extensive notes on the installation are provided in the [FAQ][link2].  
A comprehensive description of the classes and their methods can be found in the [Documentation][link3].  
Make note that Armadillo requires [BLAS][link4] and [LAPACK][link5] for marix operations.

The following repository has been tested on Linux Ubuntu 14.04.4 with g++ 4.8.4 and Armadillo 9.500.2 
```
$ # At first - make sure the system is updated
$ sudo apt update
$ sudo apt upgrade
$ # Install BLAS and LAPACK
$ sudo apt install cmake libopenblas-dev liblapack-dev
$ # Download and extract the latest table release of Armadillo
$ # Assuming it is in: $HOME/Downloads/arma: install the library
$ cd $HOME/Downloads/arma
$ cmake .
$ make
$ sudo make install
$ # Compile and run the examples in this repository
$ # Assuming it is in: $HOME/angry-statistical-computing: install the library
$ cd $HOME/angry-statistical-computing
$ g++ main.cpp -o test -DARMA_DONT_USE_WRAPPER -lopenblas -llapack -std=c++11 -O2
$ ./test
```

## Repository
This repository contains six files:

| File | Description |
| ------ | ------ |
| README.md | (this file) |
| main.cpp | main C++ source file with all the examples |
| numDeriv.c | C library with functions to approximate gradient and hessian of a given multivariate function |
| zeroin.c | C routine for root finding of an univariate function |
| modified_optim.c | modified version of the C file with a method to minimize a given multivariate function |
| LICENSE | The GNU General Public License v3.0 |

## License
Apache 2.0

[link1]: <http://arma.sourceforge.net/>
[link2]: <http://arma.sourceforge.net/faq.html>
[link3]: <http://arma.sourceforge.net/docs.html>
[link4]: <http://www.netlib.org/blas/>
[link5]: <http://www.netlib.org/lapack/>
