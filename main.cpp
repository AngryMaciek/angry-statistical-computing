//
// Efficient statistical computing with C++ Armadillo
//
// Maciej Bak
// Swiss Institute of Bioinformatics
// 15.07.2019
//

#include <iostream> //IO streams
#include <armadillo> // Armadillo library for efficient computing
#include <sys/stat.h> // Required to create directories
#include "modified_optim.c" // Minimisation of a multivariate function
extern "C" {
  #include "zeroin.c" // Root search of a univariate function
}
#include "numDeriv.c" // gradient and hessian approximation
#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

//=======================================================================================

// Simplest examples from Armadillo library 
void armadillo_toy_examples(){

  // Generate a vector from the standard uniform distribution
  arma::vec v_uniform = arma::randu(10);
  v_uniform.print("v_uniform:");
  
  // Generate a vector from the standard normal distribution
  arma::vec v_normal = arma::randn(10);
  v_normal.print("v_normal:");

  // Create a 4x4 random matrix and print it on the screen
  arma::Mat<double> A = arma::randu(4,4);
  std::cout << "A:\n" << A << "\n";

  // Multiply A with his transpose:
  std::cout << "A * A.t() =\n";
  std::cout << A * A.t() << "\n";

  // Access/Modify rows and columns from the array:
  A.row(0) = A.row(1) + A.row(3);
  A.col(3).zeros();
  std::cout << "add rows 1 and 3, store result in row 0, also fill 4th column with zeros:\n";
  std::cout << "A:\n" << A << "\n";

  // Create a new diagonal matrix using the main diagonal of A:
  arma::Mat<double>B = arma::diagmat(A);
  std::cout << "B:\n" << B << "\n";

  // New matrix: directly specify the matrix size (elements are uninitialised)
  // Watch out: printing uninitialised objects causes memory errors ~valgrind
  arma::mat C(2,3);  // typedef mat  =  Mat<double>
  std::cout << "C.n_rows: " << C.n_rows << std::endl;
  std::cout << "C.n_cols: " << C.n_cols << std::endl;

  // Directly access an element (indexing starts at 0)
  C(1,2) = 456.0;
  std::cout << "C[1][2]:\n" << C(1,2) << "\n";
  C = 5.5; // scalars are treated as a 1x1 matrix
  C.print("C:");

  // Change the size (data is not preserved)
  C.set_size(4,5);
  C.fill(5.0); // set all elements to a particular value
  C.print("C:");

  // Create matrix by-hand
  // endr indicates "end of row"
  C << 0.165300 << 0.454037 << 0.995795 << 0.124098 << 0.047084 << arma::endr
    << 0.688782 << 0.036549 << 0.552848 << 0.937664 << 0.866401 << arma::endr
    << 0.348740 << 0.479388 << 0.506228 << 0.145673 << 0.491547 << arma::endr
    << 0.148678 << 0.682258 << 0.571154 << 0.874724 << 0.444632 << arma::endr
    << 0.245726 << 0.595218 << 0.409327 << 0.367827 << 0.385736 << arma::endr;
  C.print("C:");

  // Determinant
  std::cout << "det(C): " << det(C) << std::endl;

  // Inverse
  std::cout << "inv(C): " << std::endl << inv(C) << std::endl;

  // Submatrices
  std::cout << "C( span(0,2), span(3,4) ):" << std::endl << C( arma::span(0,2), arma::span(3,4) ) << std::endl;
  std::cout << "C( 0,3, size(3,2) ):" << std::endl << C( 0,3, arma::size(3,2) ) << std::endl;
  std::cout << "C.row(0): " << std::endl << C.row(0) << std::endl;
  std::cout << "C.col(1): " << std::endl << C.col(1) << std::endl;

  // Min-Max:
  // Maximum from each column (traverse along rows)
  std::cout << "max(C): " << std::endl << max(C) << std::endl;
  // Maximum from each row (traverse along columns)
  std::cout << "max(C,1): " << std::endl << max(C,1) << std::endl;  
  // Maximum value in C
  std::cout << "max(max(C)) = " << max(max(C)) << std::endl; 

  // SUMS
  // Sum of each column (traverse along rows)
  std::cout << "sum(C): " << std::endl << sum(C) << std::endl;
  // Sum of each row (traverse along columns)
  std::cout << "sum(C,1) =" << std::endl << sum(C,1) << std::endl;
  // Sum of all elements
  std::cout << "accu(C): " << accu(C) << std::endl;
  // Trace = sum along diagonal
  std::cout << "trace(C): " << trace(C) << std::endl;
  
  // Generate the identity matrix
  arma::mat D = arma::eye<arma::mat>(4,4);
  D.print("D:");

  // Row vectors are treated like a matrix with one row
  arma::rowvec r;
  r << 0.59119 << 0.77321 << 0.60275 << 0.35887 << 0.51683;
  r.print("r:");
  
  // Column vectors are treated like a matrix with one column
  arma::vec q;
  q << 0.14333 << 0.59478 << 0.14481 << 0.58558 << 0.60809;
  q.print("q:");
  
  // Convert matrix to vector; data in matrices is stored column-by-column
  arma::vec v = vectorise(C);
  v.print("v:");
  
  // Dot or inner product
  std::cout << "as_scalar(r*q): " << as_scalar(r*q) << std::endl;
  
  // Outer product
  std::cout << "q*r: " << std::endl << q*r << std::endl;
  
  // Multiply-and-accumulate operation (no temporary matrices are created)
  std::cout << "accu(A % B) = " << accu(A % B) << std::endl;

  // imat specifies an integer matrix
  arma::imat AA;
  arma::imat BB;
  AA << 1 << 2 << 3 << arma::endr << 4 << 5 << 6 << arma::endr << 7 << 8 << 9;
  BB << 3 << 2 << 1 << arma::endr << 6 << 5 << 4 << arma::endr << 9 << 8 << 7;
  
  // Comparison of matrices (element-wise); output of a relational operator is a umat {0,1}
  arma::umat ZZ = (AA >= BB);
  ZZ.print("ZZ:");

  // cubes ("3D matrices")
  arma::cube Q( C.n_rows, C.n_cols, 2 );
  Q.slice(0) = C;
  Q.slice(1) = 2.0 * C;
  Q.print("Q:");

  // 2D field of matrices; 3D fields are also supported
  // Field - class for storing arbitrary objects in matrix-like or cube-like layouts
  arma::field<arma::mat> F(4,3); 
  for(arma::uword col=0; col < F.n_cols; ++col)
    for(arma::uword row=0; row < F.n_rows; ++row){
      F(row,col) = arma::randu<arma::mat>(2,3);  // each element in field<arma::mat> is a matrix
    }
  F.print("F:");

  // Rotate a point (0,1) by -Pi/2 ---> (1,0)
  arma::vec Pos = {0,1};
  Pos.print("Current coordinates of the point:"); //Gets printed as (x \n y) !
  double phi = -3.14159265359/2;
  // Rotation Matrix:
  arma::mat RotM = {{+cos(phi), -sin(phi)},
                    {+sin(phi), +cos(phi)}};
  std::cout << "Rotating the point " << phi*180/3.14159265359 << " deg" << std::endl;
  Pos = RotM*Pos;
  Pos.print("New coordinates of the point:");

}

//=======================================================================================

// Examples on IO: Matrices, Vectors
void armadillo_read_write_objects(){

  // Create a 5x5 matrix with random values coming from uniform distribution on [0;1]
  arma::Mat<double> uniform_matrix = arma::randu(3,5);
  // Alternatively:
  //arma::mat uniform_matrix = arma::randu<arma::mat>(4,4);
  //arma::mat uniform_matrix(4,4,arma::fill::randu);
  std::cout << "3x5 Matrix from Uniform distribution:\n" << uniform_matrix << "\n";
  
  // Save a double matrix to a csv format
  uniform_matrix.save("examples_outdir/uniform_matrix.csv", arma::csv_ascii);

  // Armadillo can save directly to files or write to pre-opened streams.
  // In order to add column names to output tables we have to
  // write the header manually to a file stream and then save the matrix to the stream
  std::ofstream file("examples_outdir/uniform_matrix_with_headers.csv");
  file << "A,B,C,D,E" << std::endl;
  uniform_matrix.save(file, arma::csv_ascii);
  file.close();

  // As Armadillo objects are numerical structures the input should not contain
  // row/column names. Armadillo should be used only for heavy computations,
  // data should be processed in R/Python prior to the statistical modelling.
  arma::Mat<double> load_matrix;
  load_matrix.load("examples_outdir/uniform_matrix.csv", arma::csv_ascii);
  std::cout << "Loaded matrix:\n" << load_matrix << "\n";
}

//=======================================================================================

// A data strucutre to hold the input data required
// to calculate values of the objective function
class uniroot_BOX {
  public:
    double x,y;
    uniroot_BOX(double x_, double y_) :
      x(x_), y(y_) {}
};

// Objective function for the root finding procedure
// All the logic related to data access should be here
// The sygnature is fixed!
// Arg1: double x, the argument of the objective function
// Arg2: void*, a void pointer to the data structure
double uniroot_objF(double x, void* void_pointer_box){
  // De-reference the void pointer to a pointer to uniroot_BOX instance
  uniroot_BOX *pointer_box = static_cast<uniroot_BOX*>(void_pointer_box); 
  // Calculate the value of the objective function at a given argument
  // F(X) = 1/X + exp(X+1/x) - y
  double value = 1/x + exp(x+1/(*pointer_box).x) - (*pointer_box).y;
  return value;
}

/*
ROOT APROXIMATION FOR AN UNIVARIATE FUNCTION
//
Let's say we have to find roots for a big amount of equations of the form:
F(X) = 1/X + exp(X+1/x) - y
for a given data series x,y.
(any formula with no explicit analytical solution would be a good objective funciton)
We will utilize the bisection method combined with Newton-Raphson iteration
from R uniroot() function.
In statistical modelling it is useful for finding roots of partial derrivatives of the
Likelihood with respect to the parameters of the model.
*/
void root_search_univariate(){

  // Simulate the data
  const int len = 10;
  arma::vec x = arma::randi<arma::vec>(len, arma::distr_param(1, 10));
  arma::vec y = arma::randi<arma::vec>(len, arma::distr_param(100, 1000));
  arma::vec roots = arma::zeros<arma::vec>(len);

  // Parameters for the R_zeroin2() procedure
  double fx_upper, fx_lower;
  int max_iter;
  double eps; // Epsilon for the root approximation
  // Define the endpoints for the search
  double X_upper = 10.0;
  double X_lower = 1;
  // Put the data into the uniroot_BOX object
  uniroot_BOX box(0,0);
  void* void_pointer_box = &box;
  int bisection_product;

  // Find roots for every (x_i,y_i) separately:
  for (int i = 0; i < len; i++) {

    // Assign parameters to the box
    box.x = x[i];
    box.y = y[i];

    // Calculate the value of the objective function at the endpoints
    fx_upper = uniroot_objF(X_upper,void_pointer_box);
    fx_lower = uniroot_objF(X_lower,void_pointer_box);
    // Stop if the signs at the endpoints are the same
    bisection_product = fx_upper * fx_lower;
    try {
    if (bisection_product >= 0){
        throw "Incorrect range for bisection! Aborting.";
      }
    } catch (const char* msg) {
      std::cerr << msg << std::endl;
      return;
    }

    // R_zeroin2 procedure:
    // max_iter and eps get re-assigned!
    // They need to be reset after every iteration
    // The sygnature: endpoints, values at endpoints, pointer to the objective function,
    // void pointer to the data, references to max_iter and eps variables
    max_iter = 10000;
    eps = pow(10,-20);
    roots[i] = R_zeroin2(X_lower, X_upper, fx_lower, fx_upper, *uniroot_objF,
                        void_pointer_box, &eps, &max_iter);
  }

  roots.print("Univariate roots:");
}


//=======================================================================================

// A data strucutre to hold the input data and parameters required
// to calculate value of the objective function and its gradient
class optim_BOX {
  public:
    arma::mat xy;
    int index;
    optim_BOX(arma::mat xy_, int index_) :
      xy(xy_), index(index_) {}
};

// Objective function for the minimisation procedure
// All the logic related to data access should be here
// The sygnature is fixed!
// Arg1: int n, the number of arguments of the objective function
// Arg2: double*, a pointer to the vector with arguments to the objective function
// Arg3: void*, a void pointer to the data structure
double optim_objF(int n, double* args, void* void_pointer_box) {
  // De-reference the void pointer to a pointer to optim_BOX instance
  optim_BOX *pointer_box = static_cast<optim_BOX*>(void_pointer_box);
  int index = (*pointer_box).index;
  arma::mat xy = (*pointer_box).xy;
  double x = xy(index,0);
  double y = xy(index,1);
  // Calculate the value of the objective function at a given argument
  // F(X1,X2) = x^X1 + 2^(-X1) + y^X2 + 2^(-X2)
  double value = pow(x,args[0]) + pow(2,-args[0]) + pow(y,args[1]) + pow(2,-args[1]);
  return value;
}

// Gradient of the objective function for the minimisation procedure
// All the logic related to data access should be here
// The sygnature is fixed!
// Arg1: int n, the number of arguments of the objective function
// Arg2: double*, a pointer to the vector with arguments to the objective function
// Arg3: double*, a pointer to the vector with the gradient values
// Arg4: void*, a void pointer to the data structure
void grad_optim_objF(int n, double* args, double* grad, void* void_pointer_box) {
  // De-reference the void pointer to a pointer to optim_BOX instance
  optim_BOX *pointer_box = static_cast<optim_BOX*>(void_pointer_box);
  int index = (*pointer_box).index;
  arma::mat xy = (*pointer_box).xy;
  double x = xy(index,0);
  double y = xy(index,1);
  // Update the values of the gradient
  grad[0] = log(x)*pow(x,args[0]) - log(2)*pow(2,-args[0]);
  grad[1] = log(y)*pow(y,args[1]) - log(2)*pow(2,-args[1]);
}

/*
MINIMISATION OF A MULTIVARIATE FUNCTION
//
Let's say we are given a multivariate function to be minimized:
F(X1,X2) = x^X1 + 2^(-X1) + y^X2 + 2^(-X2)
for a given data series x,y.
(any complex enough formula would be a good objective funciton)
We will utilize the BFGS method from R optim() function.
In statistical modelling it is useful for minimisation of the negative Likelihood
of the data given a model
*/
void minimize_multivariate(){

  // Simulate the data
  const int n_functions = 5;
  const int n_args = 2;
  arma::vec x = arma::randi<arma::vec>(n_functions, arma::distr_param(2, 10));
  arma::vec y = arma::randi<arma::vec>(n_functions, arma::distr_param(2, 10));
  arma::mat xy(5,2);
  xy.col(0) = x;
  xy.col(1) = y;

  // Prepare structures for the results
  arma::mat args(n_functions,n_args);
  arma::vec min_Fx(n_functions);

  // Put the data into the optim_BOX object
  optim_BOX box(xy,-1);
  void* void_pointer_box = &box;

  // Arguments for the vmmin C function
  // These exact data structures are enforced by the sygnature of the C function
  double *b = vect(n_args);
  double zero_zero = 0.0;
  double *Fmin;
  int maxit = 1000;
  int trace = 0;
  int *mask;
  mask = (int *) calloc(n_args, sizeof(int));
  double abstol = 1e-16;
  double reltol = 1e-16;
  int nREPORT = 10;
  int zero = 0;
  int *fncount;
  int *grcount;
  int *fail;

  // Minimize for every (x_i,y_i) separately:
  for (int i = 0; i < n_functions; i++) {

    // Reset/reassign some variables after every iteration of the loop
    // as some of them get modified.
    box.index = i;
    Fmin = &zero_zero;
    for (int j = 0; j < n_args; j++) mask[j] = 1;
    fncount = &zero;
    grcount = &zero;
    fail = &zero;
    // Initiate starting points for the minimisation
    for (int j = 0; j < n_args; j++) b[j] = 0;

    // Call the BFGS implemented in C
    // The sygnature: endpoints, values at endpoints, pointer to the objective function,
    // void pointer to the data, references to max_iter and eps variables
    vmmin(n_args, b, Fmin, *optim_objF, *grad_optim_objF, maxit, trace, mask,
          abstol, reltol, nREPORT, void_pointer_box, fncount, grcount, fail);

    // Save the optimized results:
    min_Fx[i] = *Fmin;
    args(i,0) = b[0];
    args(i,1) = b[1];
  }

  free(mask);
  free(b);

  // Print results
  min_Fx.print("Minima:");
  args.print("Minima's args:");
}

//=======================================================================================

// A very simple data strucutre to hold the parameters required
// to aproximate values of the gradient of an objective function
class BOX_gradient {
  public:
    double* A;
    BOX_gradient(double* A_) :
      A(A_) {}
};

// Objective function for the gradient approximation procedure
// All the logic related to data access should be here
// The sygnature is fixed!
// Arg1: int n, the number of arguments of the objective function
// Arg2: double*, a pointer to the vector with arguments to the objective function
// Arg3: void*, a void pointer to the data structure
double gradient_objF(short unsigned n, double* args, void* void_pointer_box) {
  double result = 0.0;
  // De-reference the void pointer to a pointer to BOX_gradient instance
  BOX_gradient *pointer_box = static_cast<BOX_gradient*>(void_pointer_box);
  double* parameters = pointer_box->A;
  // Calculate the value of the function:
  for (unsigned short i=0; i<n; i+=1) result += parameters[i] * args[i];
  return result;
}

/*
GRADIENT APPROXIMATION
//
Let's say we are given a function and a point in which we want to
evaluate the gradient:
F(X_i,...) = SUM_i[A_i*X_i]
for a given argument (X_i,...)
We will utilise the gradient() function re-implemented from R numDeriv package to C
In statistical modelling it is useful to estimate the the values (at a given point)
of partial derrivatives the objective function that is too complex
to differentiate analyticaly.
*/
void approximate_gradient(){
  // Simulate the data
  double args_test[] = {1.0,1.0,1.0,1.0};
  double params_test[] = {10.0,-5.0,0.99,0.5};
  // Put the data into the BOX_gradient object
  BOX_gradient box(params_test);
  void* void_pointer_box = &box;
  // Calculate the gradient approximation
  double grad_test[] = {0.0,0.0,0.0,0.0};
  gradient(gradient_objF, 4, args_test, grad_test, void_pointer_box,
    1e-4, 1e-4, 2.220446e-16/7e-7, 4, 2); 
  printf("Gradient approximation:\n");
  printf("%f %f %f %f\n",
    *grad_test,*(grad_test+1),*(grad_test+2),*(grad_test+3));
}

//=======================================================================================

// A very simple data strucutre to hold the parameters required
// to aproximate values of the hessian of an objective function
class BOX_hessian {
  public:
    double C;
    BOX_hessian(double C_) :
      C(C_) {}
};

// Objective function for the hessian approximation procedure
// All the logic related to data access should be here
// The sygnature is fixed!
// Arg1: int n, the number of arguments of the objective function
// Arg2: double*, a pointer to the vector with arguments to the objective function
// Arg3: void*, a void pointer to the data structure
double hessian_objF(unsigned short n, double* args, void* void_pointer_box){
  // De-reference the void pointer to a pointer to BOX_hessian instance
  BOX_hessian *pointer_box = static_cast<BOX_hessian*>(void_pointer_box);
  double C = pointer_box->C;
  // Calculate the value of the function:
  double total = 0.0;
  for(int i = 0 ; i < n; ++i) {
      total += pow(args[i], i+1);
  }
  return total*C;
}

/*
INVERSE HESSIAN APPROXIMATION
//
Let's say we are given a function and a point in which we want to
evaluate the inverse of the hessian matrix:
F(X_i,...) = C * SUM_i X_i^i
for a given argument (X_i,...) and C in (0,1)
We will utilise the hessian() function re-implemented from R numDeriv package to C
In statistical modelling it is useful for estimation of the standard deviations
of the parameters of the model (negative inverse of the hessian of the
log-likelihood of the model at its optimum is the estimator of the asymptotic
covariance matrix of the parameters).
*/
void approximate_inv_hessian(){
  // Simulate the data
  double args_test[] = {10,10,10};
  double C = 0.5;
  // Put the data into the BOX_hessian object
  BOX_hessian box(C);
  void* void_pointer_box = &box;
  // Calculate the hessian approximation
  double H[3][3];
  hessian(hessian_objF, 3, args_test, (double *)H, void_pointer_box,
    1e-4, 1e-4, 2.220446e-16/7e-7, 4, 2);
  printf("Hessian approximation:\n");
  printf("%.15f %.15f %.15f\n",H[0][0],H[0][1],H[0][2]);
  printf("%.15f %.15f %.15f\n",H[1][0],H[1][1],H[1][2]);
  printf("%.15f %.15f %.15f\n",H[2][0],H[2][1],H[2][2]);
  // Inverse
  arma::mat arma_H(3,3);
  for (unsigned short i=0; i<3; i+=1){
    for (unsigned short j=0; j<3; j+=1){
      arma_H(i,j) = H[i][j];
    }
  }
  printf("Inverse Hessian approximation:\n");
  arma::mat inv_H = inv(arma_H);
  inv_H.print();
}

//=======================================================================================

int main(int argc, const char **argv){

  // Random numbers generator seed:
  arma::arma_rng::set_seed(0);
  // To set a random seed for every execution after a single compilation use:
  //arma::arma_rng::set_seed_random();

  // Creating a directory for output files
  if (mkdir("examples_outdir", 0777) == -1) 
    std::cerr << "Error: " << strerror(errno) << std::endl; 
  else
    std::cout << "Directory created"; 

  std::cout << "Armadillo version: " << arma::arma_version::as_string() << std::endl;

  armadillo_read_write_objects();

  armadillo_toy_examples();

  root_search_univariate();

  minimize_multivariate();

  approximate_gradient();

  approximate_inv_hessian();

  return 0;
}
