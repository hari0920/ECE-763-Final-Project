Contains Code for the Final Project of ECE 763-Computer Vision along with some personal testing and benchmarking of different frameworks.

MATLAB:
R2017a

>> MATLAB_Benchmark
Testing some linear algebra functions
Eig
Elapsed time is 130.926ms
Svd
Elapsed time is 439.839ms
Inv
Elapsed time is 056.979ms
Det
Elapsed time is 029.953ms
Dot
Elapsed time is 086.701ms
Done


Python: (Need to check timing method)
Using Numpy 1.14.1 with MKL  

 Function : Timing 
test_dot : 0.001053497 s
test_det : 0.000790123 s
test_svd : 0.000658436 s
test_eigenvalue : 0.000658436 s
test_inv : 0.000658436 s

C++:
Using Eigen with MKL

Eigenvalue
Finished in 57.0145ms 
SVD
Finished in 332.574ms
Inverse
Finished in 19.5994ms
Determinant
Finished in 31.9865ms 
Dot
Finished in 191.084ms 